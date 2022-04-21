import abc
import itertools
import functools
from policy import OrderedDecisionList
from rule import Rule
from scoring import flatten_plan, PrimitiveMatchScoringFunction, IntegratedMatchScoringFunction, DirectPolicyEvaluationScoreFunction
from utils import wrap_goal_literal, get_successor_state
from pddlgym.structs import Not, Literal, LiteralConjunction, LiteralConjunction, ground_literal
from pddlgym.core import _compute_new_state_from_lifted_effects
from mutation import PolicyMutator

DEBUG = False

class TTMutator(PolicyMutator):
    def __init__(self, rng, train_score_function):
        super().__init__(rng)
        self._train_score_function = train_score_function

        #Add preprocessed ground action to preconds
        self._ground_action_to_pos_preconds = {}
        self._ground_action_to_neg_preconds = {}
        for prob_idx in self._train_score_function.problem_indices:
            self._train_score_function.env.fix_problem_index(prob_idx)
            temp_state, _ = self._train_score_function.env.reset()
            self._train_score_function.env.action_space._update_objects_from_state(temp_state)
            self._ground_action_to_pos_preconds[prob_idx] = self._train_score_function.env.action_space._ground_action_to_pos_preconds.copy()
            self._ground_action_to_neg_preconds[prob_idx] = self._train_score_function.env.action_space._ground_action_to_neg_preconds.copy()

    @functools.lru_cache(maxsize = None)
    def _get_ground_action_to_preconds(self, prob_idx, action):
        """Returns pair of pos_preconds and neg_preconds for a ground action
        """
        if action in self._ground_action_to_pos_preconds[prob_idx]:
            return self._ground_action_to_pos_preconds[prob_idx][action], self._ground_action_to_neg_preconds[prob_idx][action]
        else:
            raise Exception("Can not find pos/neg preconds from ground action")

    def _get_all_successor_args(self, policy):
        yield tuple()  # Only one way to do TT mutation

    def _get_successor_from_args(self, policy):
        #Find problem with longest plan and failed action
        kept_prob_idx = None
        shortest_plan_len = float("inf")
        kept_unflattened_plan = None
        kept_flattened_plan = None
        kept_problem_goals = None
        for prob_idx in self._train_score_function.problem_indices:
            if isinstance(self._train_score_function, PrimitiveMatchScoringFunction):
                unflattened_plan, _, result = self._train_score_function.run_primitive_search(prob_idx)
            elif isinstance(self._train_score_function, DirectPolicyEvaluationScoreFunction):
                unflattened_plan, _, result = self._train_score_function.run_primitive_search(prob_idx)
            elif isinstance(self._train_score_function, IntegratedMatchScoringFunction):
                unflattened_plan, _, result = self._train_score_function.run_integrated_search(policy, prob_idx)
            if not result["done"]: #If not done, skip
                continue
            #Step through plan and determine if contains failed action
            flattened_plan = flatten_plan(unflattened_plan)
            env = self._train_score_function.env
            env.fix_problem_index(prob_idx)
            current_state, _ = env.reset()
            for i in range(len(flattened_plan)):
                plan_action = flattened_plan[i]
                policy_action = policy.get_action(current_state)
                #Keep the longest plan with a failed action
                if plan_action != policy_action and len(flattened_plan) < shortest_plan_len:
                    kept_prob_idx = prob_idx
                    shortest_plan_len = len(flattened_plan)
                    kept_unflattened_plan = unflattened_plan
                    kept_flattened_plan = flattened_plan
                    kept_problem_goals = set(current_state[2].literals)
                current_state = get_successor_state(current_state, plan_action, env.domain)
        if kept_prob_idx is None: #Perfect policy
            return policy
        
        action_profiles = self._get_action_profiles(kept_prob_idx, kept_flattened_plan, policy)
        goal_indices = self._get_established_goal_indices(kept_problem_goals, action_profiles)
        triangle_table = TriangleTable(env, kept_prob_idx, action_profiles)

        #Find relevant indices of actions for each established goal ("directly-contributes" - recursive definition)
        failed_action_index = self._find_last_failed_action_index(action_profiles)
        closest_est_goal_index = shortestBFS(failed_action_index, triangle_table.reversed_adjacency_list, goal_indices.values())
        
        if closest_est_goal_index == None:
            return policy

        if failed_action_index == closest_est_goal_index:
            relevant_action_indices = []
        else:
            relevant_action_indices = [i for i in range(len(triangle_table.adjacency_list)) if distanceBFS(closest_est_goal_index, failed_action_index, triangle_table.adjacency_list)[i] is not None and i != failed_action_index]
        failed_action_profile = action_profiles[failed_action_index]
        est_goal = action_profiles[closest_est_goal_index].effects & kept_problem_goals
        relevant_action_profiles = [action_profiles[i] for i in relevant_action_indices]

        if DEBUG: 
            print("Plan", kept_flattened_plan, "\n")
            print("Failed action index", failed_action_index, "\n")
            print("Relevant action indices", relevant_action_indices, "\n")
            print("-----------")
            print("Previous Policy")
            print(policy)
            print("-----------")

        policy = policy.copy()
        policy = self._induce_policy_update(policy, failed_action_profile, relevant_action_profiles, est_goal)

        if DEBUG:
            print("--------------")
            print("After Policy")
            print(policy)
            print("--------------")
            new_score = self._train_score_function(policy, debug = True)
            print("New Score:", new_score)
            import ipdb; ipdb.set_trace()
        return policy 

    def _induce_policy_update(self, policy, failed_action_profile, relevant_action_profiles, est_goal):
        """est_goal = set of goal literals satisfied in the established goal"""
        rule_goal_conds = est_goal
        rule_preconds = failed_action_profile.pos_preconds | failed_action_profile.neg_preconds #Always include failed action preconds
        for goal_cond in est_goal:
            rule_preconds.add(Not(goal_cond))

        ordered_action_profiles = [relevant_action_profile for relevant_action_profile in relevant_action_profiles]
        all_valid_preconds = set()
        #Forming the "context": a smaller state to work with
        for action_profile in ordered_action_profiles:
            all_valid_preconds = all_valid_preconds | (failed_action_profile.state[0] & action_profile.pos_preconds)
        #Initialize important variables to be those in failed action and in goal condition
        important_variables = {var for goal_cond in est_goal for var in goal_cond.variables} | {var for var in failed_action_profile.variables} #Those in the goal literal and failed action
        #Initialize rule preconds to be those in valid preconds/context and all variables are important
        rule_preconds = rule_preconds | {precond for precond in all_valid_preconds if {var for var in precond.variables}.issubset(important_variables)}

        #Finding unique rule name
        existing_rule_names = {rule.name for rule in policy.rules}
        rule_name_suffix = 0
        rule_name = None
        while "TT-rule-" + str(rule_name_suffix) in existing_rule_names:
            rule_name_suffix += 1
        rule_name = "TT-rule-"+str(rule_name_suffix) 

        #Continue adding actions from ordered action profiles until we the right action
        rule = Rule(policy.env, rule_name, rule_preconds, rule_goal_conds, failed_action_profile.action)
        most_overfit_rule = rule

        if rule.get_action(failed_action_profile.state) == failed_action_profile.action:
            new_policy = policy.copy()
            new_policy = self._insert_rule_in_last_valid(new_policy, failed_action_profile, rule)
            return new_policy

        for i in range(len(ordered_action_profiles)):
            action_profile_variables = set(ordered_action_profiles[i].variables)
            if len(action_profile_variables - important_variables) == 0: #If no new variables, skip
                continue
            
            important_variables = important_variables | action_profile_variables
            rule_preconds = rule_preconds | {precond for precond in all_valid_preconds if {var for var in precond.variables}.issubset(important_variables)}
            rule = Rule(policy.env, rule_name, rule_preconds, rule_goal_conds, failed_action_profile.action)
            if i == len(ordered_action_profiles)-1:
                most_overfit_rule = rule
            try:
                if rule.get_action(failed_action_profile.state) == failed_action_profile.action:
                    new_policy = policy.copy()
                    new_policy = self._insert_rule_in_last_valid(new_policy, failed_action_profile, rule)
                    return new_policy
            except:
                pass
        
        #If not found, return the most overfit
        new_policy = policy.copy()
        new_policy = self._insert_rule_in_last_valid(new_policy, failed_action_profile, most_overfit_rule)
        return new_policy
    
    def _insert_rule_in_last_valid(self, policy, failed_action_profile, rule):
        for idx, existing_rule in enumerate(policy.rules):
            # Does this rule apply?
            if existing_rule.is_valid(failed_action_profile.state):
                break
        else:
            idx = len(policy.rules)

        # Add in the rule
        policy.rules.insert(idx, rule)
        return policy

    def _find_last_failed_action_index(self, action_profiles):
        for i in range(len(action_profiles)-1, -1, -1):
            if action_profiles[i].is_failed:
                return i

    def _get_established_goal_indices(self, problem_goals, action_profiles):
        goal_indices = {}
        for i in range(len(action_profiles)-1, -1, -1):
            action_profile = action_profiles[i]
            current_state = action_profile.state
            unsatisfied_goal_conditions = problem_goals - current_state[0]
            for unsatisfied_goal_condition in unsatisfied_goal_conditions:
                if unsatisfied_goal_condition not in goal_indices.keys():
                    goal_indices[unsatisfied_goal_condition] = i
        return goal_indices

    def _get_action_profiles(self, prob_idx, flattened_plan, policy):
        action_profiles = []
        env = self._train_score_function.env
        env.fix_problem_index(prob_idx)
        current_state, _ = env.reset()
        for i in range(len(flattened_plan)):
            plan_action = flattened_plan[i]
            policy_action = policy.get_action(current_state)
            plan_action_pos_preconds, plan_action_neg_preconds = self._get_ground_action_to_preconds(prob_idx, plan_action)
            plan_action_ground_effects = self._get_action_ground_effects(prob_idx, plan_action)
            action_profile = TTActionProfile(plan_action, policy_action, current_state, plan_action.variables, plan_action_pos_preconds, plan_action_neg_preconds, plan_action_ground_effects)
            action_profiles.append(action_profile)
            current_state = get_successor_state(current_state, plan_action, env.domain)
        return action_profiles

    @functools.lru_cache(maxsize = None)
    def _get_action_ground_effects(self, prob_idx, action):
        action_variables = action.variables
        action_effects = set() 
        for name, operator in self._train_score_function.env.domain.operators.items():
            if name.lower() == action.predicate.name.lower():
                selected_operator = operator
        if isinstance(selected_operator.effects, LiteralConjunction):
            effects = selected_operator.effects.literals
        else:
            assert isinstance(selected_operator.effects, Literal)
            effects = [selected_operator.effects]
        
        assert len(action.variables) == len(selected_operator.params)
        assignment = {selected_operator.params[i]: action.variables[i] for i in range(len(selected_operator.params))}
        for effect in effects:
            action_effects.add(ground_literal(effect, assignment))
        return action_effects

class TTActionProfile():
    def __init__(self, action, policy_action, state, variables, pos_preconds, neg_preconds, effects):
        self.action = action
        self.policy_action = policy_action
        if action != policy_action:
            self.is_failed = True
        else:
            self.is_failed = False
        self.state = state
        self.variables = variables
        self.pos_preconds = pos_preconds
        self.neg_preconds = neg_preconds
        self.effects = effects

    def __str__(self):
        return "\tAction:"+ str(self.action) + "\nPos preconds"+ str(sorted(self.pos_preconds)) + "\nNeg preconds"+ str(sorted(self.neg_preconds)) + "\nEffects"+ str(sorted(self.effects))

class TriangleTable():
    def __init__(self, env, prob_idx, profiled_plan):
        self.env = env
        self.env.fix_problem_index(prob_idx)
        self.init_state, _ = env.reset()
        self.problem_goals = set(self.env._goal.literals)
        self.prob_idx = prob_idx
        self.profiled_plan = profiled_plan

        self.n = len(profiled_plan)
        self.triangle_table = [[" " for _ in range(self.n + 1)] for _ in range(self.n + 1)]
        self.marked_init = [] #Marked clauses for operations (marked_init[i] = marked clauses for operation i)
        self.adjacency_list = [[] for _ in range(self.n)] #Note: this is a DAG directed backwards (e.g. for goal regression)
        self.reversed_adjacency_list = [[] for _ in range(self.n)] #Note: this is a DAG directed forwards (e.g. for goal finding)

        self.fill_column_zero()
        self.fill_delta_and_action()

    def fill_column_zero(self):
        current_state = set(self.init_state[0])
        init_leftover = [None for i in range(self.n+1)] #init_leftover[i] = leftovers after operation i-1 (so first entry is just the initial state)
        init_leftover[0] = current_state
        for i in range(len(self.profiled_plan)):
            action = self.profiled_plan[i].action
            current_marked_clauses = current_state & self.profiled_plan[i].pos_preconds #Note, this won't take care of negative effects
            self.marked_init.append(current_marked_clauses)
            self.triangle_table[i][0] = InitSquare(current_state - current_marked_clauses, current_marked_clauses)
            #Only want to deletions, we don't care about additions
            for effect in self.profiled_plan[i].effects:
                if effect.is_anti:
                    literal = effect.inverted_anti
                    if literal in current_state:
                        current_state.remove(literal)
            init_leftover[i+1] = current_state

    def fill_delta_and_action(self):
        for i in range(len(self.profiled_plan)):
            action = self.profiled_plan[i].action
            self.triangle_table[i][i+1] = ActionSquare(action) #Action Square
            curr_action_effects = self.profiled_plan[i].effects

            #Filter out anti action effects
            anti_action_effects = set()
            for effect in curr_action_effects:
                if effect.is_anti:
                    anti_action_effects.add(effect)
            curr_action_effects = curr_action_effects - anti_action_effects

            for j in range(i+1, len(self.profiled_plan)):
                new_action = self.profiled_plan[j].action
                new_action_pos_preconds = self.profiled_plan[j].pos_preconds
                marked_clauses = new_action_pos_preconds & curr_action_effects
                if len(marked_clauses) > 0:
                    self.adjacency_list[j].append((i, marked_clauses))
                    self.reversed_adjacency_list[i].append((j, marked_clauses))
                self.triangle_table[j][i+1] = DeltaSquare(curr_action_effects - marked_clauses, marked_clauses)
                for effect in self.profiled_plan[j].effects:
                    if effect.is_anti:
                        literal = effect.inverted_anti
                        if literal in curr_action_effects:
                            curr_action_effects.remove(literal) #Note, this won't take care of negative effects of the original action
            marked_clauses = self.problem_goals & curr_action_effects #Different from paper!
            self.triangle_table[self.n][i+1] = DeltaSquare(curr_action_effects - marked_clauses, marked_clauses)

    def display(self):
        cell_text = [[str(obj) for obj in self.triangle_table[i]] for i in range(len(self.triangle_table))]
        table = plt.table(cellText = cell_text, loc = "center", cellLoc = "center")
        table.auto_set_font_size(False)
        table.set_fontsize(5)
        table.scale(1, 2)
        plt.axis('off')
        plt.show()


class TriangleTableSquare():
    def __init__(self):
        self.prob_idx = None
        self.prob_goals = None

class DeltaSquare(TriangleTableSquare):
    def __init__(self, unmarked_clauses, marked_clauses):
        self.unmarked_clauses = unmarked_clauses 
        self.marked_clauses = marked_clauses

    def __str__(self):
        entire_str = "Marked:\n"
        for marked_clause in self.marked_clauses:
            entire_str += str(marked_clause) + "\n"
        
        entire_str += "Unmarked\n"
        for unmarked_clause in self.unmarked_clauses:
            entire_str += str(unmarked_clause) + "\n"
        return entire_str

class InitSquare(TriangleTableSquare):
    def __init__(self, unmarked_clauses, marked_clauses):
        self.leftovers = unmarked_clauses 
        self.marked_clauses = marked_clauses

    def __str__(self):
        entire_str = "Marked:\n"
        for marked_clause in self.marked_clauses:
            entire_str += str(marked_clause) + "\n"
        return entire_str

class ActionSquare(TriangleTableSquare):
    def __init__(self, action):
        self.action =  action

    def __str__(self):
        return str(self.action)

def distanceBFS(start, block, adjacency_list):
    """Returns list mapping index to shortest distance
    Should be used from an established goal (start) to find directly-contributing actions
    block = failing action (and should not search from that node)
    """
    distances = [None for _ in range(len(adjacency_list))]
    queue = [(start, 0)]
    while len(queue) != 0:
        node, distance = queue.pop(0)
        if distances[node] is not None: #If already visited
            continue
        distances[node] = distance
        for neighbor, marked_clauses in adjacency_list[node]:
            if neighbor > block:
                queue.append((neighbor, distance+1))
    return distances

def shortestBFS(start, adjacency_list, goals):
    distances = [None for _ in range(len(adjacency_list))]
    queue = [(start, 0)]
    while len(queue) != 0:
        node, distance = queue.pop(0)
        if distances[node] is not None: #If already visited
            continue
        if node in goals:
            return node
        distances[node] = distance
        for neighbor, marked_clauses in adjacency_list[node]:
            queue.append((neighbor, distance+1))
    return None

