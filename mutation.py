import abc
import itertools
import functools
from policy import OrderedDecisionList
from rule import Rule
from scoring import flatten_plan, PrimitiveMatchScoringFunction, IntegratedMatchScoringFunction, DirectPolicyEvaluationScoreFunction
from utils import wrap_goal_literal, get_successor_state
from pddlgym.structs import Not, Literal, LiteralConjunction, LiteralConjunction, ground_literal
from pddlgym.core import _compute_new_state_from_lifted_effects

class PolicyMutator:

    def __init__(self, rng):
        self._rng = rng

    def get_successors(self, policy):
        for args in self._get_all_successor_args(policy):
            yield self._get_successor_from_args(policy, *args)

    def get_random_successor(self, policy):
        all_args = list(self._get_all_successor_args(policy))
        if len(all_args) == 0:
            return policy
        args = all_args[self._rng.choice(len(all_args))]
        return self._get_successor_from_args(policy, *args)

    @abc.abstractmethod
    def _get_all_successor_args(self, policy):
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _get_successor_from_args(self, policy, *args):
        raise NotImplementedError("Override me!")


class AddCondition(PolicyMutator):
    """Adds a condition to a random rule in a policy
    """
    def _get_all_successor_args(self, policy):
        if len(policy.rules) == 0:
            return

        for rule_idx in range(len(policy.rules)):
            available_vars = {var_type : set() for var_type in policy.env.domain.types.values()} # Maps var_type to set of vars of that type
            # Find all existing variables and store by var_type (Assumes params is well-defined)
            for var in policy.rules[rule_idx].params:
                available_vars[var.var_type].add(var)

            # Choose a condition 
            possible_predicates = [cond for cond_name, cond in policy.env.domain.predicates.items() \
                                   if cond_name not in policy.env.domain.actions]

            for pred in possible_predicates:
                for variables in self._get_possible_variables(pred.var_types, available_vars):
                    for pos_or_neg in ["pos", "neg"]:
                        for goal_or_nongoal in ["goal", "nongoal"]:
                            #Skip redundant or contradictory conditions
                            cond = pred(*variables)
                            if goal_or_nongoal == "goal":
                                if cond in policy.rules[rule_idx].unwrapped_goal_conds or Not(cond) in policy.rules[rule_idx].unwrapped_goal_conds:
                                    continue
                            else:
                                if cond in policy.rules[rule_idx].preconds or Not(cond) in policy.rules[rule_idx].preconds:
                                    continue
                            yield (rule_idx, pred, variables, pos_or_neg, goal_or_nongoal)

    def _get_possible_variables(self, var_types, available_vars):
        num_new_vars = {var_type: 0 for var_type in available_vars.keys()}
        new_vars = {var_type: set() for var_type in available_vars.keys()}
        all_vars = available_vars.copy()
        for var_type in var_types:
            num_new_vars[var_type] += 1
        for var_type, num_var_type in num_new_vars.items():
            for i in range(num_var_type):
                var_name = "new" + str(var_type) + "-" + str(i)
                all_vars[var_type].add(var_type(var_name))
        choices = [sorted(all_vars.get(vt, [])) for vt in var_types]
        for choice in itertools.product(*choices):
            if len(choice) != len(set(choice)):
                continue
            yield choice

    def _get_successor_from_args(self, policy, rule_idx, pred, variables,
                                 pos_or_neg, goal_or_nongoal):                        
        new_policy = policy.copy()
        cond = pred(*variables)

        assert pos_or_neg in ["pos", "neg"]
        if pos_or_neg == "neg":
            cond = Not(cond)

        assert goal_or_nongoal in ["goal", "nongoal"]
        if goal_or_nongoal == "goal":
            new_policy.rules[rule_idx].add_goal_condition(cond)
        else:
            new_policy.rules[rule_idx].preconds.add(cond)

        return new_policy


class AddRule(PolicyMutator):
    """Adds a rule (with a random action and its preconditions) to a copy of the policy
    """
    def _get_all_successor_args(self, policy):
        # Choose action for rule and getting necessary preconds/goal conds
        operators = policy.env.domain.operators
        for operator_name in operators:
            operator = operators[operator_name]

            # Choosing rule idx
            for rule_idx in range(len(policy.rules)+1):
                yield (operator, rule_idx)

    def _get_successor_from_args(self, policy, operator, rule_idx):
        # Get action from operator
        action = None
        action_predicates = policy.env.action_predicates
        for precond in operator.preconds.literals:
            if precond.predicate in action_predicates:
                action = precond

        if action is None:
            assert policy.env.domain.operators_as_actions
            for action_predicate in action_predicates:
                if action_predicate.name == operator.name:
                    action = action_predicate(*operator.params)

        # Initialize using operator preconditions
        preconds = set(operator.preconds.literals)
        goal_conds = set()

        # Choosing rule name
        existing_rule_names = {rule.name for rule in policy.rules}
        rule_name_suffix = 0
        rule_name = None
        while "rule-" + str(rule_name_suffix) in existing_rule_names:
            rule_name_suffix += 1
        rule_name = "rule-"+str(rule_name_suffix) 

        new_policy = policy.copy()
        new_rule = Rule(policy.env, rule_name, preconds, goal_conds, action)
        new_policy.rules.insert(rule_idx, new_rule)
        return new_policy


class DeleteRule(PolicyMutator):
    """Returns copy of policy with a random rule deleted
    """
    def _get_all_successor_args(self, policy):
        if len(policy.rules) <= 1:
            return
        for i in range(len(policy.rules)):
            yield (i,)

    def _get_successor_from_args(self, policy, rule_idx):
        new_policy = policy.copy()
        del new_policy.rules[rule_idx]
        return new_policy


class DeleteCondition(PolicyMutator):
    """Returns copy of policy with a random condition in a random rule of a policy deleted
    Note: Only deletes excessive conditions - those not found in preconditions of rule action 
    """
    def _get_all_successor_args(self, policy):
        if len(policy.rules) == 0:
            return

        for rule_idx in range(len(policy.rules)):
            required_preconds = self._get_required_preconds(policy.env.domain.operators[policy.rules[rule_idx].action.predicate.name], policy.rules[rule_idx].action)
            # Remove precond
            for precond in sorted(set(policy.rules[rule_idx].preconds) - required_preconds):
                yield (rule_idx, precond, None)
            # Remove goal cond
            for goal in sorted(policy.rules[rule_idx].unwrapped_goal_conds):
                yield (rule_idx, None, goal)

    def _get_successor_from_args(self, policy, rule_idx, precond, goal_cond):
        assert precond is None or goal_cond is None
        new_policy = policy.copy()
        if precond is not None:
            new_policy.rules[rule_idx].preconds.remove(precond)
        else:
            assert goal_cond is not None
            new_policy.rules[rule_idx].remove_goal(goal_cond)

        if not is_well_formed(new_policy.rules[rule_idx]): #If not well-formed, delete rule
            del new_policy.rules[rule_idx]
        
        return new_policy

    @functools.lru_cache(maxsize = None)
    def _get_required_preconds(self, operator, action):
        assignments = {operator.params[i]: action.variables[i] for i in range(len(operator.params))}
        required_preconds = {ground_literal(precond, assignments) for precond in operator.preconds.literals}
        return required_preconds


# Helper functions
def mutate(policy, rng):
    mutators = [AddRule(rng), AddCondition(rng), DeleteRule(rng), DeleteCondition(rng)]
    mutator = mutators[rng.choice(len(mutators))]
    return mutator.get_random_successor(policy)

def is_well_formed(rule):
    required_variables = set(rule.params)
    cond_variables = set() 
    for precond in rule.preconds:
        for var in precond.variables:
            cond_variables.add(var)

    for goal_cond in rule.wrapped_goal_conds:
        for var in goal_cond.variables:
            cond_variables.add(var)

    return required_variables.issubset(cond_variables)
