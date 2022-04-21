import sys
sys.path.append('../')

import abc
import functools
import pddlgym
import numpy as np
from policy import DecisionList, OrderedDecisionList
from settings import GlobalSettings
from copy import copy, deepcopy
import random
import matplotlib.pyplot as plt
from utils import get_successor_state, get_all_ground_literals
from pddlgym.inference import check_goal
from search_planners import SearchPlanner, AstarPlanner
from relaxation_heuristics import create_ground_operators, hAddHeuristic
from collections.abc import Iterable

# Scoring Functions

class ScoringFunction:
    """A scoring function takes a policy and produces a score
    """
    def __init__(self, env, successor_fn_name, problem_indices, rng, num_iter, max_num_steps,
                 size_penalty_weight=GlobalSettings.size_penalty_weight):
        self.env = env
        self.successor_fn_name = successor_fn_name
        self.problem_indices = problem_indices
        self._rng = rng
        self.num_iter = num_iter
        self.max_num_steps = max_num_steps
        self.size_penalty_weight = size_penalty_weight

    def __call__(self, policy, debug=False):
        score = self._get_score(policy, debug=debug)
        assert -1 <= score <= 0
        # Add size penalty
        policy_size = policy.get_size()
        score -= self.size_penalty_weight * policy_size
        # Clip to -1, 0 range
        return np.clip(score, -1, 0)

    @abc.abstractmethod
    def _get_score(self, policy, debug=False):
        raise NotImplementedError("Override me!")

    @functools.lru_cache(maxsize=None)
    def run_integrated_search(self, policy, prob_idx):
        assert prob_idx in self.problem_indices
        self.env.fix_problem_index(prob_idx)
        self.env.reset()
        
        hAdd = self.generate_hAdd_heuristic(prob_idx)

        return integrated_search(self.env, policy, self.successor_fn_name, hAdd, self._rng, self.num_iter)

    @functools.lru_cache(maxsize=None)
    def run_primitive_search(self, prob_idx):
        self.env.fix_problem_index(prob_idx)
        self.env.reset()
        hAdd = self.generate_hAdd_heuristic(prob_idx)
        return integrated_search(self.env, None, "primitive_actions_only", hAdd, self._rng, self.num_iter)

    @functools.lru_cache(maxsize=None)
    def generate_hAdd_heuristic(self, prob_idx):
        """Generates hAdd heuristic for a given problem index with caching
        """
        self.env.fix_problem_index(prob_idx)
        self.env.reset()

        initial_state = self.env.get_state()
        ground_operators = create_ground_operators(list(self.env.domain.operators.values()), initial_state.objects)
        if isinstance(initial_state.goal, pddlgym.structs.Literal):
            hAdd = hAddHeuristic(initial_state.literals, [initial_state.goal], ground_operators)
        else:
            hAdd = hAddHeuristic(initial_state.literals, initial_state.goal.literals, ground_operators)
        
        return hAdd



class DirectPolicyEvaluationScoreFunction(ScoringFunction):
    """Scores policy by directly running the policy on the problems
    """
    def _get_score(self, policy, debug=False):
        results = [self.evaluate_policy(policy, prob_idx) \
                   for prob_idx in self.problem_indices]
        return min(results)

    def evaluate_policy(self, policy, prob_idx):
        self.env.fix_problem_index(prob_idx)
        state, _ = self.env.reset()
        for _ in range(self.max_num_steps):
            action = policy.get_action(state)
            if action is None:
                return -1.
            state, reward, done, _ = self.env.step(action)
            if done:
                return 0.
        return -1.


class MatchingPlannerScoringFunction(ScoringFunction):
    """A parent class for scoring functions that use the probability that the policy
    assigns to actions in some plans
    """

    def _get_score(self, policy, debug=False):
        plan_scores = [self.get_score_for_plan(policy, plan, prob_idx, debug=debug) \
                       for (plan, prob_idx) in self.get_plans(policy)]
        if len(plan_scores) == 0:
            return -1.
        score = float(min(plan_scores))
        return score

    @abc.abstractmethod
    def get_plans(self, policy):
        """Collect plans that will be used for scoring
        """
        raise NotImplementedError("Override me!")

    def get_score_for_plan(self, policy, plan, prob_idx, debug=False):
        """Give a score for a single plan
        """
        if debug:
            print("Scoring with plan:")
            print(plan)
        # Give up early if no plan was found
        if not plan[2]['done']:
            return -1.
        # Flatten plan
        actions = flatten_plan(plan[0])
        action_scores = []
        self.env.fix_problem_index(prob_idx)
        current_state, _ = self.env.reset()
        for action in actions:
            action_score = self.get_score_for_action(policy, current_state, action)
            action_scores.append(action_score)
            current_state = get_successor_state(current_state, action, self.env.domain)
        if debug:
            print("------")
            groups = []
            group_scores = []
            curr_group = []
            match = action_scores[0]
            for i in range(len(actions)):
                action = actions[i]
                action_score = action_scores[i]
                if action_score == match:
                    curr_group.append(action)
                else:
                    group_scores.append(match)
                    groups.append(curr_group.copy())
                    curr_group = [action]
                    match = action_score
            groups.append(curr_group)
            group_scores.append(match)
            for i in range(len(groups)):
                group_score = group_scores[i]
                if group_score == 0:
                    print("Policy:", groups[i])
                else:
                    print("Primitives:", groups[i])
            print()

        if len(action_scores) == 0:
            return -1.
        match_score = min(action_scores)
        score = match_score - len(action_scores) * GlobalSettings.plan_length_penalty
        score = np.clip(score, -1, 0)
        return score

    @abc.abstractmethod
    def get_score_for_action(self, policy, state, action):
        """Give a score for a single action in a state
        """
        raise NotImplementedError("Override me!")


class CanonicalMatchScoringFunction(MatchingPlannerScoringFunction):

    def get_score_for_action(self, policy, state, action):
        """Check if the policy would actually return this action in this state
        """
        policy_action = policy.get_action(state)
        """
        if action == policy_action:
            print("Comparison: Plan", action, "Policy:", policy_action, "Score: 0")
        else:
            print("Comparison: Plan", action, "Policy:", policy_action, "Score: -1")
        """
        if action == policy_action:
            return 0.
        return -1.


class PrimitiveMatchScoringFunction(MatchingPlannerScoringFunction):

    def get_plans(self, policy):
        """Use primitive plans
        """
        for prob_idx in self.problem_indices:
            plan = self.run_primitive_search(prob_idx)
            yield (plan, prob_idx)


class IntegratedMatchScoringFunction(MatchingPlannerScoringFunction):

    def get_plans(self, policy):
        """Use integrated plans
        """
        for prob_idx in self.problem_indices:
            plan = self.run_integrated_search(policy, prob_idx)
            yield (plan, prob_idx)


class PrimitiveCanonicalMatchScoringFunction(PrimitiveMatchScoringFunction,CanonicalMatchScoringFunction):
    pass

class IntegratedCanonicalMatchScoringFunction(IntegratedMatchScoringFunction,CanonicalMatchScoringFunction):
    pass


def generate_score_function(score_fn_name, env, successor_fn_name, problem_indices, rng, num_iter, max_num_steps):
    score_fn_cls = {
        "direct_policy_eval": DirectPolicyEvaluationScoreFunction,
        "primitive_canonical_match": PrimitiveCanonicalMatchScoringFunction,
        "integrated_canonical_match": IntegratedCanonicalMatchScoringFunction,
    }[score_fn_name]

    return score_fn_cls(env, successor_fn_name, problem_indices, rng, num_iter, max_num_steps)


# Helper functions
def flatten_plan(policy_plan):
    """Only flattens plan *Note: should not have Policy markers
    Converts from list (possible containing tuples) to a flattened list
    """
    plan = []
    for act in policy_plan:
        if isinstance(act, tuple):
            for inner_action in act:
                plan.append(inner_action)
        else:
            plan.append(act)
    return plan

def integrated_search(env, policy, successor_fn_name, heuristic, rng, num_iter = 1):
    """
    Performs a search for a plan with different options for how the policy is integrated. Returns a plan.
    """
    
    def primitive_actions_only_successor_fn(obs):
        possible_actions = get_all_ground_literals(env, obs) 
        for action in possible_actions:
            yield (action, get_successor_state(obs, action, env.domain), 1)
    
    def policy_as_action2_successor_fn(obs):
        # The difference is that each time the policy takes an action, a child is yielded
        policy_actions = tuple()
        policy_child_state = obs
        policy_action = None
        for i in range(num_iter):
            possible_actions = get_all_ground_literals(env, policy_child_state)
            policy_action = policy.get_action(policy_child_state)
            if policy_action == None or policy_action not in possible_actions:
                break
            policy_actions = policy_actions + (policy_action,)
            policy_child_state = get_successor_state(policy_child_state, policy_action, env.domain) 

            # This is the difference!
            yield (policy_actions, policy_child_state, 0)

            if check_goal(policy_child_state, policy_child_state[2]): #Checks whether reached goal - should never be called in first iteration of loop
                break
        
        possible_actions = get_all_ground_literals(env, obs)
        for action in possible_actions:
            yield (action, get_successor_state(obs, action, env.domain), 1)

    successor_fns = {"primitive_actions_only": primitive_actions_only_successor_fn,
                     "policy_as_action2": policy_as_action2_successor_fn,}

    assert heuristic is not None, "heuristic should be a _RelaxedHeuristic object"
    sp = AstarPlanner()
    initial_state = env.get_state()
    return sp.plan(initial_state, env._goal, successor_fns[successor_fn_name], check_goal, heuristic, rng)
