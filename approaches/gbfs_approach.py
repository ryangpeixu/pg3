"""Improve upon a policy using greedy best-first search
"""
from .score_based_approach import ScoreBasedApproach
from mutation import AddCondition, AddRule
from collections import defaultdict
import heapq as hq
import itertools
import numpy as np


DEBUG = False


class GBFSApproach(ScoreBasedApproach):
    """Learn/improve a policy using greedy best-first search
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._policy_mutators = [AddCondition(self._rng), AddRule(self._rng)]
        self._queue = [(0, -self._train_score_function(self._policy), self._rng.uniform(), self._policy)]
        self._semantic_level_states = None
        self._semantic_id_counts = defaultdict(lambda : itertools.count())

    def _create_policy_generator(self):
        best_score = self._train_score_function(self._policy)
        visited = set()

        while True:
            policy_prio1, policy_prio2, _, policy = hq.heappop(self._queue)
            if DEBUG:
                print("\n\npopped policy")
                print(policy)

            for new_policy in self._get_child_policies(policy):
                new_policy_id = self._get_policy_identifier(new_policy)
                if new_policy_id in visited:
                    continue
                visited.add(new_policy_id)

                new_score = self._train_score_function(new_policy, debug=False)

                semantic_level = self._get_policy_semantic_level(new_policy)
                hq.heappush(self._queue,
                    (semantic_level, -new_score, self._rng.uniform(), new_policy))
                yield new_policy
                # Record best policy
                if new_score > best_score:
                    self._policy = new_policy
                    best_score = new_score
                    
                    if DEBUG:
                        print("Found a new best policy:")
                        print(self._policy)
                        print("score:", best_score)
                        print("Reevaluating scoring function in debug mode...")
                        self._train_score_function(new_policy, debug=True)
                        import ipdb; ipdb.set_trace()

                    # Optimization: add back parent so we can continue expanding
                    # it later if necessary, but greedily start expanding the
                    # better found child now
                    hq.heappush(self._queue,
                        (policy_prio1, policy_prio2, self._rng.uniform(), policy))
                    break

    def _get_child_policies(self, policy):
        for mutator in self._policy_mutators:
            yield from mutator.get_successors(policy)

    def _get_policy_identifier(self, policy):
        return tuple(rule.get_identifier() for rule in policy.rules)

    def _get_policy_semantic_level(self, policy):
        if self._semantic_level_states is None:
            self._semantic_level_states = set()
            for problem_idx in self._cf.train_problem_indices:
                plan = self._train_score_function.run_primitive_search(problem_idx)[0]
                self._train_env.fix_problem_index(problem_idx)
                state, _ = self._train_env.reset()
                for act in plan:
                    self._semantic_level_states.add(state)
                    state, _, _, _ = self._train_env.step(act)
            self._semantic_level_states = sorted(self._semantic_level_states)
        semantic_id = tuple(policy.get_action(s) for s in self._semantic_level_states)
        return next(self._semantic_id_counts[semantic_id])
