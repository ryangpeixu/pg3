"""Improve upon a policy using a combination of GBFS and look-ahead (variation of bottom-up) search
"""
from .gbfs_approach import GBFSApproach
from mutation import DeleteCondition, DeleteRule, AddCondition, AddRule
from TTMutator import TTMutator

DEBUG = False


class LookAheadApproach(GBFSApproach):
    """Learn/improve a policy using top-down/look-ahead (bottomm-up) search
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._policy_mutators = [TTMutator(self._rng, self._train_score_function), AddCondition(self._rng), DeleteCondition(self._rng), DeleteRule(self._rng), AddRule(self._rng)]
