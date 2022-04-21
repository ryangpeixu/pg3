"""Just take random actions
"""
from .base_approach import BaseApproach

class RandomActionsApproach(BaseApproach):
    """Just take random actions
    """
    def learn(self, num_iters):
        """Doesn't learn anything
        """
        pass

    def get_action(self, state):
        """
        Parameters
        ----------
        state : State
            From PDDLGym structs
        Returns
        -------
        action : Literal
            From PDDLGym structs
        """
        return self._action_space.sample(state)

