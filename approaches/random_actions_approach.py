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

    def get_action(self, state, env=None):
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
        try:
            act = env.action_space.sample(state)
            print(act)
            return act
        except ValueError as e:  # dead-end
            assert "a must be a positive integer unless no samples are taken" in str(e)
        return None

