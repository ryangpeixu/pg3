"""General approach for generalized policy learning
"""
import abc

class BaseApproach:
    """General approach for generalized policy learning
    """
    def __init__(self, config, train_env, train_problem_indices, seed):
        """
        Parameters
        ----------
        train_env : PDDLGymEnv
        """
        self._cf = config
        self._train_env = train_env
        self._train_problem_indices = train_problem_indices
        self._action_space = train_env.action_space
        self._seed = seed

    @abc.abstractmethod
    def learn(self, num_iters):
        """Learn something from the train environment
        Parameters
        ----------
        num_iters : int
            Number of learning iterations
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
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
        raise NotImplementedError("Override me!")
