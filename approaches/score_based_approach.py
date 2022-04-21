"""Improve upon a policy by searching to maximize a score function
"""
from tqdm import tqdm
from .base_approach import BaseApproach
from scoring import generate_score_function
from policy import OrderedDecisionList
import abc
import numpy as np



class ScoreBasedApproach(BaseApproach):
    """Learn/improve a policy using a scoring function
    """

    def __init__(self, config, train_env, train_problem_indices, seed):
        super().__init__(config, train_env, train_problem_indices, seed)
        # Set up policy
        self._policy = OrderedDecisionList()
        if config.policy_file_name:
            self._policy.create_hardcoded_policy_from_file(config.policy_file_name, train_env)
        else:
            self._policy.reset(train_env)
        # Common rng
        self._rng = np.random.RandomState(self._seed)        
        # Initialize scoring function
        self._train_score_function = generate_score_function(config.train_score_function_name,
            train_env, config.train_successor_function_name, train_problem_indices,
            self._rng, config.num_iter_in_state, config.max_num_steps)
        # Create generator that yields every time a new policy is considered
        self._policy_generator = self._create_policy_generator()

    def learn(self, num_iters):
        """Learning continues running policy search for a certain number of iters
        """
        for _ in tqdm(range(num_iters)):
            next(self._policy_generator)
        print("Best policy found so far:")
        print(self._policy)

    @abc.abstractmethod
    def _create_policy_generator(self):
        """Yields every time a new policy is considered.
        Keeps self._policy updated with the best policy found so far.
        """
        raise NotImplementedError("Override me!")

    def get_action(self, state, env = None):
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
        return self._policy.get_action(state)
