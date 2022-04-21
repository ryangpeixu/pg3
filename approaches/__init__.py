from .random_actions_approach import RandomActionsApproach
from .gbfs_approach import GBFSApproach
from .look_ahead import LookAheadApproach

def create_approach(config, train_env, train_problem_indices,
                    seed):
    approach_cls = {
        "random_actions" : RandomActionsApproach,
        "gbfs": GBFSApproach,
        "look_ahead": LookAheadApproach,
    }[config.approach]
    return approach_cls(config, train_env, train_problem_indices, seed)
