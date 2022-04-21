from types import SimpleNamespace

class GlobalSettings:
    train_successor_function_name = "policy_as_action2"
    num_iter_in_state = 50

    max_num_steps_train = 100
    num_epochs = 10
    num_learning_iters_per_epoch = 100
    size_penalty_weight = 1e-5
    plan_length_penalty = 0.
    temperature0 = 1e-3

    @staticmethod
    def get_arg_specific_settings(args):
        """A workaround for global settings that are
        derived from the experiment-specific args
        """
        arg_settings = {}

        if args["env"] == "trapnewspapers":
            arg_settings.update(dict(
                max_num_steps=200,
                train_env_name="trapnewspapers",
                train_problem_indices=[0, 1, 2, 3, 4],
                test_env_name="trapnewspapers_test",
                test_problem_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                operators_as_actions = True,
                dynamic_action_space = True,

            ))

        elif args["env"] == "hiking":
            arg_settings.update(dict(
                max_num_steps=100,
                train_env_name="hiking",
                train_problem_indices=[0, 1, 2, 3],
                test_env_name="hiking_test",
                test_problem_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                operators_as_actions = True,
                dynamic_action_space = True,
            ))

        elif args["env"] == "manymiconic":
            arg_settings.update(dict(
                max_num_steps=1000,
                train_env_name="manymiconic",
                train_problem_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,],
                test_env_name="manymiconic_test",
                test_problem_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                operators_as_actions = True,
                dynamic_action_space = True,
            ))

        elif args["env"] == "manygripper":
            arg_settings.update(dict(
                max_num_steps=1000,
                train_env_name="manygripper",
                train_problem_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                test_env_name="manygripper_test",
                test_problem_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                operators_as_actions = True,
                dynamic_action_space = True,

            ))
        
        elif args["env"] == "manyferry":
            arg_settings.update(dict(
                max_num_steps=1000,
                train_env_name="manyferry",
                train_problem_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                test_env_name="manyferry_test",
                test_problem_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                operators_as_actions = True,
                dynamic_action_space = True,
            ))

        elif args["env"] == "spannerlearning":
            arg_settings.update(dict(
                max_num_steps=1000,
                train_env_name="spannerlearning",
                train_problem_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,], 
                test_env_name="spannerlearning_test",
                test_problem_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                operators_as_actions = True,
                dynamic_action_space = True,

            ))

        else:
            raise Exception(f"Unrecognized env {args.env}")

        return arg_settings

def create_config(cmd_arg_dict):
    attr_to_value = {}

    # Generate arg-specific global settings
    attr_to_value.update(
        GlobalSettings.get_arg_specific_settings(cmd_arg_dict)
    )

    for d in [GlobalSettings.__dict__, cmd_arg_dict]:
        for attr, value in d.items():
            if attr.startswith("_"):
                continue
            if attr in attr_to_value:
                raise Exception("Attribute name overrides not allowed")
            attr_to_value[attr] = value

    Config = SimpleNamespace(**attr_to_value)

    return Config

