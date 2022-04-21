import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, type = str)
    parser.add_argument("--approach", required=True, type = str)
    parser.add_argument("--train_score_function_name", required=True, type = str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--policy_file_name", required = False, type = str)
    args = parser.parse_args()
    return vars(args)
