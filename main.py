"""Main pipeline
"""

import sys
sys.path.append("../")

import random
import numpy as np
import pickle
import pddlgym
import os
import time
from approaches import create_approach
from policy import OrderedDecisionList
from args import parse_args
from settings import create_config


def run_single_seed(test_env, test_problem_indices, approach,
                    seed, num_epochs=5,
                    num_learning_iters_per_epoch=10,
                    max_num_steps=100):
    """Run approach on train environment and evaluate
    on test environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    # Track all results over the course of learning
    all_results = []
    # Run initial evaluation
    result = evaluate_approach(test_env, test_problem_indices,
                               approach, seed,
                               max_num_steps=max_num_steps)
    all_results.append(result)
    if approach.__class__.__name__ == "RandomActionsApproach":
        return all_results
    # Allow the approach to learn
    for epoch in range(num_epochs):
        print(f"Starting learning epoch {epoch}")
        # Learn for a certain number of iterations
        approach.learn(num_learning_iters_per_epoch)
        # Evaluate the approach by running it in the test env
        result = evaluate_approach(test_env, test_problem_indices,
                                   approach, seed,
                                   max_num_steps=max_num_steps)
        all_results.append(result)
    return all_results

def evaluate_approach(test_env, test_problem_indices, approach,
                      seed, max_num_steps=100):
    """Run the approach's policy in the test environment
    """
    successes = 0
    num_problems = len(test_problem_indices)
    # Seed the test environment
    test_env.seed(seed)
    # Loop over test problems
    for idx in test_problem_indices:
        print('Evaluating problem', idx)
        test_env.fix_problem_index(idx)
        state, _ = test_env.reset()
        # Run for some maximum number of steps
        for _ in range(max_num_steps):
            action = approach.get_action(state, env=test_env)
            if action == None:
                break
            state, reward, done, _ = test_env.step(action)
            # If done, reward > 0, then approach succeeded
            if done:
                assert reward > 0
                successes += 1
                break
    # Return fraction of successes
    result = successes / float(num_problems)
    print("Evaluation results:", result)
    return result

if __name__ == "__main__":
    start_time = time.time()

    args = parse_args()
    config = create_config(args)

    print("Full config:")
    print(config)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    pddl_dir = os.path.join(dir_path, "problems")

    train_env = pddlgym.core.PDDLEnv(os.path.join(pddl_dir, config.train_env_name + ".pddl"), os.path.join(pddl_dir, config.train_env_name + "/seed{}".format(config.seed)), operators_as_actions = config.operators_as_actions, dynamic_action_space = config.dynamic_action_space)
    test_env = pddlgym.core.PDDLEnv(os.path.join(pddl_dir, config.train_env_name + ".pddl"), os.path.join(pddl_dir, config.test_env_name + "/seed{}".format(config.seed)), operators_as_actions = config.operators_as_actions, dynamic_action_space = config.dynamic_action_space)

    approach = create_approach(config, train_env, config.train_problem_indices, config.seed)
    
    all_results = run_single_seed(test_env, config.test_problem_indices, approach, config.seed, config.num_epochs, config.num_learning_iters_per_epoch, config.max_num_steps)

    print("All results:", all_results)
    print("Universal time:", time.time() - start_time)

    # Save results
    if not os.path.exists("results/"):
        os.makedirs("results/")
    filename = f"{config.env}__{config.approach}__{config.train_score_function_name}__{config.seed}.p"
    outfile = os.path.join("results", filename)
    with open(outfile, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Dumped results to {outfile}.")

