"""Utilities
"""
import os
import sys
sys.path.append('../')

from relaxation_heuristics import hAddHeuristic
from pddlgym.structs import Predicate
from collections import namedtuple, defaultdict
from itertools import count
import abc
import tempfile
import time
import heapq as hq
import numpy as np


class PlanningFailure(Exception):
    pass


class SearchPlanner:
    """Base class for A*, best-first search
    """
    Node = namedtuple("Node", ["state", "parent", "action", "g"])

    def plan(self, state, goal, successor_fn, check_goal, heuristic, rng, 
             max_node_expansions = 1000, verbose=False, invalid_substates=None):
        queue = []
        state_to_best_g = defaultdict(lambda : float("inf"))

        root_node = self.Node(state=state, parent=None, action=None, g=0)
        root_h = heuristic(frozenset(root_node.state.literals))
        hq.heappush(queue, (self._get_priority(root_node.g, root_h),
                            rng.uniform(), root_node))
        num_expansions = 0
        loop = 0
        while len(queue) > 0 and num_expansions <= max_node_expansions:
            _, _, node = hq.heappop(queue)
            if state_to_best_g[node.state] < node.g:
                continue
            # If the goal holds, return
            if check_goal(node.state, goal):
                if verbose:
                    print("\nPlan found!")
                return self._finish_plan(node), {'node_expansions' : num_expansions}, {'done': True}
            num_expansions += 1
            if verbose:
                print(f"Expanding node {num_expansions}", end='\r', flush=True)
            # Generate successors
            for action, child_state, cost in successor_fn(node.state):
                # Check if the child state is invalid
                if invalid_substates:
                    is_valid = True
                    for invalid_substate in invalid_substates:
                        if check_goal(child_state, invalid_substate):
                            is_valid = False
                            break
                    if not is_valid:
                        continue
                # If we already found a better path to child, don't bother
                if state_to_best_g[child_state] <= node.g+cost:
                    continue
                # Add new node
                child_node = self.Node(state=child_state, parent=node, action=action,
                                       g=node.g+cost)
                child_h = heuristic(frozenset(child_node.state.literals))
                priority = self._get_priority(child_node.g, child_h)
                hq.heappush(queue, (priority, rng.uniform(), child_node))
                state_to_best_g[child_state] = child_node.g
            loop+=1
        
        return self._finish_plan(node), {'node_expansions' : num_expansions}, {'done': False}
        #raise PlanningFailure("Planning timed out!")
    
    def _finish_plan(self, node):
        plan = []
        while node.parent is not None:
            plan.append(node.action)
            node = node.parent
        plan.reverse()
        return plan

    @abc.abstractmethod
    def _get_priority(self, g, h):
        raise NotImplementedError("Override me!")


class AstarPlanner(SearchPlanner):
    """Planning with A* search.
    """
    def _get_priority(self, g, h):
        return (g + h, h)

