#Stores different representations of policies (currently, FSC and decision list)
import os
import functools
import random
from rule import RuleParser, RuleNotApplicable
from copy import copy
from random import sample

POLICY_STR = """(define (policy {NAME})
{RULES}
)
"""

class Policy():
    pass

class DecisionList(Policy):

    def __init__(self):
        self.policy_file_name = None
        self.env = None
        self.rules = None

    def create_hardcoded_policy_from_file(self, file_name, env):
        """Creates policy from file_name
        Args:
            file_name (str): file location of ordered list of rules
            env (PDDLEnv): PDDLEnv, already called with make to a domain
        """
        self.policy_file_name = file_name
        self.env = env
        dir_path = os.path.dirname(os.path.realpath(__file__))
        policy_file = os.path.join(dir_path, file_name)
        self.rules = list(RuleParser(policy_file, env).rules.values())

    def reset(self, env):
        self.env = env
        self.rules = []

    def get_size(self):
        size = 0
        for rule in self.rules:
            size += len(rule.preconds) + len(rule.wrapped_goal_conds)
            size += len(rule.params)
        return size

    def pddl_str(self):
        """Returns string of policy in PDDL-like format
        """
        return POLICY_STR.format(NAME = "temp_policy", RULES = "".join([rule.pddl_str() for rule in self.rules]))
    
    def write(self, file_name):
        """Writes policy in PDDL-like format to file_name
        """
        with open(file_name, 'w') as f:
            f.write(self.pddl_str())

class OrderedDecisionList(DecisionList):
    def __init__(self):
        super().__init__()
    
    def get_called_rule_id(self, state):
        called_rule_id = None
        for i in range(len(self.rules)):
            if self.rules[i].is_valid(state):
                called_rule_id = i
                break
        return called_rule_id

    @functools.lru_cache(maxsize=None)
    def get_action(self, state):
        for rule in self.rules:
            try:
                return rule.get_action(state)
            except RuleNotApplicable:
                continue
        return None
        
    def sample_action(self, state, rng=None):
        called_rule_id = self.get_called_rule_id(state)
        if called_rule_id == None:
            return None
        action_frequencies = self.rules[called_rule_id].get_action_frequencies(state)
        all_actions = [action for action in action_frequencies.keys() for i in range(action_frequencies[action])]
        if len(all_actions) == 0:
            return None
        if rng is None:
            return random.sample(all_actions, 1)[0]
        return all_actions[rng.choice(len(all_actions))]
    
    def get_action_probability(self, state, action):
        called_rule_id = self.get_called_rule_id(state)
        if called_rule_id == None:
            return 0.0 
        action_frequencies = self.rules[called_rule_id].get_action_frequencies(state)
        if action not in action_frequencies.keys():
            return 0.0
        else:
            total_num_actions = sum(action_frequencies.values())
            return action_frequencies[action]/total_num_actions

    def __copy__(self):
        copied_rules = [copy(rule) for rule in self.rules]
        copied_dl = OrderedDecisionList()
        copied_dl.policy_file_name = self.policy_file_name
        copied_dl.env = self.env
        copied_dl.rules = copied_rules
        return copied_dl

    def copy(self):
        return self.__copy__()

    def __str__(self):
        return "Ordered Decision List\n" + "\n".join(str(rule) for rule in self.rules)
