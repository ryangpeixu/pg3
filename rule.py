import pddlgym
import re
import os
from copy import copy
from pddlgym.parser import PDDLParser, Operator, parse_plan_step
from pddlgym.inference import find_satisfying_assignments
from pddlgym.structs import Literal, LiteralConjunction
from utils import wrap_goal_literal, get_all_ground_literals
from functools import lru_cache

RULE_STR = """\t(:rule {name}
\t\t:parameters {parameters}
\t\t:preconditions {preconditions}
\t\t:goals {unwrapped_goals}
\t\t:action {action}
\t)
"""

class RuleNotApplicable(Exception):
    pass

class Rule:
    """Class to hold an rule."""
    def __init__(self, env, name, preconds, goals, action):
        self.env = env #PDDLGym Env
        self.name = name  #string

        #self.preconds is a set of literals 
        self.preconds = preconds
                
        #self.unwrapped_goal_conds and self.wrapped_goal_conds are sets of literals
        self.unwrapped_goal_conds = goals 
        self.wrapped_goal_conds = {wrap_goal_literal(lit) for lit in goals} 

        self.action = action #structs.Literal object of action

    def __str__(self):
        return "Name:" + str(self.name) + \
            "\nParams: " + str(self.params) + \
            "\nPreconds: " + str(sorted(self.preconds)) + \
            "\nWrapped Goal Conditions: " + str(sorted(self.wrapped_goal_conds)) + \
            "\nAction: " + str(self.action)

    @property
    def params(self):
        all_lits = set(self.preconds) | self.wrapped_goal_conds | {self.action}
        return sorted({v for lit in all_lits for v in lit.variables})

    def get_identifier(self):
        return (tuple(self.params), self.action, frozenset(self.preconds),
                frozenset(self.wrapped_goal_conds))

    def get_action(self, state):
        return self._get_action_helper(state, frozenset(self.preconds),
            frozenset(self.wrapped_goal_conds), self.action, self.env.domain)

    @classmethod
    @lru_cache(maxsize = None)
    def _get_action_helper(cls, state, preconds, wrapped_goal_conds, action, domain):
        # Note: no more all_ground_literals
        if isinstance(state.goal, pddlgym.structs.Literal):
            kb = state.literals | {wrap_goal_literal(state.goal)}
        else:
            kb = state.literals | set(wrap_goal_literal(lit) for lit in state.goal.literals)

        conds = preconds | wrapped_goal_conds
        assignments = find_satisfying_assignments(
            kb, conds,
            mode='csp',
            constants=domain.constants,
            type_to_parent_types=domain.type_to_parent_types,
            max_assignment_count=1)

        if not assignments:
            raise RuleNotApplicable()

        # Finish action
        assert len(assignments) == 1
        assignment = assignments[0]
        objs = []
        for v in action.variables:
            try:
                obj = assignment[v]
            except KeyError:
                # Assumes that all variables must be bound
                raise RuleNotApplicable()
            objs.append(obj)

        return action.predicate(*objs)
   
    def is_valid(self, state):
        try:
            _ = self.get_action(state)
            return True
        except RuleNotApplicable:
            return False

    def __copy__(self):
        return Rule(self.env, self.name, self.preconds.copy(), self.unwrapped_goal_conds.copy(), self.action)

    def add_goal_condition(self, cond):
        self.unwrapped_goal_conds.add(cond)
        self.wrapped_goal_conds.add(wrap_goal_literal(cond))

    def remove_goal(self, cond):
        self.unwrapped_goal_conds.remove(cond)
        self.wrapped_goal_conds.remove(wrap_goal_literal(cond))

    def pddl_str(self):
        """
        Returns string of rule in PDDL-like format
        """
        assert isinstance(self.action, pddlgym.structs.Literal), "self Action is not a single pddlgym.structs.Literal"

        if len(self.preconds) == 1:
            preconditions = next(iter(self.preconds)).pddl_str()
        else: 
            preconditions = pddlgym.structs.LiteralConjunction(list(self.preconds)).pddl_str()

        if len(self.unwrapped_goal_conds) == 1:
            unwrapped_goals = next(iter(self.unwrapped_goal_conds)).pddl_str()
        else: 
            unwrapped_goals = pddlgym.structs.LiteralConjunction(list(self.unwrapped_goal_conds)).pddl_str()
        
        parameters = "(" + " ".join([str(param).replace(":", " - ") for param in self.params]) + ")"

        return RULE_STR.format(name = self.name, parameters = parameters, preconditions = preconditions, unwrapped_goals = unwrapped_goals, action = self.action.pddl_str())

#Modified from PDDLGym
class RuleParser(PDDLParser):
    def __init__(self, policy_fname, env, constants = None):
        self.policy_fname = policy_fname 
        self.domain_name = env.domain.domain_name
        self.types = env.domain.types
        self.predicates = env.domain.predicates
        self.action_names = env.domain.actions
        self.env = env
        self.uses_typing = not ("default" in self.types)
        self.constants = constants or []
        with open(policy_fname, "r") as f:
            self.policy= f.read().lower()
        self._parse_policy()

    def _parse_policy(self):
        patt = r"\(policy(.*?)\)"
        self.policy_name = re.search(patt, self.policy).groups()[0].strip()
        self._parse_rules()

    def _parse_rules(self):
        matches = re.finditer(r"\(:rule", self.policy)
        self.rules = {}
        for match in matches:
            start_ind = match.start()
            rule = self._find_balanced_expression(self.policy, start_ind).strip()
            patt = r"\(:rule(.*):parameters(.*):preconditions(.*):goals(.*):action(.*)\)"
            rule_match = re.match(patt, rule, re.DOTALL)
            rule_name, params, preconds, goals, action = rule_match.groups()

            rule_name = rule_name.strip()

            params = params.strip()[1:-1].split("?")

            if self.uses_typing:
                params = [(param.strip().split("-", 1)[0].strip(),
                           param.strip().split("-", 1)[1].strip())
                          for param in params[1:]]
                params = [self.types[v]("?"+k) for k, v in params]
            else:
                params = [param.strip() for param in params[1:]]
                params = [self.types["default"]("?"+k) for k in params]
             
            if preconds.strip() == '()':
                preconds = LiteralConjunction([])
            else:
                preconds = self._parse_into_literal(preconds.strip(), params + self.constants)
                if isinstance(preconds, Literal): #If only one literal, convert to LiteralConjunction
                    preconds = LiteralConjunction([preconds])

            if goals.strip() == '()':
                goals = LiteralConjunction([])
            else:
                goals = self._parse_into_literal(goals.strip(), params + self.constants)
                if isinstance(goals, Literal): #If only one literal, convert to LiteralConjunction 
                    goals = LiteralConjunction([goals])

            if action.strip()  == '()':
                action = "(trans)" #If no action, set action string to be trans - for transition
            else:
                action = action.strip() 
                action_as_literal = self._parse_into_literal(action, params + self.constants)

            self.rules[rule_name] = Rule(self.env, rule_name, {lit for lit in preconds.literals}, {lit for lit in goals.literals}, action_as_literal)
