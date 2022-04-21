"""This file is lightly modified from pyperplan
See https://github.com/aibasel/pyperplan/blob/master/src/pyperplan/heuristics/relaxation.py
for the original file.
"""
from collections import namedtuple
from pddlgym.utils import get_object_combinations
from pddlgym.structs import Literal, LiteralConjunction
import heapq
import functools


# Utilities for creating ground operators from lifted ones
GroundOperator = namedtuple("GroundOperator", ["name", "preconds", "effects"])

def substitute(literals, assignments):
    """Substitute variables in literals with given dict of assignments.
    """
    new_literals = set()
    for lit in literals:
        new_vars = []
        for var in lit.variables:
            assert var in assignments
            new_vars.append(assignments[var])
        new_literals.add(lit.predicate(*new_vars))
    return new_literals

def create_ground_operators(operators, objects):
    """Create all possible ground operators with objects
    """
    ground_operators = []
    for operator in operators:
        for params in get_object_combinations(objects, len(operator.params),
            var_types=[o.var_type for o in operator.params]):
            subs = dict(zip(operator.params, params))
            preconds = substitute(operator.preconds.literals, subs)
            if isinstance(operator.effects, Literal):
                effects = substitute({operator.effects}, subs)
            else:
                effects = substitute(operator.effects.literals, subs)
            name = operator.name + "-" + "-".join([o.name for o in params])
            ground_operator = GroundOperator(name, preconds, effects)
            ground_operators.append(ground_operator)
    return ground_operators




""" This module contains the relaxation heuristics hAdd, hMax, hSA and hFF. """


class RelaxedFact:
    """This class represents a relaxed fact."""

    def __init__(self, name):
        """Construct a new relaxed fact.
        Keyword arguments:
        name -- the name of the relaxed fact.
        Member variables:
        name -- the name of the relaxed fact.
        precondition_of -- a list that contains all operators, this fact is a
                           precondition of.
        expanded -- stores whether this fact has been expanded during the
                    Dijkstra forward pass.
        distance -- stores the heuristic distance value
        sa_set -- stores a set of operators that have been applied to make this
                  fact True (only for hSA).
        cheapest_achiever -- stores the cheapest operator that was applied to
                             reach this fact (only for hFF).
        """
        self.name = name
        self.precondition_of = []
        self.expanded = False
        self.sa_set = None
        self.cheapest_achiever = None
        self.distance = float("inf")


class RelaxedOperator:
    """ This class represents a relaxed operator (no delete effects)."""

    def __init__(self, name, preconditions, add_effects):
        """Construct a new relaxed operator.
        Keyword arguments:
        name -- the name of the relaxed operator.
        preconditions -- the preconditions of this operator
        add_effects -- the add effects of this operator
        Member variables:
        name -- the name of the relaxed operator.
        preconditions -- the preconditions of this operator
        counter -- alternative method to check whether all preconditions are
                   True
        add_effects -- the add effects of this operator
        cost -- the cost for applying this operator
        """
        self.name = name
        self.preconditions = preconditions
        self.add_effects = add_effects
        self.cost = 1
        self.counter = len(preconditions)


class _RelaxationHeuristic:
    """This class is the base class for all relaxation heuristics.
    It is not meant to be instantiated. Nevertheless it is in principle an
    implementation of the hAdd heuristic.
    """

    def __init__(self, initial_state, goals, operators):
        """Construct a instance of _RelaxationHeuristic.
        """
        self.facts = dict()
        self.operators = []
        self.goals = goals
        self.init = initial_state
        self.tie_breaker = 0
        self.start_state = RelaxedFact("start")

        all_facts = self._derive_all_facts(initial_state, goals, operators)
        # Create relaxed facts for all facts in the task description.
        for fact in all_facts:
            self.facts[fact] = RelaxedFact(fact)

        for op in operators:
            # Relax operators and add them to operator list.
            add_effects = {effect for effect in op.effects if not effect.is_anti}
            ro = RelaxedOperator(op.name, op.preconds, add_effects)
            self.operators.append(ro)

            # Initialize precondition_of-list for each fact
            for var in op.preconds:
                self.facts[var].precondition_of.append(ro)

            # Handle operators that have no preconditions.
            if not op.preconds:
                # We add this operator to the precondtion_of list of the start
                # state. This way it can be applied to the start state. This
                # helps also when the initial state is empty.
                self.start_state.precondition_of.append(ro)

        #print("self.facts", self.facts, len(self.facts))

    def _derive_all_facts(self, initial_state, goals, operators):
        """Added for convenience!
        """
        all_facts = set(initial_state) | set(goals)
        for op in operators:
            all_facts |= set(op.preconds)
            add_effects = {effect for effect in op.effects if not effect.is_anti}
            all_facts |= add_effects
        return all_facts

    @functools.lru_cache(maxsize=None)
    def __call__(self, state): 
        """This function is called whenever the heuristic needs to be computed.
        Keyword arguments:
        state -- the current state
        """
        # Reset distance and set to default values.
        self.init_distance(state)

        # Construct the priority queue.
        heap = []
        # Add a dedicated start state, to cope with operators without
        # preconditions and empty initial state.
        heapq.heappush(heap, (0, self.tie_breaker, self.start_state))
        self.tie_breaker += 1

        for fact in state:
            # Its order is determined by the distance the facts.
            # As a tie breaker we use a simple counter.
            heapq.heappush(
                heap, (self.facts[fact].distance, self.tie_breaker, self.facts[fact])
            )
            self.tie_breaker += 1

        # Call the Dijkstra search that performs the forward pass.
        self.dijkstra(heap)

        # Extract the goal heuristic.
        h_value = self.calc_goal_h()

        return h_value

    def init_distance(self, state):
        """
        This function resets all member variables that store information that
        needs to be recomputed for each call of the heuristic.
        """

        def reset_fact(fact):
            fact.expanded = False
            fact.cheapest_achiever = None
            if fact.name in state:
                fact.distance = 0
                fact.sa_set = set()
            else:
                fact.sa_set = None
                fact.distance = float("inf")

        # Reset start state
        reset_fact(self.start_state)

        # Reset facts.
        for fact in self.facts.values():
            reset_fact(fact)

        # Reset operators.
        for operator in self.operators:
            operator.counter = len(operator.preconditions)

    def get_cost(self, operator, pre):
        """This function calculated the cost of applying an operator.
        For hMax and hAdd this nothing has to be changed here, but to use
        different functions for eval. hFF and hSA overwrite this function.
        """

        if operator.preconditions:
            # If this operator has preconditions, we sum / maximize over the
            # heuristic values of all preconditions.
            cost = self.eval(
                [self.facts[pre].distance for pre in operator.preconditions]
            )
        else:
            # If there are no preconditions for this operator, its cost is 0.
            cost = 0

        # The return value is a tuple, because in hSA instead of None, the
        # unioned set is returned.
        return None, cost + operator.cost

    def calc_goal_h(self):
        """This function calculates the heuristic value of the whole goal.
        As get_cost, it is makes use of the eval function, and has to be
        overwritten for hSA and hFF.
        If the goal is empty: Return 0
        """
        if self.goals:
            return self.eval([self.facts[fact].distance for fact in self.goals])
        else:
            return 0

    def finished(self, achieved_goals, queue):
        """
        This function is used as a stopping criterion for the Dijkstra search,
        which differs for different heuristics.
        """
        return achieved_goals == self.goals or not queue

    def dijkstra(self, queue):
        """This function is an implementation of a Dijkstra search.
        For efficiency reasons, it is used instead of an explicit graph
        representation of the problem.
        """
        # Stores the achieved subgoals. Needed for abortion criterion of hMax.
        achieved_goals = set()
        while not self.finished(achieved_goals, queue):
            # Get the fact with the lowest heuristic value.
            (_dist, _tie, fact) = heapq.heappop(queue)
            # If this node is part of the goal, we add to the goal set, which
            # is used as an abort criterion.
            if fact.name in self.goals:
                achieved_goals.add(fact.name)
            # Check whether we already expanded this fact.
            if not fact.expanded:
                # Iterate over all operators this fact is a precondition of.
                for operator in fact.precondition_of:
                    # Decrease the precondition counter.
                    operator.counter -= 1
                    # Check whether all preconditions are True and we can apply
                    # this operator.
                    if operator.counter <= 0:
                        for n in operator.add_effects:
                            neighbor = self.facts[n]
                            # Calculate the cost of applying this operator.
                            (unioned_sets, tmp_dist) = self.get_cost(operator, fact)
                            if tmp_dist < neighbor.distance:
                                # If the new costs are cheaper, then the old
                                # costs, we change the neighbors heuristic
                                # values.
                                neighbor.distance = tmp_dist
                                neighbor.sa_set = unioned_sets
                                neighbor.cheapest_achiever = operator
                                # And push it on the queue.
                                heapq.heappush(
                                    queue, (tmp_dist, self.tie_breaker, neighbor)
                                )
                                self.tie_breaker += 1
                # Finally the fact is marked as expanded.
                fact.expanded = True


class hAddHeuristic(_RelaxationHeuristic):
    """This class is an implementation of the hADD heuristic.
    It derives from the _RelaxationHeuristic class.
    """

    def __init__(self, *args, **kwargs):
        """
        To make this class an implementation of hADD, apart from deriving from
        _RelaxationHeuristic,  we only need to set eval to sum().
        """
        super().__init__(*args, **kwargs)
        self.eval = sum
