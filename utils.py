import pddlgym
import functools

def wrap_goal_literal(x):
    """Helper for converting a state to required input representation
    """
    if isinstance(x, pddlgym.structs.LiteralConjunction):
        wrapped_body = [wrap_goal_literal(lit) for lit in x.literals]
        return pddlgym.structs.LiteralConjunction(wrapped_body)
    if isinstance(x, pddlgym.structs.ForAll):
        wrapped_body = wrap_goal_literal(x.body)
        return pddlgym.structs.ForAll(wrapped_body, x.variables, is_negative=x.is_negative)
    if isinstance(x, pddlgym.structs.Predicate):
        return pddlgym.structs.Predicate("WANT"+x.name, x.arity, var_types=x.var_types,
                         is_negative=x.is_negative, is_anti=x.is_anti)
    assert isinstance(x, pddlgym.structs.Literal)
    new_predicate = wrap_goal_literal(x.predicate)
    return new_predicate(*x.variables)

@functools.lru_cache(maxsize=None)
def get_successor_state(*args, **kwargs):
    return pddlgym.core.get_successor_state(*args, **kwargs)

@functools.lru_cache(maxsize=None)
def get_all_ground_literals(env, obs):
    """Returns the action_space with caching
    """
    if env.action_space._initial_state.objects != obs.objects:
        env.action_space.reset_initial_state(obs)

    return sorted(env.action_space.all_ground_literals(obs))
