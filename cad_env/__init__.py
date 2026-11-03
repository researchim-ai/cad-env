"""CAD environments and verified learning data; optional interfaces load lazily."""
__version__ = '0.2.0'
__author__ = 'CAD Environment Team'
__all__ = ['CADEnvironment', 'CADAPI', 'LLMInterface', 'CADAgentEnv']


def __getattr__(name):
    if name == 'CADEnvironment':
        from .core import CADEnvironment
        return CADEnvironment
    if name == 'CADAPI':
        from .api import CADAPI
        return CADAPI
    if name == 'LLMInterface':
        from .llm_interface import LLMInterface
        return LLMInterface
    if name == 'CADAgentEnv':
        from .agent.environment import CADAgentEnv
        return CADAgentEnv
    raise AttributeError(name)
