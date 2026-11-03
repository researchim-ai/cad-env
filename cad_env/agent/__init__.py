"""Executable FreeCAD tasks, trajectories, and verifiable rewards."""

from .backend import FreeCADBackend
from .schema import ACTION_GUIDE, TOOLS, validate_action, validate_program
from .tasks import FAMILIES, MODES, make_task

__all__ = [
    "ACTION_GUIDE",
    "FAMILIES",
    "MODES",
    "TOOLS",
    "FreeCADBackend",
    "make_task",
    "validate_action",
    "validate_program",
]
