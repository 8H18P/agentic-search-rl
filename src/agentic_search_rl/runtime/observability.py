"""Compatibility facade for champion_runtime.observability; original implementation remains authoritative."""
from importlib import import_module
import sys

# Alias the module, not copied functions: mutable runtime state must stay shared.
sys.modules[__name__] = import_module("champion_runtime.observability")
