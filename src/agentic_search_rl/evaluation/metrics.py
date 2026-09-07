"""Compatibility facade for rl_agent.evaluator; original implementation remains authoritative."""
from importlib import import_module
import sys

# Alias the module, not copied functions: mutable runtime state must stay shared.
sys.modules[__name__] = import_module("rl_agent.evaluator")
