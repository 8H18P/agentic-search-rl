"""Compatibility facade for online_grpo.adapter; original implementation remains authoritative."""
from importlib import import_module
import sys

# Alias the module, not copied functions: mutable runtime state must stay shared.
sys.modules[__name__] = import_module("online_grpo.adapter")
