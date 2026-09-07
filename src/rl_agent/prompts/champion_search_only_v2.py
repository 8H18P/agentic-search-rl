"""Champion-SearchOnly-v2 prompt derived from the vendored Champion prompt."""
from __future__ import annotations
import importlib.util
import re
from datetime import date
from pathlib import Path

_path = Path(__file__).resolve().parents[2] / "champion_agent" / "prompts.py"
_spec = importlib.util.spec_from_file_location("vendored_champion_prompts", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CHAMPION_SYSTEM_PROMPT = _mod.SYSTEM_PROMPT + str(date.today())
# The vendored template contains Visit only in its standalone JSON schema line.
CHAMPION_SEARCH_ONLY_USER_PROMPT_TEMPLATE = re.sub(
    r"\n\{\"name\": \"visit\".*?\}\n", "\n", _mod.USER_PROMPT_TEMPLATE, flags=re.S
)
if "visit" in CHAMPION_SEARCH_ONLY_USER_PROMPT_TEMPLATE.lower():
    raise AssertionError("Search-only prompt still contains Visit")
if "<tool_call_search>" in CHAMPION_SEARCH_ONLY_USER_PROMPT_TEMPLATE:
    raise AssertionError("Malformed tool-call example in prompt")
