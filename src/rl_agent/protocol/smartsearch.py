"""Native SmartSearch protocol adapter; deliberately independent of Champion."""
import re
from dataclasses import dataclass

@dataclass
class SmartAction:
    kind: str
    value: str = ""
    strict_protocol_valid: bool = False
    logical_action_valid: bool = False

def parse(text: str) -> SmartAction:
    search = re.search(r"<search>\s*(.*?)\s*</search>", text, re.S | re.I)
    answer = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.S | re.I)
    first = min([x.start() for x in (search, answer) if x] or [10**9])
    if search and search.start() == first and search.group(1).strip():
        return SmartAction("search", search.group(1).strip(), True, True)
    if answer and answer.start() == first and answer.group(1).strip():
        return SmartAction("answer", answer.group(1).strip(), True, True)
    return SmartAction("invalid")

def first_action(text: str) -> str:
    match = re.search(r"(?:<search>.*?</search>|<answer>.*?</answer>)", text, re.S | re.I)
    return text[:match.end()] if match else text
