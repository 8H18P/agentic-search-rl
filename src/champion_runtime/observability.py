"""Write-only JSONL observability for Champion core."""
from __future__ import annotations
import copy, json, time
from pathlib import Path
from typing import Any
class JsonlObserver:
    def __init__(self, path: str | Path, trajectory_id: str):
        self.path, self.trajectory_id = Path(path), trajectory_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
    def record_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        try:
            item={"event_type":event_type,"trajectory_id":self.trajectory_id,"timestamp":time.time(),"payload":copy.deepcopy(payload or {})}
            with self.path.open("a",encoding="utf-8") as f: f.write(json.dumps(item,ensure_ascii=False,default=str)+"\n"); f.flush()
        except Exception: pass
class NullObserver:
    def record_event(self, event_type, payload=None): pass
