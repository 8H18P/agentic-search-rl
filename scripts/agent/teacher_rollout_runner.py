#!/usr/bin/env python3
"""Hardened serial teacher-rollout runner.

Run directory layout (logs/teacher_rollout/<run_id>/):
  manifest.json         copy of the input manifest
  events.jsonl          AUTHORITATIVE per-trajectory events (question_id injected)
  results.jsonl         derived per-question index (append-only, fsync per line)
  completed_ids.json    atomic resume ledger  -> prevents duplicate re-roll
  summary.json          written at end
  stdout.log            captured by caller

Usage:
  scripts/teacher_rollout_runner.py --manifest M.jsonl --run-id pilot100_001 [--resume]

Reuses react_agent_with_sidecar + rl_agent.evaluator. Does NOT modify src/champion_runtime.
"""
from __future__ import annotations
import argparse, asyncio, json, sys, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from champion_runtime.agent_loop import react_agent_with_sidecar   # noqa: E402
from rl_agent.evaluator import aliases, canonicalize_final_answer, em_f1  # noqa: E402
from rollout_io import (  # noqa: E402
    create_run_dir, release_lock, RunObserver, ResultWriter,
    load_completed, update_completed, repair_results_tail,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSONL of questions (question + id + golden_answers)")
    ap.add_argument("--run-id", required=True, help="unique run dir name under logs/teacher_rollout/")
    ap.add_argument("--log-root", default=str(ROOT / "logs" / "teacher_rollout"))
    ap.add_argument("--resume", action="store_true", help="resume a crashed run (take over lock, skip completed)")
    args = ap.parse_args()

    manifest_rows = [json.loads(l) for l in Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not manifest_rows:
        raise SystemExit("empty manifest")

    run_dir = create_run_dir(Path(args.log_root) / args.run_id, resume=args.resume)
    manifest_dst = run_dir / "manifest.json"
    if not manifest_dst.exists():
        manifest_dst.write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # Repair any partial trailing line from a prior crash before appending.
    results_path = run_dir / "results.jsonl"
    repair_results_tail(results_path)
    completed = load_completed(run_dir)
    events_path = run_dir / "events.jsonl"
    rw = ResultWriter(results_path)
    results = []
    t_start = time.time()
    print(f"[run] {args.run_id} manifest={len(manifest_rows)} completed_already={len(completed)}", flush=True)
    try:
        for i, row in enumerate(manifest_rows, start=1):
            qid = str(row.get("id") or row.get("question_id") or i)
            if qid in completed:
                print(f"[skip] {qid} (already completed)", flush=True)
                continue
            question = row.get("question") or row.get("query") or row.get("input")
            if not question:
                raise SystemExit(f"row {i} has no question")
            traj = str(uuid.uuid4())
            sidecar = RunObserver(events_path, traj, qid)
            t0 = time.time()
            prediction = asyncio.run(react_agent_with_sidecar(question, sidecar))
            sidecar.close()
            gold = aliases(row.get("golden_answers", row.get("answer", row.get("gold_answer"))))
            eval_pred = canonicalize_final_answer(prediction)
            em, f1 = em_f1(eval_pred, gold)
            rec = {
                "question_id": qid, "source": row.get("source"), "question": question,
                "golden_answers": gold, "prediction": prediction,
                "evaluation_prediction": eval_pred, "answer_em": em, "answer_f1": f1,
                "runtime": "champion_core_edd28d_search_only", "retriever": "offline_e5",
                "latency": time.time() - t0, "trajectory_id": traj,
            }
            rw.append(rec)
            results.append(rec)
            update_completed(run_dir, qid)
            print(f"[done] {qid} em={em} f1={round(f1, 3)} ({i}/{len(manifest_rows)})", flush=True)
        rw.close()
        summary = {
            "run_id": args.run_id, "manifest": str(args.manifest),
            "num_manifest": len(manifest_rows), "num_completed": len(load_completed(run_dir)),
            "num_new_results": len(results), "elapsed_seconds": round(time.time() - t_start, 3),
            "results_file": "results.jsonl", "events_file": "events.jsonl",
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("RUN_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        release_lock(run_dir)


if __name__ == "__main__":
    main()
