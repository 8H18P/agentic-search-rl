"""Small public command surface. No training or network side effects on import/help."""
from __future__ import annotations

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("entrypoints", help="Show the public and developer entrypoint guide")
    score = commands.add_parser("score-answer", help="Existing alias-aware EM/F1, no model or API")
    score.add_argument("--prediction", required=True)
    score.add_argument("--gold", action="append", required=True, help="Repeat for answer aliases")
    preflight = commands.add_parser("sft-preflight", help="Explicit local model/dataset construction; no training")
    preflight.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.command == "entrypoints":
        print("See docs/ENTRYPOINTS.md. SFT: HF/PEFT; DPO: LLaMA-Factory bridge; GRPO: veRL target.")
        print("Only preflight and answer scoring are exposed here; this CLI never invokes training.")
    elif args.command == "score-answer":
        from agentic_search_rl.evaluation.metrics import em_f1
        em, f1 = em_f1(args.prediction, args.gold)
        print(json.dumps({"answer_em": em, "answer_f1": f1}))
    else:
        from agentic_search_rl.training.sft import preflight
        print(json.dumps(preflight(args.config), indent=2))


if __name__ == "__main__":
    main()
