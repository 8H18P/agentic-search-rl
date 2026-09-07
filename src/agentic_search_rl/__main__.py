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
    sft_train = commands.add_parser("sft-train", help="Run formal SFT training")
    sft_train.add_argument("--config", required=True)
    dpo_train = commands.add_parser("dpo-train", help="Run formal canonical DPO training")
    dpo_train.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.command == "entrypoints":
        print("See docs/ENTRYPOINTS.md. SFT: HF/PEFT; DPO: native canonical runner; GRPO: TRL + Champion.")
        print("This CLI exposes SFT/DPO training; GRPO uses scripts/grpo/train.sh.")
    elif args.command == "score-answer":
        from agentic_search_rl.evaluation.metrics import em_f1
        em, f1 = em_f1(args.prediction, args.gold)
        print(json.dumps({"answer_em": em, "answer_f1": f1}))
    elif args.command == "sft-preflight":
        from agentic_search_rl.training.sft import preflight
        print(json.dumps(preflight(args.config), indent=2))
    elif args.command == "sft-train":
        from canonical_sft.runner import run
        print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
    else:
        from canonical_sft.dpo_runner import run
        print(json.dumps(run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
