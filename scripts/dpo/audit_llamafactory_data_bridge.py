#!/usr/bin/env python3
"""CPU-only real canonical token parity plus upstream source-function check.

Does not import a trainer, load model weights, run backward, or call a service.
This deliberately does NOT certify real-model DPO parity.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from canonical_sft.dpo import CanonicalDPODataset, _tokenize_messages, sequence_logps
from canonical_sft.llamafactory_dpo import CanonicalLlamaFactoryCollator
from canonical_sft.trl_dpo import CanonicalTRLDataCollator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--lf-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    before = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    dataset = CanonicalDPODataset(args.dataset)
    if not len(dataset):
        raise ValueError("empty canonical dataset")
    trl = CanonicalTRLDataCollator(tokenizer, args.max_length)
    lf = CanonicalLlamaFactoryCollator(tokenizer, args.max_length)
    records = []
    for row in dataset:
        metadata = row["metadata"]
        assert metadata["policy_identity_match"] is True
        assert metadata["raw_model_output_used_as_action_truth"] is False
        assert metadata["validation400_used"] is False
        a, b = trl([row]), lf([row])
        assert torch.equal(a["input_ids"], b["input_ids"])
        assert torch.equal(a["attention_mask"], b["attention_mask"])
        assert torch.equal(a["completion_mask"].bool(), b["labels"].ne(-100))
        prefix = torch.tensor(_tokenize_messages(tokenizer, row["prompt"])[1])
        for branch in range(2):
            assert torch.equal(b["input_ids"][branch, :len(prefix)], prefix)
            assert (b["labels"][branch, :len(prefix)] == -100).all()
        overflow_rejected = False
        try:
            CanonicalLlamaFactoryCollator(tokenizer, 1)([row])
        except ValueError as error:
            overflow_rejected = "truncation" in str(error)
        assert overflow_rejected
        records.append({
            "pair_id": row["pair_id"], "token_ids_equal": True,
            "shared_prefix_equal": True, "labels_equal": True,
            "overflow_rejected": True,
            "lengths": b["attention_mask"].sum(1).tolist(),
            "supervised_tokens": b["labels"].ne(-100).sum(1).tolist(),
            "token_tensor_sha256": hashlib.sha256(b["input_ids"].numpy().tobytes()).hexdigest(),
        })

    # Execute exactly the audited upstream function, not a reimplemented loss.
    # This is a source-level synthetic-logit check, not an LF runtime import.
    source = args.lf_source / "src/llamafactory/train/trainer_utils.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_batch_logps")
    namespace = {"torch": torch, "Optional": Optional, "IGNORE_INDEX": -100}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
    torch.manual_seed(42)
    logits = torch.randn(2, 11, 31)
    labels = torch.randint(0, 31, (2, 11))
    labels[:, [0, 1, 4, 5, 9]] = -100
    actual, counts = namespace["get_batch_logps"](logits, labels)
    expected = sequence_logps(logits, labels)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.equal(counts, labels[:, 1:].ne(-100).sum(1))
    assert hashlib.sha256(args.dataset.read_bytes()).hexdigest() == before
    result = {
        "dataset_sha256": before, "pair_count": len(records), "pairs": records,
        "CANONICAL_DATA_BRIDGE_PARITY_PASS": True,
        "UPSTREAM_SOURCE_LOGP_REDUCTION_UNIT_PASS": True,
        "source_function_sha256": hashlib.sha256(ast.get_source_segment(source.read_text(), fn).encode()).hexdigest(),
        "LLAMAFACTORY_RUNTIME_IMPORTED": False,
        "TRL_LLAMAFACTORY_DPO_PARITY_PASS": False,
        "full_parity_status": "not_run_real_model_and_explicit_reference_required",
        "MODEL_LOADED": False, "BACKWARD_STARTED": False, "TRAINING_STARTED": False,
        "VALIDATION400_USED": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
