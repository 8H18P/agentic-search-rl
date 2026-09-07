#!/usr/bin/env python3
"""生成 GRPO 入口所需的本地模型与 tokenizer 身份清单。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--tokenizer-path', type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    required = [
        args.model_path / 'config.json',
        args.tokenizer_path / 'tokenizer.json',
        args.tokenizer_path / 'tokenizer_config.json',
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'缺少模型身份文件: {missing}')
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, local_files_only=True, trust_remote_code=True
    )
    if not tokenizer.chat_template:
        raise ValueError('tokenizer.chat_template 为空，无法冻结训练模板身份')
    manifest = {
        'model_id': args.model_id,
        'expected_revision': args.revision,
        'resolved_revision': args.revision,
        'model_local_path': str(args.model_path.resolve()),
        'tokenizer_local_path': str(args.tokenizer_path.resolve()),
        'config_sha256': sha256(required[0]),
        'tokenizer_json_sha256': sha256(required[1]),
        'tokenizer_config_sha256': sha256(required[2]),
        'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
