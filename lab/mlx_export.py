#!/usr/bin/env python3
"""Fold a LoRA adapter into its base and convert the result to MLX, for serving on Apple silicon.

mlx-lm cannot read a PEFT adapter: its own LoRA layout uses different parameter names, so the adapter is merged into the base weights here and what MLX loads is a plain Qwen3 checkpoint.

    python lab/mlx_export.py --adapter runs/s3_full_17/adapter --out models/decider-4b
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path


def flatten_rope(config_path):
    """transformers 5 writes rope settings nested under "rope_parameters"; mlx-lm reads the flat "rope_theta" and "rope_scaling" keys and errors out without them."""
    config = json.loads(config_path.read_text())
    nested = config.pop("rope_parameters", None)
    if nested:
        config.setdefault("rope_theta", nested.get("rope_theta"))
        scaling = {k: v for k, v in nested.items() if k != "rope_theta"}
        config.setdefault("rope_scaling",
                          None if scaling.get("rope_type") == "default" else scaling)
        config_path.write_text(json.dumps(config, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--base", default="Qwen/Qwen3-4B")
    p.add_argument("--out", required=True)
    p.add_argument("--quantize", type=int, default=0, help="bits (4 or 8); 0 keeps bf16")
    p.add_argument("--keep-merged", help="also keep the merged HF checkpoint here")
    args = p.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged = Path(args.keep_merged or tempfile.mkdtemp(prefix="merge_"))
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(merged)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(merged)
    del model
    flatten_rope(merged / "config.json")

    from mlx_lm import convert
    convert(str(merged), mlx_path=args.out, quantize=bool(args.quantize),
            q_bits=args.quantize or None, dtype=None if args.quantize else "bfloat16")
    if not args.keep_merged:
        shutil.rmtree(merged)
    print(args.out)


if __name__ == "__main__":
    main()
