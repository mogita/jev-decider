#!/usr/bin/env python3
"""QLoRA fine-tune for single-token constrained decoding.

One forward pass per decision. The answer is read from the logits at the candidate letter tokens, never generated, so latency does not depend on how many categories there are. Loss is cross-entropy over those candidate letters only, which is the same normalization used at inference.
"""
import argparse
import collections
import json
import math
import random
import string
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROMPT = """Below is a bank transaction.

{state}

Which budget category does it belong to?

{options}

Reply with the single letter only."""


def load(split_path, letters):
    records = [json.loads(l) for l in Path(split_path).read_text().splitlines() if l.strip()]
    criteria = records[0]["questions"]["category"]["criteria"]
    ids = list(criteria)
    options = "\n".join(f"{letters[i]}. {criteria[k]}" for i, k in enumerate(ids))
    return [{"prompt": PROMPT.format(state=r["state"], options=options),
             "gold": ids.index(r["gold"]["category"])} for r in records], ids


def class_weights(items, n_classes, device, cap=6.0):
    """Inverse-frequency weights. Without this the model collapses minority classes into their dominant neighbour: shopping was predicted as groceries 10 times out of 12.

    Capped, because uncapped inverse frequency hands a 1-row class a 26x weight and the model then overpredicts it everywhere. A class absent from this stage gets 0 and simply does not contribute; another stage supplies it.
    """
    counts = collections.Counter(it["gold"] for it in items)
    weights = [min(cap, len(items) / (n_classes * counts[i])) if counts[i] else 0.0
               for i in range(n_classes)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def encode(tokenizer, items, device):
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": it["prompt"]}],
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False) for it in items]
    enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    gold = torch.tensor([it["gold"] for it in items], device=device)
    return enc, gold


@torch.no_grad()
def evaluate(model, tokenizer, items, letter_ids, batch_size, collect=False):
    """Balanced accuracy is the headline: plain accuracy rewards guessing the common class."""
    model.eval()
    top1 = top3 = 0
    loss_sum = 0.0
    per_class = collections.defaultdict(lambda: [0, 0])
    rows = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        enc, gold = encode(tokenizer, chunk, model.device)
        scores = model(**enc).logits[:, -1, :][:, letter_ids].float()
        loss_sum += torch.nn.functional.cross_entropy(scores, gold, reduction="sum").item()
        order = scores.argsort(dim=-1, descending=True)
        for logits, rank, g in zip(scores.tolist(), order.tolist(), gold.tolist()):
            hit = rank[0] == g
            top1 += hit
            top3 += g in rank[:3]
            per_class[g][0] += hit
            per_class[g][1] += 1
            if collect:
                rows.append({"gold": g, "pred": rank[0], "logits": [round(v, 4) for v in logits]})
    balanced = sum(h / n for h, n in per_class.values()) / len(per_class)
    metrics = {"top1": top1 / len(items), "top3": top3 / len(items),
               "balanced": balanced, "ce": loss_sum / len(items)}
    return (metrics, rows) if collect else metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--data-dir", default="data/lunchmoney",
                   help="supplies dev/test and, unless --stage is given, train")
    # "path[:epochs[:lr[:limit]]]", repeatable, applied in order with the user's data last.
    p.add_argument("--stage", action="append", default=[],
                   help="extra training stage: file.jsonl[:epochs[:lr[:max_rows]]]")
    p.add_argument("--us-test", help="second eval split, reported separately")
    p.add_argument("--class-weighted", action="store_true",
                   help="weight the loss by inverse class frequency")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=17)
    # qlora suits a model too big to hold in fp32 gradients; a 0.6B is cheaper trained whole.
    p.add_argument("--mode", choices=["qlora", "lora", "full"], default="qlora")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    probe = json.loads(Path(args.data_dir, "train.jsonl").read_text().splitlines()[0])
    letters = string.ascii_uppercase[:len(probe["questions"]["category"]["criteria"])]
    letter_ids = []
    for ch in letters:
        enc = tokenizer.encode(ch, add_special_tokens=False)
        if len(enc) != 1:
            raise SystemExit(f"letter {ch!r} is {len(enc)} tokens; pick different labels")
        letter_ids.append(enc[0])

    splits, ids = {}, None
    for name in ("train", "dev", "test"):
        splits[name], ids = load(Path(args.data_dir, f"{name}.jsonl"), letters)
    us_test = load(Path(args.us_test), letters)[0] if args.us_test else None

    stages = []
    for spec in args.stage:
        parts = spec.split(":")
        path = parts[0]
        epochs = int(parts[1]) if len(parts) > 1 and parts[1] else 1
        lr = float(parts[2]) if len(parts) > 2 and parts[2] else args.lr
        limit = int(parts[3]) if len(parts) > 3 and parts[3] else None
        data = load(Path(path), letters)[0]
        if limit and len(data) > limit:
            data = random.sample(data, limit)
        stages.append({"name": Path(path).parent.name or Path(path).stem,
                       "data": data, "epochs": epochs, "lr": lr})
    stages.append({"name": "real", "data": splits["train"], "epochs": args.epochs, "lr": args.lr})

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    ) if args.mode == "qlora" else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cuda", attn_implementation="sdpa",
        dtype=torch.bfloat16 if quant is None else None, quantization_config=quant)
    if args.mode == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if args.mode != "full":
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(json.dumps({"mode": args.mode, "trainable": trainable, "total": total,
                      "pct": round(100 * trainable / total, 4)}), flush=True)

    base = evaluate(model.eval(), tokenizer, splits["test"], letter_ids, args.eval_batch_size)
    print(json.dumps({"stage": "before_training", **{f"test_{k}": v for k, v in base.items()}}), flush=True)

    trained_keys = {n for n, p_ in model.named_parameters() if p_.requires_grad}
    best, snapshot, log, started = -1.0, {}, [], time.perf_counter()
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]

    for stage in stages:
        data = stage["data"]
        weights = class_weights(data, len(letter_ids), model.device) if args.class_weighted else None
        optimizer = torch.optim.AdamW(trainable, lr=stage["lr"])
        steps_per_epoch = math.ceil(len(data) / (args.batch_size * args.accum))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, stage["epochs"] * steps_per_epoch))
        print(json.dumps({"stage": stage["name"], "rows": len(data), "epochs": stage["epochs"],
                          "lr": stage["lr"], "steps_per_epoch": steps_per_epoch}), flush=True)

        for epoch in range(stage["epochs"]):
            model.train()
            order = list(range(len(data)))
            random.shuffle(order)
            micro = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
            for step in range(steps_per_epoch):
                optimizer.zero_grad(set_to_none=True)
                groups = micro[step * args.accum:(step + 1) * args.accum]
                if not groups:
                    continue
                seen = sum(len(g) for g in groups)
                for group in groups:
                    enc, gold = encode(tokenizer, [data[i] for i in group], model.device)
                    scores = model(**enc).logits[:, -1, :][:, letter_ids].float()
                    loss = torch.nn.functional.cross_entropy(
                        scores, gold, weight=weights, reduction="sum") / seen
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()

            dev = evaluate(model, tokenizer, splits["dev"], letter_ids, args.eval_batch_size)
            entry = {"stage": stage["name"], "epoch": epoch + 1,
                     **{f"dev_{k}": v for k, v in dev.items()},
                     "elapsed_seconds": time.perf_counter() - started}
            log.append(entry)
            print(json.dumps(entry), flush=True)
            # Only the final stage may set the checkpoint: earlier stages are pretraining and
            # their dev score is not what we are optimizing for.
            if stage is stages[-1] and dev["balanced"] > best:
                best = dev["balanced"]
                model.save_pretrained(out / "adapter")
                (out / "best_epoch.json").write_text(json.dumps(entry) + "\n")
                snapshot = {k: v.detach().to("cpu", copy=True)
                            for k, v in model.state_dict().items()
                            if v.requires_grad or k in trained_keys}

    missing, unexpected = model.load_state_dict(
        {k: v.to(model.device) for k, v in snapshot.items()}, strict=False)
    if unexpected or not snapshot:
        raise SystemExit(f"checkpoint restore failed: {len(unexpected)} unexpected keys, "
                         f"{len(snapshot)} snapshot entries")
    print(json.dumps({"restored_tensors": len(snapshot)}), flush=True)

    test, preds = evaluate(model, tokenizer, splits["test"], letter_ids,
                           args.eval_batch_size, collect=True)
    (out / "predictions_test.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in preds), encoding="utf-8")
    (out / "categories.json").write_text(json.dumps(ids, indent=1) + "\n")
    summary = {"model": args.model, "mode": args.mode, "seed": args.seed, "epochs": args.epochs,
               "rank": args.rank, "alpha": args.alpha, "lr": args.lr,
               **{f"zeroshot_test_{k}": v for k, v in base.items()},
               **{f"test_{k}": v for k, v in test.items()},
               "training_seconds": time.perf_counter() - started,
               "stages": [{k: v for k, v in st.items() if k != "data"} | {"rows": len(st["data"])}
                          for st in stages],
               "class_weighted": args.class_weighted, "log": log}
    if us_test:
        us = evaluate(model, tokenizer, us_test, letter_ids, args.eval_batch_size)
        summary.update({f"us_{k}": v for k, v in us.items()})
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "log"}), flush=True)


if __name__ == "__main__":
    main()
