#!/usr/bin/env python3
"""The decision service of lab/serve_decision.py, running on MLX instead of CUDA.

mlx_lm.server is not usable for this: it is an OpenAI chat-completions endpoint that generates tokens and returns at most a handful of logprobs, so the distribution over 17 categories would have to be inferred from samples. Here the whole distribution is read from the one logit vector that already holds it, in a single forward pass, which is also what keeps latency flat in K.

    python lab/serve_mlx.py --model models/decider-4b-bf16 --bench 50
    python lab/serve_mlx.py --model models/decider-4b-bf16 --port 8900
"""
import argparse
import json
import os
import string
import threading
import time
from http.server import HTTPServer
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from serve_decision import GENERIC, PROMPT, handler_for, options_text, parse_criteria, render


class MlxDecider:
    def __init__(self, path, categories, temperature):
        self.model, self.tokenizer = load(path)
        self.criteria = categories
        self.ids = list(categories)
        self.letters = string.ascii_uppercase
        # All 26 letters are tokenized once; a request uses the first len(criteria) of them.
        self.letter_ids = mx.array(
            [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in self.letters])
        self.options = options_text(categories)
        # Reading the logits of one position per row means the language head only has to run on
        # those positions, not on every token of a padded batch, where it would dominate the cost.
        self.head = (self.model.model.embed_tokens.as_linear
                     if self.model.args.tie_word_embeddings else self.model.lm_head)
        self.temperature = temperature
        self.last_used = time.monotonic()
        # The warmup thread and the request thread both reach the model.
        self.lock = threading.Lock()

    def decide(self, state, top_k=None, template=None, criteria=None):
        return self.decide_batch([state], top_k, template, criteria)[0]

    def decide_batch(self, states, top_k=None, template=None, criteria=None, chunk=8):
        """Score any number of records, a chunk of rows per forward pass.

        Batching buys far less here than it does for token generation. Generation is starved for
        weights and a batch rides along on weights already loaded; a 260 token prompt already
        saturates the GPU with arithmetic, so a second row mostly adds a second row of work. What
        the batch does save is padding and per-call overhead, and rows are sorted by length before
        chunking so short records are not padded up to the longest one in the request.
        """
        started = time.perf_counter()
        self.last_used = time.monotonic()
        choices, custom = parse_criteria(criteria, self.criteria)
        ids = list(choices)
        template = template or (GENERIC if custom else PROMPT)
        top_k = len(ids) if top_k is None else top_k
        options = options_text(choices)
        rows = [self.tokenizer.apply_chat_template(
            [{"role": "user", "content": render(template, state, options)}],
            add_generation_prompt=True, enable_thinking=False) for state in states]
        table = [None] * len(rows)
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
        with self.lock:
            for start in range(0, len(order), max(1, chunk)):
                group = order[start:start + max(1, chunk)]
                width = max(len(rows[i]) for i in group)
                # Padding goes on the right. The mask is causal, so a row's last real token never
                # attends to anything after it, and the padding cannot reach the logits read back.
                batch = mx.array([rows[i] + [0] * (width - len(rows[i])) for i in group])
                hidden = self.model.model(batch)
                last = mx.stack([hidden[n, len(rows[i]) - 1] for n, i in enumerate(group)])
                scores = self.head(last)[:, self.letter_ids[:len(ids)]]
                probs = mx.softmax(scores.astype(mx.float32) / self.temperature, axis=-1).tolist()
                for i, row in zip(group, probs):
                    table[i] = row
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        results = []
        for probs, row in zip(table, rows):
            rank = sorted(range(len(probs)), key=lambda i: -probs[i])[:max(1, min(top_k, len(ids)))]
            results.append({
                "decision": ids[rank[0]],
                "confidence": round(probs[rank[0]], 4),
                "alternatives": [{"category": ids[i], "p": round(probs[i], 4)} for i in rank],
                "latency_ms": elapsed,
                "prompt_tokens": len(row),
                "custom_categories": custom,
            })
        return results


def keep_warm(decider, seconds):
    """macOS reclaims the weights when the process idles, and the next request then pays 13 seconds to fault 7.5 GB back in. One cheap forward pass per interval keeps them resident."""
    def loop():
        while True:
            time.sleep(seconds)
            if time.monotonic() - decider.last_used > seconds:
                decider.decide("Payee: warmup\nAmount: -1.00 EUR", 1)
    threading.Thread(target=loop, daemon=True).start()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/decider-4b-bf16", help="MLX model directory")
    p.add_argument("--data-dir", default="data/lunchmoney")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--host", default="127.0.0.1", help="Bind address; the tunnel reaches loopback")
    p.add_argument("--port", type=int, default=8900)
    p.add_argument("--token", default=os.environ.get("DECIDER_TOKEN"),
                   help="Require 'Authorization: Bearer <token>'; defaults to $DECIDER_TOKEN")
    p.add_argument("--warm-seconds", type=int, default=60,
                   help="Idle interval after which a warmup pass runs; 0 disables")
    p.add_argument("--bench", type=int, default=0, help="Run N timed requests locally and exit")
    p.add_argument("--batch", type=int, default=1,
                   help="With --bench, also score the same rows in batches of this size")
    args = p.parse_args()

    records = [json.loads(l) for l in
               Path(args.data_dir, "test.jsonl").read_text().splitlines() if l.strip()]
    decider = MlxDecider(args.model, records[0]["questions"]["category"]["criteria"],
                         args.temperature)

    if args.bench:
        for r in records[:3]:
            decider.decide(r["state"], args.top_k)          # warm up the kernels
        pool = [r["state"] for r in
                (records * (args.bench // len(records) + 1))[:args.bench]]
        one = [decider.decide(state, args.top_k) for state in pool]
        times = sorted(r["latency_ms"] for r in one)
        report = {"n": len(times), "p50": times[len(times) // 2],
                  "p95": times[int(len(times) * 0.95)], "min": times[0], "max": times[-1],
                  "sequential_ms": round(sum(times))}
        if args.batch > 1:
            started = time.perf_counter()
            many = decider.decide_batch(pool, args.top_k, chunk=args.batch)
            report["batch"] = args.batch
            report["batched_ms"] = round((time.perf_counter() - started) * 1000)
            report["speedup"] = round(sum(times) / report["batched_ms"], 2)
            # The batch has to agree with the same rows scored one at a time, or it is not the
            # same service with a faster wrapper, it is a different one.
            report["agree"] = sum(a["decision"] == b["decision"] for a, b in zip(one, many))
        print(json.dumps(report))
        return

    if args.warm_seconds:
        keep_warm(decider, args.warm_seconds)
    print(json.dumps({"serving": args.model, "host": args.host, "port": args.port,
                      "auth": bool(args.token), "warm_seconds": args.warm_seconds}), flush=True)
    HTTPServer((args.host, args.port),
               handler_for(decider, args.top_k, args.token)).serve_forever()


if __name__ == "__main__":
    main()
