#!/usr/bin/env python3
"""Jev-style decision service: structured state in, typed distribution out.

One forward pass per request. The answer is read from the logits at the candidate letter tokens, never generated, so the response is a probability distribution over the declared enum and a type error is impossible. Latency is flat in the number of categories, since they all live in one logit vector.

    python lab/serve_decision.py --adapter runs/qlora14b_s17/adapter --port 8900
    curl -s localhost:8900/decide -H 'content-type: application/json' \
      -d '{"state": "Payee: Albert Heijn\nAmount: -12.40 EUR"}' | jq
"""
import argparse
import json
import os
import string
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROMPT = """Below is a bank transaction.

{state}

Which budget category does it belong to?

{options}

Reply with the single letter only."""

# Used whenever a caller brings its own schema, since it supplies no wording of its own. Measured
# against per-task prompts on two public sets: +2.0 points on AG News, +1.3 on emotion, both
# inside the margin, so asking callers for wording bought nothing.
GENERIC = """Below is a record.

{state}

Which of the following does it belong to?

{options}

Reply with the single letter only."""

MAX_STATE = 4000
MAX_TEMPLATE = 4000
MAX_CHOICES = 26        # one letter each, A to Z
MAX_LABEL = 80
# One call holds at most this many records. The origin is a single laptop that scores about five
# records a second, so ten is about two seconds of work: worth one round trip, and small enough
# that a request which slips past the edge rate limit still cannot buy much of the machine.
MAX_INPUTS = 10


def parse_input(value):
    """One record, or a list of them scored in a single call.

    The reply mirrors what was sent: a string gets an object back, a list gets a list back in the
    same order, so a caller never has to unwrap a result it did not ask to be wrapped.
    """
    many = isinstance(value, list)
    states = value if many else [value]
    if not states:
        raise ValueError("input must hold at least one record")
    if len(states) > MAX_INPUTS:
        raise ValueError(f"input must hold at most {MAX_INPUTS} records")
    for state in states:
        if not isinstance(state, str) or not state.strip():
            raise ValueError("every record must be a non-empty string")
        if len(state) > MAX_STATE:
            raise ValueError(f"a record is longer than {MAX_STATE} characters")
    return states, many


def decide_all(decider, states, criteria):
    """Use the batched path when the decider has one; the CUDA decider still scores one at a time."""
    batched = getattr(decider, "decide_batch", None)
    if batched:
        return batched(states, criteria=criteria)
    return [decider.decide(state, criteria=criteria) for state in states]


def parse_criteria(criteria, fallback):
    """A caller may bring its own enum. The letters carry the decision either way, so the only thing that changes is which label each position maps back to. Order is the caller's.

    The tuning taught this model one 17-way budget schema; another schema rides the base model's own ability and is not covered by any accuracy or calibration number measured here.
    """
    if criteria is None:
        return fallback, False
    if isinstance(criteria, list):
        if not all(isinstance(item, str) for item in criteria):
            raise ValueError("every entry in a categories list must be a string")
        if len(set(criteria)) != len(criteria):
            raise ValueError("category labels must be distinct")
        criteria = {item: item for item in criteria}
    if not isinstance(criteria, dict):
        raise ValueError("categories must be an object of id to label, or a list of labels")
    if not 2 <= len(criteria) <= MAX_CHOICES:
        raise ValueError(f"categories must hold 2 to {MAX_CHOICES} entries")
    for key, label in criteria.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"label for {key!r} must be a non-empty string")
        if len(label) > MAX_LABEL:
            raise ValueError(f"label for {key!r} is longer than {MAX_LABEL} characters")
    return criteria, True


def options_text(criteria):
    return "\n".join(f"{letter}. {criteria[key]}"
                     for letter, key in zip(string.ascii_uppercase, criteria))


def render(template, state, options):
    """Substitute by replacement, not str.format: the template is caller-supplied in the demo, and format() would let it walk attributes of whatever is passed in."""
    template = template or PROMPT
    if len(template) > MAX_TEMPLATE:
        raise ValueError(f"prompt longer than {MAX_TEMPLATE} characters")
    if "{state}" not in template:
        raise ValueError("prompt must contain {state}")
    if len(state) > MAX_STATE:
        raise ValueError(f"input longer than {MAX_STATE} characters")
    return template.replace("{state}", state).replace("{options}", options)


class Decider:
    def __init__(self, model_id, adapter, categories, temperature):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cuda", attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.eval()
        self.criteria = categories
        self.ids = list(categories)
        # All 26 letters are tokenized once; a request uses the first len(criteria) of them.
        self.letter_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0]
                           for c in string.ascii_uppercase]
        self.options = options_text(categories)
        self.temperature = temperature

    @torch.no_grad()
    def decide(self, state, top_k=None, template=None, criteria=None):
        started = time.perf_counter()
        choices, custom = parse_criteria(criteria, self.criteria)
        ids = list(choices)
        template = template or (GENERIC if custom else PROMPT)
        top_k = len(ids) if top_k is None else top_k
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": render(template, state, options_text(choices))}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        scores = self.model(**enc).logits[:, -1, :][0, self.letter_ids[:len(ids)]].float()
        probs = torch.softmax(scores / self.temperature, dim=-1)
        order = probs.argsort(descending=True)[:max(1, min(top_k, len(ids)))].tolist()
        return {
            "decision": ids[order[0]],
            "confidence": round(probs[order[0]].item(), 4),
            "alternatives": [{"category": ids[i], "p": round(probs[i].item(), 4)} for i in order],
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "prompt_tokens": int(enc.input_ids.shape[1]),
            "custom_categories": custom,
        }


def handler_for(decider, top_k, token=None, max_bytes=65536):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, result):
            payload = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def authorized(self):
            # The tunnel puts this process on the public internet, so the shared secret is the only thing between the edge Worker and anyone who guesses the hostname.
            if token and self.headers.get("authorization") != f"Bearer {token}":
                self.send_error(401, "unauthorized")
                return False
            return True

        def do_POST(self):
            if self.path != "/decide":
                self.send_error(404)
                return
            if not self.authorized():
                return
            try:
                length = int(self.headers.get("content-length") or 0)
                if not 0 < length <= max_bytes:
                    raise ValueError(f"body must be 1 to {max_bytes} bytes")
                body = json.loads(self.rfile.read(length))
                # Two fields on the wire: what to judge, and optionally what to judge it
                # against. The wording is the service's business, not the caller's.
                states, many = parse_input(body["input"])
                results = decide_all(decider, states, body.get("categories"))
            except (KeyError, ValueError, TypeError) as exc:
                self.send_error(400, str(exc))
                return
            self.reply(results if many else results[0])

        def do_GET(self):
            if not self.authorized():
                return
            self.reply({"ok": True, "categories": decider.criteria, "prompt": PROMPT,
                        "max_inputs": MAX_INPUTS})
    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--adapter", help="LoRA adapter directory; omit to serve the frozen base")
    p.add_argument("--data-dir", default="data/lunchmoney")
    p.add_argument("--temperature", type=float, default=1.0, help="1.0 is already well calibrated here, see lab/calibrate.py")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--host", default="127.0.0.1", help="Bind address; the tunnel reaches loopback")
    p.add_argument("--port", type=int, default=8900)
    p.add_argument("--token", default=os.environ.get("DECIDER_TOKEN"),
                   help="Require 'Authorization: Bearer <token>'; defaults to $DECIDER_TOKEN")
    p.add_argument("--bench", type=int, default=0, help="Run N timed requests locally and exit")
    args = p.parse_args()

    records = [json.loads(l) for l in Path(args.data_dir, "test.jsonl").read_text().splitlines() if l.strip()]
    decider = Decider(args.model, args.adapter, records[0]["questions"]["category"]["criteria"],
                      args.temperature)

    if args.bench:
        for r in records[:3]:
            decider.decide(r["state"], args.top_k)          # warm up the kernels
        times = [decider.decide(r["state"], args.top_k)["latency_ms"] for r in records[:args.bench]]
        times.sort()
        print(json.dumps({"n": len(times), "p50": times[len(times) // 2],
                          "p95": times[int(len(times) * 0.95)], "min": times[0], "max": times[-1]}))
        return

    print(json.dumps({"serving": args.model, "adapter": args.adapter, "host": args.host,
                      "port": args.port, "auth": bool(args.token)}), flush=True)
    HTTPServer((args.host, args.port),
               handler_for(decider, args.top_k, args.token)).serve_forever()


if __name__ == "__main__":
    main()
