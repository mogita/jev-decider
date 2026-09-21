# jev-decider

A small model that categorizes bank transactions in one forward pass, and the page that demos it.

Live at **https://mojev.mogita.rocks**, served off a laptop.

## What it does

A record goes in and a category comes out. The model never writes a sentence. It reads the record once, scores every candidate in that single pass by reading the logits at the candidate letter tokens, and returns the strongest. Two things follow: an invalid category is impossible, because the answer can only be one of the letters offered, and response time does not grow with the number of categories, because they all live in one logit vector.

Callers may bring their own categories, and up to ten records ride in one request.

## Numbers

| | |
| --- | --- |
| Base | Qwen3-4B, LoRA rank 16 alpha 32, merged to bf16 |
| Accuracy | 69.2% balanced on a merchant-disjoint Dutch holdout, 94.8% on US |
| Calibration | 0.080 expected calibration error, temperature 1.0 |
| Unseen schemas | 86.2% on AG News, 47.2% on emotion, neither in training |
| Latency | 214 ms median on an M4 Max, about 270 ms end to end through the edge |
| Training cost | 47 minutes, $0.42 of rented GPU |

Full write-up, including the approaches that lost, in [lab/RESULTS.md](lab/RESULTS.md).

## Layout

| Path | What it is |
| --- | --- |
| `lab/serve_mlx.py` | the service, MLX on Apple silicon |
| `lab/serve_decision.py` | prompts, request validation, HTTP handler, shared by both servers |
| `lab/qlora_train.py` | the trainer that produced the shipped adapter |
| `lab/mlx_export.py` | merge the adapter into the base and convert to MLX |
| `lab/calibrate.py`, `lab/schema_eval.py`, `lab/ablate.py` | measurement |
| `lab/build_lunchmoney.py`, `lab/synth_nl.py`, `lab/us_convert.py` | dataset building |
| `demo/` | the public page and the Cloudflare Worker in front of the model |

## Running the service

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
python lab/mlx_export.py --adapter runs/s3_full_17/adapter --out models/decider-4b-bf16
python lab/serve_mlx.py --model models/decider-4b-bf16 --bench 32 --batch 8
python lab/serve_mlx.py --model models/decider-4b-bf16 --port 8900
```

```bash
curl -s localhost:8900/decide -H 'content-type: application/json' \
  -d '{"input": "Payee: Albert Heijn\nAmount: -12.40 EUR"}'
```

`input` takes one record or a list of up to ten, and the reply mirrors the shape that was sent. `categories` is optional and replaces the trained seventeen.

## What is not in this repository

Weights, runs and data are ignored: `data/` holds a personal bank export, `models/` and `runs/` hold several gigabytes of checkpoints. The synthetic Dutch data can be regenerated with `lab/synth_nl.py`, and the US set is [public](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2).

## Lineage

The first attempt used the decision heads from [NanoJev](https://github.com/TianyuCodings/NanoJev), a separate project, and none of its code is here. That approach scores each candidate as its own sequence, which is too slow to serve; the comparison that led to single-token decoding instead is in the write-up. One command in that write-up runs in a NanoJev checkout, because the trainer it names belongs to that project and is not vendored here.
