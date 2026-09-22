# Transaction Decider demo

A single page and a Cloudflare Worker in front of the 4B decision model, which runs locally on
Apple silicon. The page and the API share one origin; only the Worker ever talks to the machine.

```
browser ──► <site hostname>/                  static asset, served by the Worker's asset layer
        ──► <site hostname>/api/v1/decider    Worker: rate limit, validate, forward
                 └──► <origin hostname>       cloudflared ingress ──► 127.0.0.1:8900
```

The two hostnames cannot be one. A Worker on a custom domain that fetches that same domain
reaches itself, so the model needs a name of its own. Only the site hostname is ever published;
the origin answers 401 to everything without the bearer token.

## Layout

| Path | What it is |
| --- | --- |
| `public/index.html` | the whole page: markup, styles, script |
| `public/_headers` | CSP and the other response headers for static assets |
| `src/worker.js` | rate limiting, request validation, origin call |
| `wrangler.jsonc` | Worker config, asset directory, rate limit bindings |
| `.dev.vars` | local-only secrets, never committed |

## Protection

Two layers, both at the edge, checked before the origin is touched.

**A WAF rate limiting rule is the one that actually holds.** Zone scope, `http_ratelimit` phase, 5 requests per 10 seconds per IP on `/api/v1/decider`, blocking for 10 seconds once tripped. The Free plan allows exactly one rule, counting by IP, with a 10 second period, which is what this uses. Verified against the deployed site: 30 parallel requests produced 20 blocks, and a browser sending 9 in a row got 4 answers then 5 blocks.

A blocked request never reaches the Worker, and Cloudflare answers it with its own page rather than JSON: plain `error code: 1015` to a bare client, an HTML block page to a browser. The page therefore parses the body defensively and maps the status itself, instead of assuming an API path returns JSON.

```
"rules": [{
  "ratelimit": {
    "characteristics": ["ip.src", "cf.colo.id"],
    "period": 10, "requests_per_period": 5,
    "requests_to_origin": false, "mitigation_timeout": 10
  },
  "expression": "(http.request.uri.path eq \"/api/v1/decider\")",
  "action": "block", "enabled": true
}]
```

**The Worker's `ratelimit` bindings are a best-effort second layer**, not protection on their own. Measured against a 5 per 10s budget, 100 parallel requests produced 4 rejections out of 48 that were counted. That is the documented behaviour, not a bug: the API is "permissive, eventually consistent, and intentionally designed to not be used as an accurate accounting system", and counters are local to each Cloudflare location. Treat these as a cheap pre-filter that catches the obvious case.

| Binding | Key | Budget | Enforcement |
| --- | --- | --- | --- |
| `BURST` | client IP | 5 per 10s | approximate, duplicated by the WAF rule |
| `SUSTAINED` | client IP | 20 per 60s | approximate, no free-plan WAF equivalent |
| `GLOBAL` | constant | 60 per 10s | approximate, the only cap on a spread-out flood |

A `namespace_id` carries its own limit. Editing `simple.limit` in place left the binding inert, so changing a budget means allocating a new id. Beyond that: 8 KB body cap at the edge, 16 KB at the origin,
4,000 character caps on both the text and the prompt, a rebuilt request body so no
extra field reaches the origin, and a bearer token the origin checks on every request.

The origin binds to `127.0.0.1`, so the tunnel is its only route in, and it only ever returns
a category distribution. It cannot generate text, so no training data can leave through it.

## Request

Two fields. What to judge, and optionally what to judge it against. The response reports
`custom_categories` so a caller always knows which of the two paths answered it, and therefore
which measured figures apply.

| Field | Type | Default | |
| --- | --- | --- | --- |
| `input` | string | required | What to judge, up to 4000 characters |
| `categories` | list | the trained 17 | 2 to 26 labels of your own |

The prompt wording is the service's, not the caller's. It uses the trained wording when no
categories are supplied, so the accuracy and calibration figures still describe that path, and a
neutral wording otherwise. Measured against prompts written per task, the neutral one scored
+2.0 points on AG News and +1.3 on emotion, both inside the margin, so there was nothing to
gain by asking callers for wording.

Unknown fields are ignored rather than rejected: the Worker rebuilds the body from the two it
knows, so nothing else reaches the origin.

## Run it locally

Two processes. First the model:

```sh
export DECIDER_TOKEN=$(openssl rand -hex 32)
.venv/bin/python lab/serve_mlx.py --model models/decider-4b-bf16 --port 8900
```

Then the edge, from this directory, with a `.dev.vars` holding the same token:

```sh
cat > .dev.vars <<EOF
DECIDER_TOKEN=$DECIDER_TOKEN
ORIGIN=http://127.0.0.1:8900
EOF
npm install
npm run dev          # http://127.0.0.1:8788
```

## Deploy

1. Point a hostname at the model through the existing tunnel. In `~/.cloudflared/config.yml`,
   above the catch-all:

   ```yaml
     - hostname: decider.example.com
       service: http://127.0.0.1:8900
       originRequest:
         connectTimeout: 5s
   ```

   Then `cloudflared tunnel ingress validate`, `cloudflared tunnel route dns <tunnel> decider.example.com`,
   and restart `cloudflared` so it reads the new file. A restart drops live SSH sessions on the
   same tunnel.

2. Ship both secrets, then the Worker:

   ```sh
   npx wrangler secret put DECIDER_TOKEN
   npx wrangler secret put ORIGIN        # the origin hostname, e.g. https://decider.example.com
   npm run deploy
   ```

   `ORIGIN` is a secret rather than a `vars` entry so that a public repository does not name the
   tunnel that reaches the machine. A bearer token is what actually protects it, but the hostname
   is free to withhold. A binding cannot be a var and a secret at once, so if it is currently a
   var, remove it and deploy before running `secret put`.

3. Add the site hostname under `routes` in `wrangler.jsonc` with `custom_domain: true`. That
   hostname must have no DNS record of any kind first, including a tunnel record, or the
   deployment fails with `code: 100117`; Wrangler creates and owns the record itself. Nothing in
   the page changes, since the API path is relative.

The model process has to be running for the demo to answer. When it is not, the page loads and
says so.
