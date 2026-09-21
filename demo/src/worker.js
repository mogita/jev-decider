// Edge front door for the decision model. The model itself runs on a laptop behind a Cloudflare tunnel, so everything that protects it has to happen here: rate limits, a size cap, a strict request shape, and a shared secret the tunnel origin checks.

const MAX_BODY = 65536;
const MAX_INPUTS = 10;
const ORIGIN_TIMEOUT_MS = 15000;

const json = (data, status = 200, headers = {}) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store", ...headers },
  });

// Per-IP burst and sustained caps, plus one global cap. The global one matters most: the origin is a single machine, and a spread-out flood never trips a per-address limit.
// Cost is the number of records in the call, and each one spends a unit, because a call carrying 25 records asks the origin for 25 records of work. A limiter that counted calls would let a batch walk a whole queue past it.
async function withinLimits(env, ip, cost) {
  const checks = [[env.BURST, ip], [env.SUSTAINED, ip], [env.GLOBAL, "all"]];
  for (const [limiter, key] of checks) {
    const verdicts = await Promise.all(
      Array.from({ length: cost }, () => limiter.limit({ key })));
    if (verdicts.some(({ success }) => !success)) return false;
  }
  return true;
}

function parseCategories(categories) {
  if (!Array.isArray(categories)) throw new Error("categories must be a list of labels");
  if (!categories.every((label) => typeof label === "string")) {
    throw new Error("every category must be a string");
  }
  const labels = categories.map((label) => label.trim()).filter(Boolean);
  if (labels.length < 2 || labels.length > 26) {
    throw new Error("give between 2 and 26 categories, one per line");
  }
  if (labels.some((label) => label.length > 80)) throw new Error("a label is over 80 characters");
  if (new Set(labels).size !== labels.length) throw new Error("category labels must be distinct");
  return labels;
}

// One record, or a list of them scored in a single call. The reply mirrors the shape that came in, so a caller that sent a string never has to reach into an array to read its answer.
function parseInput(input) {
  const records = Array.isArray(input) ? input : [input];
  if (!records.length) throw new Error("input must hold at least one record");
  if (records.length > MAX_INPUTS) {
    throw new Error(`input must hold at most ${MAX_INPUTS} records`);
  }
  for (const record of records) {
    if (typeof record !== "string" || !record.trim()) {
      throw new Error("every record must be a non-empty string");
    }
    if (record.length > 4000) throw new Error("a record is longer than 4000 characters");
  }
  return records;
}

function parseDecideRequest(raw) {
  const { input, categories } = JSON.parse(raw);
  const records = parseInput(input);
  // Rebuilt rather than forwarded, so no extra field reaches the origin.
  return {
    input: Array.isArray(input) ? records : records[0],
    ...(categories === undefined || categories === null ? {} : { categories: parseCategories(categories) }),
  };
}

async function callOrigin(env, path, payload) {
  return fetch(`${env.ORIGIN}${path}`, {
    method: payload ? "POST" : "GET",
    headers: {
      authorization: `Bearer ${env.DECIDER_TOKEN}`,
      ...(payload ? { "content-type": "application/json" } : {}),
    },
    body: payload ? JSON.stringify(payload) : undefined,
    signal: AbortSignal.timeout(ORIGIN_TIMEOUT_MS),
    cf: payload ? undefined : { cacheTtl: 60, cacheEverything: true },
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname !== "/api/v1/decider") return json({ error: "not found" }, 404);
    if (!["GET", "POST"].includes(request.method)) {
      return json({ error: "method not allowed" }, 405, { allow: "GET, POST" });
    }

    // The body is read before the limiters run, because what a call costs is how many records it carries, and that is only known once it has been parsed.
    let payload = null;
    if (request.method === "POST") {
      const raw = await request.text();
      if (raw.length > MAX_BODY) return json({ error: `body over ${MAX_BODY} bytes` }, 413);
      try {
        payload = parseDecideRequest(raw);
      } catch (error) {
        return json({ error: error.message }, 400);
      }
    }

    const ip = request.headers.get("cf-connecting-ip") ?? "unknown";
    const cost = Array.isArray(payload?.input) ? payload.input.length : 1;
    if (!(await withinLimits(env, ip, cost))) {
      return json({ error: "Too fast, please be gentle to my potato." }, 429,
        { "retry-after": "10" });
    }

    const started = Date.now();
    let response;
    try {
      response = await callOrigin(env, payload ? "/decide" : "/", payload);
    } catch {
      return json({ error: "My laptop is asleep or offline." }, 503);
    }
    if (!response.ok) {
      // The origin's own message can echo the request back, so only the status is relayed, and anything in the 5xx range (including Cloudflare's own 530 when the tunnel is down) means the same thing to a visitor: the machine is not answering.
      if (response.status === 400) return json({ error: "The model rejected that input." }, 400);
      return json({ error: "My laptop is asleep or offline." }, 503);
    }
    const result = await response.json();
    const edge_ms = Date.now() - started;
    return json(Array.isArray(result) ? result.map((row) => ({ ...row, edge_ms }))
                                      : { ...result, edge_ms });
  },
};
