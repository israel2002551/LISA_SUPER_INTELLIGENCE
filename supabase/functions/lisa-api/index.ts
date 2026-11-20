// =============================================================================
// LISA device API as a Supabase Edge Function (Deno)
// =============================================================================
// Port of the Python server's device-facing endpoints (server/app/routers/device.py,
// server/app/ingest.py, server/app/uart_analysis.py), so the API runs inside the
// same Supabase project as the database and storage: free, always on, and on a
// host the board already trusts (same TLS chain as model downloads).
//
// Base URL for the board (EDGE_SERVER_BASE_URL):
//   https://<project-ref>.supabase.co/functions/v1/lisa-api
// Routes (same as the Python server):
//   GET  /  ,  GET /healthz
//   POST /api/v1/telemetry           node key
//   POST /api/v1/uart/classify       node key
//   GET  /api/v1/ota/check           node key
//   POST /api/v1/evolution/trigger   ADMIN_TOKEN   (queues a run for the GitHub Actions worker)
//   GET  /api/v1/evolution/runs      ADMIN_TOKEN
//   POST /api/v1/admin/nodes         ADMIN_TOKEN   (register a board, returns its key once)
//   POST /api/v1/admin/models/{v}/activate  ADMIN_TOKEN
//
// Deploy: Dashboard -> Edge Functions -> Deploy a new function -> Via editor,
// name "lisa-api", paste this file, deploy. Then TURN OFF "Enforce JWT verification"
// for it: boards authenticate with their own node key, not a Supabase JWT.
// Or with the CLI:  supabase functions deploy lisa-api --no-verify-jwt
//
// Secrets: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are provided automatically.
// Optional: ADMIN_TOKEN (enables the admin routes), AI_NAME (default "LISA").
// =============================================================================

const SUPABASE_URL = (Deno.env.get("SUPABASE_URL") ?? "").replace(/\/$/, "");
const SERVICE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
const ADMIN_TOKEN = Deno.env.get("ADMIN_TOKEN") ?? "";
const AI_NAME = Deno.env.get("AI_NAME") ?? "LISA";

const REST = `${SUPABASE_URL}/rest/v1`;
const STORAGE = `${SUPABASE_URL}/storage/v1`;
const SERVICE_HEADERS: Record<string, string> = { apikey: SERVICE_KEY, Authorization: `Bearer ${SERVICE_KEY}` };

type Row = Record<string, unknown>;

class HttpError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
const isObject = (v: unknown): v is Row => typeof v === "object" && v !== null && !Array.isArray(v);
const asInt = (v: unknown): number | null => (typeof v === "number" && Number.isInteger(v) ? v : null);
const asStr = (v: unknown, limit = 4000): string | null => (typeof v === "string" ? v.slice(0, limit) : null);

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

async function sha256Hex(s: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function safeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

function newApiKey(): string {   // same shape as Python's secrets.token_urlsafe(32)
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function hexToBytes(hex: string): Uint8Array | null {
  if (hex.length % 2 !== 0 || !/^[0-9a-fA-F]*$/.test(hex)) return null;
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.slice(2 * i, 2 * i + 2), 16);
  return out;
}

// ---------------------------------------------------------------------------
// Supabase REST (PostgREST + Storage) with the service role key
// ---------------------------------------------------------------------------
async function rest(
  method: string,
  table: string,
  opts: { query?: Record<string, string>; body?: unknown; prefer?: string } = {},
): Promise<Row[]> {
  const url = new URL(`${REST}/${table}`);
  for (const [k, v] of Object.entries(opts.query ?? {})) url.searchParams.set(k, v);
  const headers: Record<string, string> = { ...SERVICE_HEADERS };
  if (opts.body !== undefined) headers["Content-Type"] = "application/json";
  if (opts.prefer) headers["Prefer"] = opts.prefer;
  const resp = await fetch(url, { method, headers, body: opts.body === undefined ? undefined : JSON.stringify(opts.body) });
  const text = await resp.text();
  if (!resp.ok) throw new Error(`${method} ${table} -> ${resp.status}: ${text.slice(0, 300)}`);
  if (!text) return [];
  const parsed = JSON.parse(text);
  return Array.isArray(parsed) ? parsed : [parsed];
}

async function one(table: string, query: Record<string, string>): Promise<Row | null> {
  const rows = await rest("GET", table, { query: { select: "*", limit: "1", ...query } });
  return rows[0] ?? null;
}

async function upsertDevice(nodeId: string, deviceKey: string, fields: Row): Promise<Row> {
  const rows = await rest("POST", "uart_devices", {
    query: { on_conflict: "node_id,device_key" },
    body: { node_id: nodeId, device_key: deviceKey, ...fields, last_seen_at: new Date().toISOString() },
    prefer: "return=representation,resolution=merge-duplicates",
  });
  return rows[0];
}

async function signedUrl(bucket: string, path: string, expiresIn = 3600): Promise<string> {
  const objectPath = path.split("/").map(encodeURIComponent).join("/");
  const resp = await fetch(`${STORAGE}/object/sign/${encodeURIComponent(bucket)}/${objectPath}`, {
    method: "POST",
    headers: { ...SERVICE_HEADERS, "Content-Type": "application/json" },
    body: JSON.stringify({ expiresIn }),
  });
  const text = await resp.text();
  if (!resp.ok) throw new Error(`sign ${bucket}/${path} -> ${resp.status}: ${text.slice(0, 200)}`);
  const body = JSON.parse(text);
  const signed: string = body.signedURL ?? body.signedUrl;
  return signed.startsWith("http") ? signed : `${STORAGE}${signed}`;
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------
function bearer(req: Request): string {
  const auth = req.headers.get("authorization") ?? "";
  if (!auth.startsWith("Bearer ")) throw new HttpError(401, "missing bearer token");
  return auth.slice(7).trim();
}

async function requireNode(req: Request): Promise<Row> {
  const key = bearer(req);
  const nodeId = req.headers.get("x-node-id") ?? "";
  const node = /^[A-Za-z0-9_-]{1,40}$/.test(nodeId) ? await one("edge_nodes", { id: `eq.${nodeId}` }) : null;
  if (!node || !safeEqual(String(node.api_key_hash), await sha256Hex(key))) {
    throw new HttpError(401, "unknown node or bad key");
  }
  return node;
}

function requireAdmin(req: Request): void {
  if (!ADMIN_TOKEN) throw new HttpError(403, "admin routes are disabled: set the ADMIN_TOKEN secret");
  if (!safeEqual(bearer(req), ADMIN_TOKEN)) throw new HttpError(401, "bad admin token");
}

// ---------------------------------------------------------------------------
// UART stream analysis (port of server/app/uart_analysis.py)
// ---------------------------------------------------------------------------
interface StreamAnalysis {
  format: string;
  device_type: string;
  intent: string;
  confidence: number;
  fields: string[];
  framing: string;
  source: string;
}

function analysis(format: string, device_type = "unknown", intent = "unknown", confidence = 0.3, fields: string[] = [],
  framing = "unknown"): StreamAnalysis {
  return { format, device_type, intent, confidence, fields, framing, source: "heuristic" };
}

function modbusCrc(data: Uint8Array): number {
  let crc = 0xffff;
  for (const b of data) {
    crc ^= b;
    for (let i = 0; i < 8; i++) crc = crc & 1 ? (crc >> 1) ^ 0xa001 : crc >> 1;
  }
  return crc;
}

const isText = (b: number) => (b >= 0x20 && b < 0x7f) || b === 0x0a || b === 0x0d || b === 0x09;

function analyzeStream(sample: Uint8Array): StreamAnalysis {
  if (sample.length === 0) return analysis("unknown", "unknown", "unknown", 0.0);
  const textRatio = sample.filter(isText).length / sample.length;
  if (textRatio >= 0.9) {
    const lines = new TextDecoder("ascii").decode(sample).split(/\r\n|\r|\n/).map((l) => l.trim()).filter((l) => l);
    if (lines.length === 0) return analysis("ascii_lines", "unknown", "unknown", 0.2, [], "newline");
    const objs: Row[] = [];
    for (const l of lines) {
      if (l.startsWith("{") && l.endsWith("}")) {
        try {
          const o = JSON.parse(l);
          if (isObject(o)) objs.push(o);
        } catch { /* not JSON */ }
      }
    }
    if (objs.length * 2 >= lines.length) {
      const keys = [...new Set(objs.flatMap((o) => Object.keys(o)))].sort();
      return analysis("json_lines", "json_sensor", "telemetry ingestion", 0.8, keys, "newline");
    }
    const nmea = lines.filter((l) => l.startsWith("$") && l.includes("*"));
    if (nmea.length * 2 >= lines.length) {
      const sentences = [...new Set(nmea.map((l) => l.slice(1, 6)))].sort();
      const gps = sentences.some((s) => ["GGA", "RMC", "GLL", "VTG"].includes(s.slice(2)));
      return analysis("nmea", gps ? "gps" : "nmea_device", gps ? "position reporting" : "telemetry", gps ? 0.9 : 0.7,
        sentences, "newline, $...*CS");
    }
    const csv = lines.filter((l) => (l.match(/,/g) ?? []).length >= 2);
    if (csv.length * 2 >= lines.length) {
      const cols = Math.max(...csv.map((l) => (l.match(/,/g) ?? []).length + 1));
      return analysis("csv", "csv_sensor", "periodic measurements", 0.6,
        Array.from({ length: cols }, (_, i) => `col${i}`), "newline, comma-separated");
    }
    return analysis("ascii_lines", "text_console", "status messages", 0.4, [], "newline");
  }
  for (let n = 4; n <= Math.min(sample.length, 256); n++) {
    const crc = modbusCrc(sample.subarray(0, n - 2));
    if (sample[n - 2] === (crc & 0xff) && sample[n - 1] === crc >> 8) {
      return analysis("modbus_rtu", "modbus_slave", "register polling", 0.75,
        [`slave=${sample[0]}`, `function=${sample[1]}`], `RTU frame, ${n} bytes, CRC-16`);
    }
  }
  return analysis("binary", "unknown", "unknown", 0.2, [], "unknown binary");
}

// ---------------------------------------------------------------------------
// Telemetry ingestion (port of server/app/ingest.py). Idempotent on (node_id, device_seq).
// ---------------------------------------------------------------------------
const EVENT_KINDS = new Set(["interaction", "device", "handshake", "observation", "feedback", "ota", "boot", "error"]);
const KIND_BY_RECORD_TYPE: Record<number, string> = {
  1: "interaction", 2: "device", 3: "error", 4: "ota", 5: "boot", 6: "feedback", 7: "observation", 8: "handshake",
};
const STATUSES = new Set(["ok", "nop", "invalid_output", "rejected", "timeout", "error"]);
const BUSES = new Set(["uart", "uart_fast", "espnow", "tcp", "spi"]);
const REWARD_BY_STATUS: Record<string, number> = { ok: 1.0, timeout: 0.3, rejected: 0.1, invalid_output: 0.0 };
const LOG_COLUMNS = ["node_id", "uart_device_id", "device_seq", "event_kind", "act_id", "raw_input", "prompt",
  "model_output", "action", "model_version", "latency_ms", "status", "error_detail", "payload", "reward",
  "used_for_training"];

function deviceFields(d: Row): Row {
  const identified = Boolean(d.identified);
  const fields: Row = {
    identified,
    port: asInt(d.port) || 1,
    baud: asInt(d.baud),
    bus: typeof d.link === "string" && BUSES.has(d.link) ? d.link : "uart",
    data_format: asStr(d.format, 32) || "unknown",
  };
  let manifest: unknown = d.manifest;
  if (manifest == null && typeof d.manifest_text === "string") {
    try {
      manifest = JSON.parse(d.manifest_text);
    } catch {
      manifest = null;
    }
  }
  if (isObject(manifest)) {
    fields.capability_manifest = manifest;
    if (Array.isArray(manifest.needs) && manifest.needs.length) {
      fields.inferred_intent = manifest.needs.map(String).join(", ").slice(0, 200);
    }
  }
  const type = asStr(d.type, 64);
  if (type && type !== "unknown") fields.device_type = type;
  if (typeof d.sample_hex === "string" && d.sample_hex && !identified) {
    const bytes = hexToBytes(d.sample_hex);
    if (bytes) {
      const a = analyzeStream(bytes);
      fields.protocol_schema = a;
      if (!("device_type" in fields)) fields.device_type = a.device_type;
      fields.inferred_intent = a.intent;
    }
  }
  return fields;
}

async function ingestTelemetry(node: Row, batch: Row): Promise<Row> {
  const nodeId = String(node.id);
  if (batch.node_id !== nodeId) throw new HttpError(403, "node_id doesn't match the authenticated node");
  const events = Array.isArray(batch.events) ? batch.events : [];
  if (events.length > 2000) throw new HttpError(422, "too many events in one batch (max 2000)");
  for (const e of events) {
    if (!isObject(e) || !(asInt(e.seq) && (e.seq as number) >= 1) || asInt(e.type) === null || !isObject(e.data)) {
      throw new HttpError(422, "each event needs integer seq >= 1, integer type and an object data");
    }
  }

  // 1. Node status
  const st: Row = isObject(batch.status) ? batch.status : {};
  const parts: Row = isObject(st.partitions) ? st.partitions : {};
  const activeSlot = parts.active_slot === 0 || parts.active_slot === 1 ? parts.active_slot : null;
  const slots = Array.isArray(parts.slots) ? parts.slots : [];
  const slot = activeSlot !== null ? slots[activeSlot as number] : undefined;
  const activeSha = isObject(slot) && typeof slot.sha256 === "string" && slot.sha256 ? slot.sha256 : null;
  const mv = asInt(st.model_version);
  const w25: Row = isObject(st.w25q) ? st.w25q : {};
  const known = mv ? await one("model_versions", { version: `eq.${mv}` }) : null;
  await rest("PATCH", "edge_nodes", {
    query: { id: `eq.${nodeId}` },
    body: {
      fw_version: asStr(batch.fw_version, 64), status: st, partition_state: parts, active_slot: activeSlot,
      active_model_version: known ? mv : null, active_model_sha256: activeSha, w25q_jedec_id: asStr(w25.jedec_id, 16),
      w25q_total_bytes: asInt(w25.size), last_heartbeat_at: new Date().toISOString(),
    },
  });

  // 2. Devices
  const deviceIds = new Map<string, string>();
  const deviceId = async (key: string | null, fields?: Row): Promise<string | null> => {
    if (!key) return null;
    if (fields) {
      deviceIds.set(key, String((await upsertDevice(nodeId, key, fields)).id));
    } else if (!deviceIds.has(key)) {
      const existing = await one("uart_devices", { node_id: `eq.${nodeId}`, device_key: `eq.${key}` });
      deviceIds.set(key, String((existing ?? (await upsertDevice(nodeId, key, {}))).id));
    }
    return deviceIds.get(key) ?? null;
  };
  for (const d of Array.isArray(batch.devices) ? batch.devices : []) {
    if (isObject(d) && typeof d.device_key === "string" && d.device_key) {
      await deviceId(d.device_key.slice(0, 64), deviceFields(d));
    }
  }

  // 3. Events -> interaction_logs rows
  const rows: Row[] = [];
  const feedback: Array<[number, number]> = [];
  for (const e of events as Row[]) {
    const data = e.data as Row;
    const kind = typeof data.kind === "string" && EVENT_KINDS.has(data.kind)
      ? data.kind
      : KIND_BY_RECORD_TYPE[e.type as number] ?? "error";
    const devKey = asStr(data.dev, 64);
    const devId = kind === "device" ? await deviceId(devKey, deviceFields(data)) : await deviceId(devKey);
    const row: Row = Object.fromEntries(LOG_COLUMNS.map((c) => [c, null]));
    Object.assign(row, { node_id: nodeId, device_seq: e.seq, event_kind: kind, payload: data, uart_device_id: devId,
      used_for_training: false });
    if (kind === "interaction") {
      const status = typeof data.status === "string" && STATUSES.has(data.status) ? data.status : "error";
      Object.assign(row, {
        raw_input: asStr(data.input), prompt: asStr(data.prompt), model_output: asStr(data.output),
        action: isObject(data.action) ? data.action : null, model_version: asInt(data.mv), latency_ms: asInt(data.lat),
        status, error_detail: asStr(data.err, 500), act_id: asInt(data.act_id), reward: REWARD_BY_STATUS[status] ?? null,
      });
    } else if (kind === "feedback" && asInt(data.ref_act_id) && (data.value === "good" || data.value === "bad")) {
      feedback.push([data.ref_act_id as number, data.value === "good" ? 1.0 : 0.0]);
    }
    rows.push(row);
  }

  let inserted = 0;
  if (rows.length) {
    inserted = (await rest("POST", "interaction_logs", {
      query: { on_conflict: "node_id,device_seq" },
      body: rows,
      prefer: "return=representation,resolution=ignore-duplicates",
    })).length;
  }
  for (const [actId, reward] of feedback) {   // after the insert: the interaction may be in this same batch
    await rest("PATCH", "interaction_logs", {
      query: { node_id: `eq.${nodeId}`, act_id: `eq.${actId}`, event_kind: "eq.interaction" },
      body: { reward },
    });
  }
  const ackedSeq = events.reduce((m: number, e) => Math.max(m, (e as Row).seq as number), 0);
  return { acked_seq: ackedSeq, inserted, rediscover: false };
}

// ---------------------------------------------------------------------------
// OTA check
// ---------------------------------------------------------------------------
async function otaCheck(node: Row, params: URLSearchParams): Promise<Row> {
  if (params.get("node_id") !== node.id) throw new HttpError(403, "node_id doesn't match the authenticated node");
  const current = (params.get("current_sha256") ?? "").toLowerCase();
  const none = { update: false, rollback: false, kind: null, version: null, size: null, sha256: null, signature: null,
    url: null };
  const active = await one("model_versions", { is_active: "is.true" });
  if (!active || current === active.sha256) return { ...none, version: active ? active.version : null };
  let kind = "full";
  let url = await signedUrl("models", String(active.storage_path));
  if (active.delta_from_version && active.delta_storage_path) {
    const base = await one("model_versions", { version: `eq.${active.delta_from_version}` });
    if (base && base.sha256 === current) {
      kind = "delta";
      url = await signedUrl("deltas", String(active.delta_storage_path));
    }
  }
  return { update: true, rollback: false, kind, version: active.version, size: active.size_bytes, sha256: active.sha256,
    signature: active.signature ?? null, url };
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------
function routeOf(pathname: string): string {
  // Supabase passes "/lisa-api/api/v1/..."; locally it may be "/api/v1/...".
  const i = pathname.indexOf("/api/v1/");
  if (i >= 0) return pathname.slice(i);
  if (pathname.endsWith("/healthz")) return "/healthz";
  return "/";
}

async function readJson(req: Request): Promise<Row> {
  try {
    const body = await req.json();
    if (isObject(body)) return body;
  } catch { /* fall through */ }
  throw new HttpError(422, "body must be a JSON object");
}

async function handle(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const route = routeOf(url.pathname);
  const m = req.method;

  if (route === "/" && (m === "GET" || m === "HEAD")) {
    return m === "HEAD" ? new Response(null, { status: 200 })
      : json({ name: `${AI_NAME} Evolution Server`, status: "running", health: "healthz" });
  }
  if (route === "/healthz" && m === "GET") {
    const active = await one("model_versions", { is_active: "is.true" });
    return json({ ok: true, ai: AI_NAME, dev_mode: false, runtime: "supabase-edge-function", trainer: "github-actions",
      teacher: "rule", active_model: active ? active.version : null });
  }

  // ---- device routes (node key) ----
  if (route === "/api/v1/telemetry" && m === "POST") {
    const node = await requireNode(req);
    return json(await ingestTelemetry(node, await readJson(req)));
  }
  if (route === "/api/v1/ota/check" && m === "GET") {
    const node = await requireNode(req);
    return json(await otaCheck(node, url.searchParams));
  }
  if (route === "/api/v1/uart/classify" && m === "POST") {
    const node = await requireNode(req);
    const body = await readJson(req);
    if (body.node_id !== node.id) throw new HttpError(403, "node_id doesn't match the authenticated node");
    const key = asStr(body.device_key, 64);
    const hex = asStr(body.sample_hex, 1024) ?? "";
    if (!key || !/^[0-9a-fA-F]*$/.test(hex)) throw new HttpError(422, "device_key and a hex sample_hex are required");
    const result = analyzeStream(hexToBytes(hex) ?? new Uint8Array());
    await upsertDevice(String(node.id), key, {
      baud: asInt(body.baud) || null, data_format: result.format, device_type: result.device_type,
      inferred_intent: result.intent, protocol_schema: result,
    });
    return json(result);
  }

  // ---- admin routes (ADMIN_TOKEN) ----
  if (route === "/api/v1/evolution/trigger" && m === "POST") {
    requireAdmin(req);
    const body = await readJson(req).catch(() => ({} as Row));
    const params: Row = {};
    if (body.force_publish === true) params.force_publish = true;
    for (const k of ["trainer", "method"]) if (typeof body[k] === "string") params[k] = body[k];
    if (asInt(body.steps)) params.steps = body.steps;
    const [run] = await rest("POST", "evolution_runs", {
      body: { trigger: "manual", status: "queued", trainer: (params.trainer as string) ?? "torch", params },
      prefer: "return=representation",
    });
    return json({ run_id: run.id, status: "queued", executor: "worker" });
  }
  if (route === "/api/v1/evolution/runs" && m === "GET") {
    requireAdmin(req);
    const limit = Math.min(Number(url.searchParams.get("limit") ?? 20) || 20, 200);
    return json(await rest("GET", "evolution_runs", { query: { select: "*", order: "created_at.desc", limit: `${limit}` } }));
  }
  if (route === "/api/v1/admin/nodes" && m === "POST") {
    requireAdmin(req);
    const body = await readJson(req);
    const nodeId = asStr(body.node_id, 40);
    if (!nodeId || !/^[A-Za-z0-9_-]{1,40}$/.test(nodeId)) throw new HttpError(422, "node_id: 1-40 of A-Z a-z 0-9 _ -");
    if (await one("edge_nodes", { id: `eq.${nodeId}` })) throw new HttpError(409, "node already registered");
    const apiKey = newApiKey();
    await rest("POST", "edge_nodes", {
      body: { id: nodeId, display_name: asStr(body.display_name, 120), api_key_hash: await sha256Hex(apiKey) },
    });
    return json({ node_id: nodeId, api_key: apiKey });
  }
  const activate = route.match(/^\/api\/v1\/admin\/models\/(\d+)\/activate$/);
  if (activate && m === "POST") {
    requireAdmin(req);
    const version = Number(activate[1]);
    if (!(await one("model_versions", { version: `eq.${version}` }))) throw new HttpError(404, "unknown model version");
    await rest("POST", "rpc/activate_model_version", { body: { p_version: version } });
    return json({ active_version: version });
  }

  throw new HttpError(404, "not found");
}

Deno.serve(async (req: Request) => {
  try {
    return await handle(req);
  } catch (e) {
    if (e instanceof HttpError) return json({ detail: e.message }, e.status);
    console.error(e);
    return json({ detail: "internal error" }, 500);
  }
});
