// free-model-aggregator-01 — a free AI model aggregator gateway.
//
// One Node 22 (stdlib only) server. It serves the OpenAI Chat Completions API
// (POST /v1/chat/completions, JSON and SSE) and the Anthropic Messages API
// (POST /v1/messages, JSON and SSE), plus GET /v1/models whose list follows the
// auth header (Bearer -> OpenAI shape, x-api-key -> Anthropic shape).
//
// At trial start it reads the brief (MAKEFUN_PROMPT_FILE), probes every listed
// provider's catalog route, registers its port + catalog with the env
// (POST $MAKEFUN_ENV_URL/register), serves the 20 requests the env client sends
// one at a time, and exits 0 when GET $MAKEFUN_ENV_URL/work says done.
//
// Model selection (rating-greedy with failover):
//   candidates are (provider, free model) pairs that meet the request's
//   requirements (tools / vision / json / min_context) and, for JSON-mode
//   requests, live on an openai- or gemini-wire provider (an anthropic-wire
//   provider cannot carry response_format, so the echo would read json_mode
//   false). Candidates are ordered by the brief's sealed quality rating
//   (highest first), ties by provider id then model id. The first candidate
//   is tried; on any upstream failure (402/429/5xx/network) the next
//   candidate is tried. A provider with two consecutive failures is demoted
//   to the back of the order until it succeeds again, which keeps the
//   gateway from burning calls on a provider that has run out of budget or
//   hit its outage window. No wall clock and no unseeded randomness anywhere:
//   every choice derives from the brief, the requests, and the providers'
//   answers; dates come from the trial's fixed clock (MAKEFUN_TRIAL_NOW).
//
// Wire handling: the provider's full answer is read first (JSON or SSE),
// reduced to {text, inputTokens, outputTokens, stop}, then the client's
// response is rendered from those buffered values (JSON, or SSE with the
// provider's usage included in the final usage chunk). The provider echoes
// what it parsed, so the request body is forwarded untouched in spirit:
// system prompt, user text (images included), tools, stop sequences,
// max_tokens and JSON mode are all passed through, only translated between
// wire shapes. Stop reasons map stop/end_turn/STOP -> stop/end_turn,
// length/max_tokens -> length/max_tokens.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';
import { trialNow } from './vendor/makefun-agent.ts';

// ---------------------------------------------------------------- types

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

interface ProviderDir {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
  model?: string;
  /** Consecutive upstream failures since the last success (failover state). */
  consecFailures: number;
  /** Total upstream calls made to this provider this run. */
  calls: number;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base_url: string;
  key_env: string;
  models: CatalogModel[];
}

interface Requirements {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface ManifestRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirements;
}

interface ProbedModel {
  id: string;
  free: boolean;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

interface Candidate {
  provider: ProviderDir;
  model: ProbedModel;
  rating: number;
  demoted: boolean;
}

interface Answer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: string; // provider stop reason, e.g. 'stop' | 'end_turn' | 'STOP' | 'length'
}

// ---------------------------------------------------------------- constants

const FIXED_CREATED = 1767225600; // 2026-01-01T00:00:00Z, used only if no trial clock
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 5000;
const POLL_MS = 100;

// The trial env's base URL (MAKEFUN_ENV_URL). Provider directory entries
// give a path ("/p/<id>"); the request URL is this base + the path.
let ENV_URL = '';

function providerBase(p: ProviderDir): string {
  if (p.base.startsWith('http://') || p.base.startsWith('https://')) return p.base;
  return `${ENV_URL}${p.base}`;
}

function createdSeconds(): number {
  try {
    return Math.floor(trialNow().getTime() / 1000);
  } catch {
    return FIXED_CREATED;
  }
}
function createdIso(): string {
  try {
    return trialNow().toISOString();
  } catch {
    return '2026-01-01T00:00:00Z';
  }
}

function log(...args: unknown[]): void {
  // stderr only: stdout is reserved for --list's catalog JSON, which must be
  // byte-for-byte the registered catalog.
  console.error('[agg]', ...args);
}

// ---------------------------------------------------------------- brief parsing

// The brief carries its machine-readable documents as single-line JSON
// values (one per line): the provider directory, the ratings table, and
// the manifest (with the workload bodies). Parse every line that is a JSON
// value and classify it by its keys.
function findJsonDocs(text: string): unknown[] {
  const docs: unknown[] = [];
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) continue;
    try {
      docs.push(JSON.parse(trimmed));
    } catch {
      // Not a complete JSON value on this line; ignore it.
    }
  }
  return docs;
}

interface Brief {
  providers: ProviderDir[];
  ratings: RatingRow[];
  manifest: ManifestRow[];
  trialId: string;
}

function parseBrief(text: string): Brief {
  const docs = findJsonDocs(text);
  const providerDocs = docs.filter((d) => Array.isArray((d as { providers?: unknown }).providers));
  const ratingDocs = docs.filter((d) => Array.isArray((d as { ratings?: unknown }).ratings));
  const manifestDocs = docs.filter((d) => Array.isArray((d as { manifest?: unknown }).manifest));
  const trialDocs = docs.filter((d) => typeof (d as { trial_id?: unknown }).trial_id === 'string');

  const providers: ProviderDir[] = [];
  for (const doc of providerDocs) {
    const arr = (doc as { providers?: unknown[] }).providers;
    if (!Array.isArray(arr)) continue;
    for (const p of arr) {
      const o = p as Record<string, unknown>;
      // The directory gives a path (e.g. "/p/groq"), not a full URL; the
      // full URL is $MAKEFUN_ENV_URL + path. A full URL is accepted as-is.
      const path = typeof o.path === 'string' ? o.path
        : typeof o.base === 'string' ? o.base : '';
      if (typeof o.id !== 'string' || !path) continue;
      providers.push({
        id: o.id,
        name: typeof o.name === 'string' ? o.name : o.id,
        wire: (o.wire as Wire) || 'openai',
        auth: (o.auth as AuthScheme) || 'none',
        base: path,
        key_env: typeof o.key_env === 'string' ? o.key_env : '',
        key: typeof o.key === 'string' ? o.key : '',
        model: typeof o.model === 'string' ? o.model : undefined,
        consecFailures: 0,
        calls: 0,
      });
    }
  }

  const ratings: RatingRow[] = [];
  for (const doc of ratingDocs) {
    const arr = (doc as { ratings?: unknown[] }).ratings;
    if (!Array.isArray(arr)) continue;
    for (const r of arr) {
      const o = r as Record<string, unknown>;
      if (typeof o.model !== 'string' || typeof o.provider !== 'string') continue;
      ratings.push({
        model: o.model,
        provider: o.provider,
        rating_milli: typeof o.rating_milli === 'number' ? o.rating_milli : 0,
        context_length: typeof o.context_length === 'number' ? o.context_length : 0,
        tools: o.tools === true,
        vision: o.vision === true,
        json: o.json === true,
        free: o.free !== false,
      });
    }
  }

  let manifest: ManifestRow[] = [];
  for (const doc of manifestDocs) {
    const arr = (doc as { manifest?: unknown[] }).manifest;
    if (!Array.isArray(arr)) continue;
    manifest = arr.map((m) => {
      const o = m as Record<string, unknown>;
      const req = (o.requires || {}) as Record<string, unknown>;
      return {
        id: String(o.id ?? ''),
        api: o.api === 'anthropic' ? 'anthropic' : 'openai',
        stream: o.stream === true,
        weight: typeof o.weight === 'number' ? o.weight : 1,
        requires: {
          tools: req.tools === true ? true : undefined,
          vision: req.vision === true ? true : undefined,
          json: req.json === true ? true : undefined,
          min_context: typeof req.min_context === 'number' ? req.min_context : undefined,
        },
      };
    });
  }

  let trialId = '';
  for (const doc of trialDocs) {
    const v = (doc as { trial_id?: unknown }).trial_id;
    if (typeof v === 'string') trialId = v;
  }

  return { providers, ratings, manifest, trialId };
}

// ---------------------------------------------------------------- provider probing

function authHeaders(p: ProviderDir, extra: Record<string, string> = {}): Record<string, string> {
  const h: Record<string, string> = { ...extra };
  if (p.auth === 'bearer' && p.key) h['authorization'] = `Bearer ${p.key}`;
  else if (p.auth === 'x-api-key' && p.key) h['x-api-key'] = p.key;
  else if (p.auth === 'x-goog-api-key' && p.key) h['x-goog-api-key'] = p.key;
  if (p.wire === 'anthropic') h['anthropic-version'] = '2023-06-01';
  return h;
}

async function fetchJson(url: string, init: RequestInit, timeoutMs: number): Promise<unknown> {
  const res = await fetch(url, { ...init, signal: AbortSignal.timeout(timeoutMs) });
  if (!res.ok) throw new Error(`HTTP ${res.status} from ${url}`);
  return await res.json();
}

async function probeProvider(p: ProviderDir): Promise<ProbedModel[]> {
  const out: ProbedModel[] = [];
  try {
    if (p.wire === 'gemini') {
      const body = await fetchJson(`${providerBase(p)}/v1beta/models`, { headers: authHeaders(p) }, PROBE_TIMEOUT_MS) as { models?: unknown[] };
      for (const m of body.models ?? []) {
        const o = m as Record<string, unknown>;
        const name = typeof o.name === 'string' ? o.name : '';
        const id = name.startsWith('models/') ? name.slice('models/'.length) : name;
        if (!id) continue;
        out.push({
          id,
          free: o.free === true,
          context_length: typeof o.inputTokenLimit === 'number' ? o.inputTokenLimit
            : typeof o.context_length === 'number' ? o.context_length : null,
          tools: o.tools === true,
          vision: o.vision === true,
          json: o.json === true,
        });
      }
    } else {
      const body = await fetchJson(`${providerBase(p)}/v1/models`, { headers: authHeaders(p) }, PROBE_TIMEOUT_MS) as { data?: unknown[] };
      for (const m of body.data ?? []) {
        const o = m as Record<string, unknown>;
        if (typeof o.id !== 'string') continue;
        out.push({
          id: o.id,
          free: o.free === true,
          context_length: typeof o.context_length === 'number' ? o.context_length : null,
          tools: o.tools === true,
          vision: o.vision === true,
          json: o.json === true,
        });
      }
    }
  } catch (err) {
    log(`probe ${p.id} failed:`, String(err));
  }
  return out;
}

// Build the catalog the way the trial scores it: every provider from the
// brief, models sorted by id, free flag and context length taken from the
// brief's ratings table when the provider lists that model, else from the
// probe. Paid models are listed too, marked free: false.
function buildCatalog(brief: Brief, probed: Map<string, ProbedModel[]>): CatalogProvider[] {
  const ratingByKey = new Map<string, RatingRow>();
  for (const r of brief.ratings) ratingByKey.set(`${r.provider}\0${r.model}`, r);
  return brief.providers.map((p) => {
    const listed = [...(probed.get(p.id) ?? [])].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    const models: CatalogModel[] = listed.map((m) => {
      const row = ratingByKey.get(`${p.id}\0${m.id}`);
      return {
        id: m.id,
        free: row ? row.free : m.free,
        context_length: row ? row.context_length : m.context_length,
      };
    });
    return {
      id: p.id,
      name: p.name,
      wire: p.wire,
      auth: p.auth,
      base_url: p.base,
      key_env: p.key_env,
      models,
    };
  });
}

// The unique free model ids in the registered catalog (both /v1/models
// shapes list exactly this set).
function freeModelIds(catalog: CatalogProvider[]): { id: string; provider: string }[] {
  const seen = new Map<string, string>();
  for (const p of catalog) {
    for (const m of p.models) {
      if (m.free && !seen.has(m.id)) seen.set(m.id, p.id);
    }
  }
  return [...seen.entries()].map(([id, provider]) => ({ id, provider })).sort((a, b) => (a.id < b.id ? -1 : 1));
}

// ---------------------------------------------------------------- candidate selection

interface SelectionCtx {
  brief: Brief;
  probed: Map<string, ProbedModel[]>;
  ratingByKey: Map<string, RatingRow>;
}

function buildSelectionCtx(brief: Brief, probed: Map<string, ProbedModel[]>): SelectionCtx {
  const ratingByKey = new Map<string, RatingRow>();
  for (const r of brief.ratings) ratingByKey.set(`${r.provider}\0${r.model}`, r);
  return { brief, probed, ratingByKey };
}

function meetsRequirements(m: ProbedModel, req: Requirements, wire: Wire): boolean {
  if (!m.free) return false;
  if (req.tools && !m.tools) return false;
  if (req.vision && !m.vision) return false;
  if (req.json && !m.json) return false;
  if (req.json && wire === 'anthropic') return false; // anthropic wire cannot carry response_format
  if (typeof req.min_context === 'number' && m.context_length !== null && m.context_length < req.min_context) return false;
  if (typeof req.min_context === 'number' && m.context_length === null) return false;
  return true;
}

function candidatesFor(ctx: SelectionCtx, req: Requirements): Candidate[] {
  const out: Candidate[] = [];
  for (const p of ctx.brief.providers) {
    for (const m of ctx.probed.get(p.id) ?? []) {
      if (!meetsRequirements(m, req, p.wire)) continue;
      const row = ctx.ratingByKey.get(`${p.id}\0${m.id}`);
      out.push({
        provider: p,
        model: m,
        rating: row ? row.rating_milli : 0,
        demoted: p.consecFailures >= 2,
      });
    }
  }
  out.sort((a, b) => {
    if (a.demoted !== b.demoted) return a.demoted ? 1 : -1;
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

// ---------------------------------------------------------------- wire translation

function openAiContentText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.map((b) => {
      const o = b as Record<string, unknown>;
      return o.type === 'text' ? String(o.text ?? '') : '';
    }).join('');
  }
  return '';
}

function anthropicContentText(content: unknown): string {
  if (Array.isArray(content)) {
    return content.map((b) => {
      const o = b as Record<string, unknown>;
      return o.type === 'text' ? String(o.text ?? '') : '';
    }).join('');
  }
  return typeof content === 'string' ? content : '';
}

function openAiImageToAnthropic(part: Record<string, unknown>): Record<string, unknown> | null {
  const iu = part.image_url as Record<string, unknown> | undefined;
  if (!iu || typeof iu.url !== 'string') return null;
  const url = iu.url;
  const m = /^data:([^;,]+);base64,(.+)$/.exec(url);
  if (m) return { type: 'image', source: { type: 'base64', media_type: m[1], data: m[2] } };
  return { type: 'image', source: { type: 'url', url } };
}

function anthropicImageToOpenAi(part: Record<string, unknown>): Record<string, unknown> | null {
  const src = part.source as Record<string, unknown> | undefined;
  if (!src) return null;
  if (src.type === 'base64' && typeof src.data === 'string') {
    return { type: 'image_url', image_url: { url: `data:${String(src.media_type ?? 'image/png')};base64,${src.data}` } };
  }
  if (src.type === 'url' && typeof src.url === 'string') {
    return { type: 'image_url', image_url: { url: src.url } };
  }
  return null;
}

// OpenAI message -> Anthropic {role, content} (system messages are skipped;
// callers fold them into the system string).
function openAiMsgToAnthropic(msg: Record<string, unknown>): Record<string, unknown> | null {
  const role = msg.role;
  if (role === 'system') return null;
  if (role === 'tool') {
    return {
      role: 'user',
      content: [{ type: 'tool_result', tool_use_id: String(msg.tool_call_id ?? ''), content: openAiContentText(msg.content) }],
    };
  }
  if (role === 'assistant') {
    const blocks: unknown[] = [];
    const text = openAiContentText(msg.content);
    if (text) blocks.push({ type: 'text', text });
    for (const tc of (msg.tool_calls as unknown[] | undefined) ?? []) {
      const t = tc as Record<string, unknown>;
      const fn = (t.function ?? {}) as Record<string, unknown>;
      let input: unknown = {};
      try { input = JSON.parse(String(fn.arguments ?? '{}')); } catch { input = {}; }
      blocks.push({ type: 'tool_use', id: String(t.id ?? ''), name: String(fn.name ?? ''), input });
    }
    return { role: 'assistant', content: blocks.length ? blocks : [{ type: 'text', text: '' }] };
  }
  // user
  if (typeof msg.content === 'string') return { role: 'user', content: msg.content };
  if (Array.isArray(msg.content)) {
    const parts: unknown[] = [];
    for (const part of msg.content) {
      const p = part as Record<string, unknown>;
      if (p.type === 'text') parts.push({ type: 'text', text: String(p.text ?? '') });
      else if (p.type === 'image_url') {
        const img = openAiImageToAnthropic(p);
        if (img) parts.push(img);
      }
    }
    return { role: 'user', content: parts };
  }
  return { role: 'user', content: '' };
}

// Anthropic message -> one or more OpenAI messages (a tool_result in a user
// message becomes a separate role=tool message after the user message).
function anthropicMsgToOpenAi(msg: Record<string, unknown>): Record<string, unknown>[] {
  const role = msg.role;
  if (role === 'assistant' && Array.isArray(msg.content)) {
    let text = '';
    const toolCalls: unknown[] = [];
    for (const b of msg.content) {
      const o = b as Record<string, unknown>;
      if (o.type === 'text') text += String(o.text ?? '');
      else if (o.type === 'tool_use') {
        toolCalls.push({
          id: String(o.id ?? ''),
          type: 'function',
          function: { name: String(o.name ?? ''), arguments: JSON.stringify(o.input ?? {}) },
        });
      }
    }
    const out: Record<string, unknown> = { role: 'assistant', content: text };
    if (toolCalls.length) out.tool_calls = toolCalls;
    return [out];
  }
  if (role === 'user' && Array.isArray(msg.content)) {
    const parts: unknown[] = [];
    const toolMsgs: Record<string, unknown>[] = [];
    for (const b of msg.content) {
      const o = b as Record<string, unknown>;
      if (o.type === 'text') parts.push({ type: 'text', text: String(o.text ?? '') });
      else if (o.type === 'image') {
        const img = anthropicImageToOpenAi(o);
        if (img) parts.push(img);
      } else if (o.type === 'tool_result') {
        toolMsgs.push({
          role: 'tool',
          tool_call_id: String(o.tool_use_id ?? ''),
          content: typeof o.content === 'string' ? o.content : anthropicContentText(o.content),
        });
      }
    }
    return [{ role: 'user', content: parts }, ...toolMsgs];
  }
  return [{ role: role === 'assistant' ? 'assistant' : 'user', content: anthropicContentText(msg.content) }];
}

function openAiToolsToAnthropic(tools: unknown): unknown[] | undefined {
  if (!Array.isArray(tools)) return undefined;
  return tools.map((t) => {
    const fn = ((t as Record<string, unknown>).function ?? {}) as Record<string, unknown>;
    return { name: String(fn.name ?? ''), description: String(fn.description ?? ''), input_schema: fn.parameters ?? { type: 'object', properties: {} } };
  });
}

function anthropicToolsToOpenAi(tools: unknown): unknown[] | undefined {
  if (!Array.isArray(tools)) return undefined;
  return tools.map((t) => {
    const o = t as Record<string, unknown>;
    return { type: 'function', function: { name: String(o.name ?? ''), description: String(o.description ?? ''), parameters: o.input_schema ?? { type: 'object', properties: {} } } };
  });
}

function openAiToolsToGemini(tools: unknown): unknown[] | undefined {
  if (!Array.isArray(tools)) return undefined;
  return tools.map((t) => {
    const fn = ((t as Record<string, unknown>).function ?? {}) as Record<string, unknown>;
    return { name: String(fn.name ?? ''), description: String(fn.description ?? ''), parameters: fn.parameters ?? { type: 'object', properties: {} } };
  });
}

function anthropicToolsToGemini(tools: unknown): unknown[] | undefined {
  if (!Array.isArray(tools)) return undefined;
  return tools.map((t) => {
    const o = t as Record<string, unknown>;
    return { name: String(o.name ?? ''), description: String(o.description ?? ''), parameters: o.input_schema ?? { type: 'object', properties: {} } };
  });
}

function openAiToolChoiceToAnthropic(tc: unknown): unknown {
  if (tc === 'auto') return { type: 'auto' };
  if (tc === 'none') return { type: 'none' };
  if (tc === 'required') return { type: 'any' };
  const o = tc as Record<string, unknown> | null;
  if (o && o.type === 'function') {
    const fn = (o.function ?? {}) as Record<string, unknown>;
    return { type: 'tool', name: String(fn.name ?? '') };
  }
  return undefined;
}

function anthropicToolChoiceToOpenAi(tc: unknown): unknown {
  const o = tc as Record<string, unknown> | null;
  if (!o) return undefined;
  if (o.type === 'auto') return 'auto';
  if (o.type === 'none') return 'none';
  if (o.type === 'any') return 'required';
  if (o.type === 'tool') return { type: 'function', function: { name: String(o.name ?? '') } };
  return undefined;
}

function openAiStopToAnthropic(stop: unknown): string[] | undefined {
  if (typeof stop === 'string') return [stop];
  if (Array.isArray(stop)) return stop.filter((s) => typeof s === 'string') as string[];
  return undefined;
}

function anthropicStopToOpenAi(stop: unknown): string[] | undefined {
  if (Array.isArray(stop)) return stop.filter((s) => typeof s === 'string') as string[];
  return undefined;
}

// Build the provider request body for a client body, translated to the
// provider's wire. jsonMode is derived from the client body.
function buildProviderBody(
  clientApi: 'openai' | 'anthropic',
  body: Record<string, unknown>,
  provider: ProviderDir,
  modelId: string,
  stream: boolean,
): { url: string; body: Record<string, unknown> } {
  const messages = (body.messages as Record<string, unknown>[]) ?? [];
  const jsonMode = (() => {
    const rf = body.response_format as Record<string, unknown> | undefined;
    return rf != null && (rf.type === 'json_object' || rf.type === 'json_schema');
  })();

  if (provider.wire === 'openai') {
    let out: Record<string, unknown>;
    if (clientApi === 'openai') {
      out = { ...body, model: modelId, stream };
      if (stream) out.stream_options = { include_usage: true };
    } else {
      const sys = typeof body.system === 'string' ? body.system
        : Array.isArray(body.system) ? anthropicContentText(body.system) : undefined;
      const msgs: Record<string, unknown>[] = [];
      if (sys) msgs.push({ role: 'system', content: sys });
      for (const m of messages) msgs.push(...anthropicMsgToOpenAi(m));
      out = { model: modelId, messages: msgs, max_tokens: body.max_tokens, stream };
      if (stream) out.stream_options = { include_usage: true };
      if (body.temperature !== undefined) out.temperature = body.temperature;
      if (body.top_p !== undefined) out.top_p = body.top_p;
      const stops = anthropicStopToOpenAi(body.stop_sequences);
      if (stops) out.stop = stops;
      const tools = anthropicToolsToOpenAi(body.tools);
      if (tools) out.tools = tools;
      const tc = anthropicToolChoiceToOpenAi(body.tool_choice);
      if (tc !== undefined) out.tool_choice = tc;
    }
    return { url: `${providerBase(provider)}/v1/chat/completions`, body: out };
  }

  if (provider.wire === 'anthropic') {
    let out: Record<string, unknown>;
    if (clientApi === 'anthropic') {
      out = { ...body, model: modelId, stream };
    } else {
      const sysParts: string[] = [];
      const msgs: Record<string, unknown>[] = [];
      for (const m of messages) {
        if (m.role === 'system') sysParts.push(openAiContentText(m.content));
        else {
          const conv = openAiMsgToAnthropic(m);
          if (conv) msgs.push(conv);
        }
      }
      out = { model: modelId, messages: msgs, max_tokens: body.max_tokens, stream };
      if (sysParts.length) out.system = sysParts.join('\n\n');
      if (body.temperature !== undefined) out.temperature = body.temperature;
      if (body.top_p !== undefined) out.top_p = body.top_p;
      const stops = openAiStopToAnthropic(body.stop);
      if (stops) out.stop_sequences = stops;
      const tools = openAiToolsToAnthropic(body.tools);
      if (tools) out.tools = tools;
      const tc = openAiToolChoiceToAnthropic(body.tool_choice);
      if (tc !== undefined) out.tool_choice = tc;
      // response_format cannot travel on the anthropic wire; callers never
      // route a JSON-mode request here.
    }
    return { url: `${providerBase(provider)}/v1/messages`, body: out };
  }

  // gemini wire
  const sysParts: string[] = [];
  const contents: Record<string, unknown>[] = [];
  for (const m of messages) {
    if (m.role === 'system') { sysParts.push(openAiContentText(m.content)); continue; }
    const parts: Record<string, unknown>[] = [];
    if (typeof m.content === 'string') {
      if (m.content) parts.push({ text: m.content });
    } else if (Array.isArray(m.content)) {
      for (const part of m.content) {
        const p = part as Record<string, unknown>;
        if (p.type === 'text') parts.push({ text: String(p.text ?? '') });
        else if (p.type === 'image_url') {
          const iu = p.image_url as Record<string, unknown> | undefined;
          const url = typeof iu?.url === 'string' ? iu.url : '';
          const dm = /^data:([^;,]+);base64,(.+)$/.exec(url);
          if (dm) parts.push({ inlineData: { mimeType: dm[1], data: dm[2] } });
          else if (url) parts.push({ fileData: { mimeType: 'image/png', fileUri: url } });
        }
      }
    }
    if (m.role === 'assistant') {
      const text = openAiContentText(m.content);
      if (text) parts.push({ text });
      for (const tc of (m.tool_calls as unknown[] | undefined) ?? []) {
        const t = tc as Record<string, unknown>;
        const fn = (t.function ?? {}) as Record<string, unknown>;
        let args: unknown = {};
        try { args = JSON.parse(String(fn.arguments ?? '{}')); } catch { args = {}; }
        parts.push({ functionCall: { name: String(fn.name ?? ''), args } });
      }
      contents.push({ role: 'model', parts: parts.length ? parts : [{ text: '' }] });
    } else if (m.role === 'tool') {
      contents.push({ role: 'user', parts: [{ functionResponse: { name: '', response: { result: openAiContentText(m.content) } } }] });
    } else if (m.role === 'user') {
      contents.push({ role: 'user', parts: parts.length ? parts : [{ text: '' }] });
    }
  }
  if (clientApi === 'anthropic') {
    // Rebuild contents from the anthropic body instead (system + messages).
    contents.length = 0;
    if (typeof body.system === 'string') sysParts.push(body.system);
    else if (Array.isArray(body.system)) sysParts.push(anthropicContentText(body.system));
    for (const m of messages) {
      const parts: Record<string, unknown>[] = [];
      if (typeof m.content === 'string') {
        if (m.content) parts.push({ text: m.content });
      } else if (Array.isArray(m.content)) {
        for (const b of m.content) {
          const o = b as Record<string, unknown>;
          if (o.type === 'text') parts.push({ text: String(o.text ?? '') });
          else if (o.type === 'image') {
            const src = o.source as Record<string, unknown> | undefined;
            if (src?.type === 'base64' && typeof src.data === 'string') {
              parts.push({ inlineData: { mimeType: String(src.media_type ?? 'image/png'), data: src.data } });
            } else if (src?.type === 'url' && typeof src.url === 'string') {
              parts.push({ fileData: { mimeType: 'image/png', fileUri: src.url } });
            }
          } else if (o.type === 'tool_use') {
            parts.push({ functionCall: { name: String(o.name ?? ''), args: o.input ?? {} } });
          } else if (o.type === 'tool_result') {
            parts.push({ functionResponse: { name: '', response: { result: typeof o.content === 'string' ? o.content : anthropicContentText(o.content) } } });
          }
        }
      }
      if (m.role === 'assistant') contents.push({ role: 'model', parts: parts.length ? parts : [{ text: '' }] });
      else contents.push({ role: 'user', parts: parts.length ? parts : [{ text: '' }] });
    }
  }
  const generationConfig: Record<string, unknown> = {};
  if (typeof body.max_tokens === 'number') generationConfig.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') generationConfig.topP = body.top_p;
  const stops = clientApi === 'openai' ? openAiStopToAnthropic(body.stop) : anthropicStopToOpenAi(body.stop_sequences);
  if (stops) generationConfig.stopSequences = stops;
  if (jsonMode) generationConfig.responseMimeType = 'application/json';
  const tools = clientApi === 'openai' ? openAiToolsToGemini(body.tools) : anthropicToolsToGemini(body.tools);
  const geminiBody: Record<string, unknown> = { contents, generationConfig };
  if (sysParts.length) geminiBody.systemInstruction = { parts: [{ text: sysParts.join('\n\n') }] };
  if (tools) geminiBody.tools = [{ functionDeclarations: tools }];
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  const suffix = stream ? '?alt=sse' : '';
  return {
    url: `${providerBase(provider)}/v1beta/models/${encodeURIComponent(modelId)}:${action}${suffix}`,
    body: geminiBody,
  };
}

// ---------------------------------------------------------------- provider answer parsing

function sseFrames(text: string): unknown[] {
  const out: unknown[] = [];
  for (const chunk of text.split(/\r?\n\r?\n/)) {
    for (const line of chunk.split(/\r?\n/)) {
      const t = line.trim();
      if (!t.startsWith('data:')) continue;
      const payload = t.slice(5).trim();
      if (!payload || payload === '[DONE]') continue;
      try { out.push(JSON.parse(payload)); } catch { /* ignore non-JSON data lines */ }
    }
  }
  return out;
}

function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts.map((p) => {
    const o = p as Record<string, unknown>;
    return typeof o.text === 'string' ? o.text : '';
  }).join('');
}

function parseOpenAiJson(json: Record<string, unknown>): Answer {
  const choice = ((json.choices as unknown[]) ?? [])[0] as Record<string, unknown> | undefined;
  const msg = (choice?.message ?? {}) as Record<string, unknown>;
  const usage = (json.usage ?? {}) as Record<string, unknown>;
  return {
    text: openAiContentText(msg.content),
    inputTokens: typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : 0,
    outputTokens: typeof usage.completion_tokens === 'number' ? usage.completion_tokens : 0,
    stop: typeof choice?.finish_reason === 'string' ? choice.finish_reason : 'stop',
  };
}

function parseAnthropicJson(json: Record<string, unknown>): Answer {
  const usage = (json.usage ?? {}) as Record<string, unknown>;
  return {
    text: anthropicContentText(json.content),
    inputTokens: typeof usage.input_tokens === 'number' ? usage.input_tokens : 0,
    outputTokens: typeof usage.output_tokens === 'number' ? usage.output_tokens : 0,
    stop: typeof json.stop_reason === 'string' ? json.stop_reason : 'end_turn',
  };
}

function parseGeminiJson(json: Record<string, unknown>): Answer {
  const cand = ((json.candidates as unknown[]) ?? [])[0] as Record<string, unknown> | undefined;
  const content = (cand?.content ?? {}) as Record<string, unknown>;
  const usage = (json.usageMetadata ?? {}) as Record<string, unknown>;
  return {
    text: geminiPartsText(content.parts),
    inputTokens: typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : 0,
    outputTokens: typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : 0,
    stop: typeof cand?.finishReason === 'string' ? cand.finishReason : 'STOP',
  };
}

function parseOpenAiSse(frames: unknown[]): Answer {
  let text = '';
  let input = 0;
  let output = 0;
  let stop = 'stop';
  for (const f of frames) {
    const o = f as Record<string, unknown>;
    const choice = ((o.choices as unknown[]) ?? [])[0] as Record<string, unknown> | undefined;
    const delta = (choice?.delta ?? {}) as Record<string, unknown>;
    if (typeof delta.content === 'string') text += delta.content;
    if (typeof choice?.finish_reason === 'string') stop = choice.finish_reason;
    const usage = o.usage as Record<string, unknown> | undefined;
    if (usage) {
      if (typeof usage.prompt_tokens === 'number') input = usage.prompt_tokens;
      if (typeof usage.completion_tokens === 'number') output = usage.completion_tokens;
    }
  }
  return { text, inputTokens: input, outputTokens: output, stop };
}

function parseAnthropicSse(frames: unknown[]): Answer {
  let text = '';
  let input = 0;
  let output = 0;
  let stop = 'end_turn';
  for (const f of frames) {
    const o = f as Record<string, unknown>;
    if (o.type === 'message_start') {
      const msg = (o.message ?? {}) as Record<string, unknown>;
      const usage = (msg.usage ?? {}) as Record<string, unknown>;
      if (typeof usage.input_tokens === 'number') input = usage.input_tokens;
    } else if (o.type === 'content_block_delta') {
      const delta = (o.delta ?? {}) as Record<string, unknown>;
      if (delta.type === 'text_delta' && typeof delta.text === 'string') text += delta.text;
    } else if (o.type === 'message_delta') {
      const delta = (o.delta ?? {}) as Record<string, unknown>;
      if (typeof delta.stop_reason === 'string') stop = delta.stop_reason;
      const usage = (o.usage ?? {}) as Record<string, unknown>;
      if (typeof usage.output_tokens === 'number') output = usage.output_tokens;
    }
  }
  return { text, inputTokens: input, outputTokens: output, stop };
}

function parseGeminiSse(frames: unknown[]): Answer {
  let text = '';
  let input = 0;
  let output = 0;
  let stop = 'STOP';
  for (const f of frames) {
    const o = f as Record<string, unknown>;
    const cand = ((o.candidates as unknown[]) ?? [])[0] as Record<string, unknown> | undefined;
    if (cand) {
      const content = (cand.content ?? {}) as Record<string, unknown>;
      text += geminiPartsText(content.parts);
      if (typeof cand.finishReason === 'string') stop = cand.finishReason;
    }
    const usage = (o.usageMetadata ?? {}) as Record<string, unknown>;
    if (typeof usage.promptTokenCount === 'number') input = usage.promptTokenCount;
    if (typeof usage.candidatesTokenCount === 'number') output = usage.candidatesTokenCount;
  }
  return { text, inputTokens: input, outputTokens: output, stop };
}

async function parseProviderAnswer(wire: Wire, res: Response): Promise<Answer> {
  const contentType = res.headers.get('content-type') ?? '';
  const raw = await res.text();
  const isSse = contentType.includes('text/event-stream') || raw.includes('data:');
  if (isSse) {
    const frames = sseFrames(raw);
    if (wire === 'openai') return parseOpenAiSse(frames);
    if (wire === 'anthropic') return parseAnthropicSse(frames);
    return parseGeminiSse(frames);
  }
  const json = JSON.parse(raw) as Record<string, unknown>;
  if (wire === 'openai') return parseOpenAiJson(json);
  if (wire === 'anthropic') return parseAnthropicJson(json);
  return parseGeminiJson(json);
}

// ---------------------------------------------------------------- client response rendering

function mapStop(providerStop: string, clientApi: 'openai' | 'anthropic'): string {
  const s = providerStop.toLowerCase();
  if (s === 'stop' || s === 'end_turn') return clientApi === 'openai' ? 'stop' : 'end_turn';
  if (s === 'length' || s === 'max_tokens') return clientApi === 'openai' ? 'length' : 'max_tokens';
  if (s === 'tool_calls' || s === 'tool_use') return clientApi === 'openai' ? 'tool_calls' : 'tool_use';
  if (s === 'content_filter' || s === 'refusal') return clientApi === 'openai' ? 'content_filter' : 'refusal';
  return clientApi === 'openai' ? 'stop' : 'end_turn';
}

function chunkText(text: string, size: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
  return out;
}

function renderOpenAiJson(a: Answer, modelId: string, stop: string): Record<string, unknown> {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: createdSeconds(),
    model: modelId,
    choices: [{ index: 0, message: { role: 'assistant', content: a.text }, finish_reason: stop, logprobs: null }],
    usage: { prompt_tokens: a.inputTokens, completion_tokens: a.outputTokens, total_tokens: a.inputTokens + a.outputTokens },
  };
}

function renderOpenAiSse(a: Answer, modelId: string, stop: string): string {
  const created = createdSeconds();
  const frames: Record<string, unknown>[] = [];
  frames.push({ id: 'chatcmpl-agg', object: 'chat.completion.chunk', created, model: modelId, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] });
  for (const piece of chunkText(a.text, 13)) {
    frames.push({ id: 'chatcmpl-agg', object: 'chat.completion.chunk', created, model: modelId, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] });
  }
  frames.push({ id: 'chatcmpl-agg', object: 'chat.completion.chunk', created, model: modelId, choices: [{ index: 0, delta: {}, finish_reason: stop }] });
  frames.push({
    id: 'chatcmpl-agg', object: 'chat.completion.chunk', created, model: modelId,
    choices: [], usage: { prompt_tokens: a.inputTokens, completion_tokens: a.outputTokens, total_tokens: a.inputTokens + a.outputTokens },
  });
  return frames.map((f) => `data: ${JSON.stringify(f)}\n\n`).join('') + 'data: [DONE]\n\n';
}

function renderAnthropicJson(a: Answer, modelId: string, stop: string): Record<string, unknown> {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model: modelId,
    content: [{ type: 'text', text: a.text }],
    stop_reason: stop,
    stop_sequence: null,
    usage: { input_tokens: a.inputTokens, output_tokens: a.outputTokens },
  };
}

function renderAnthropicSse(a: Answer, modelId: string, stop: string): string {
  // Anthropic streams carry an event name on every frame; the client
  // requires it ("SSE event 1 carries no event name" otherwise).
  // Anthropic streams carry an event name on every frame; the client
  // requires it ("SSE event 1 carries no event name" otherwise).
  const frames: [string, Record<string, unknown>][] = [
    ['message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg', type: 'message', role: 'assistant', model: modelId,
        content: [], stop_reason: null, stop_sequence: null,
        usage: { input_tokens: a.inputTokens, output_tokens: 0 },
      },
    }],
    ['content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }],
  ];
  for (const piece of chunkText(a.text, 13)) {
    frames.push(['content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }]);
  }
  frames.push(['content_block_stop', { type: 'content_block_stop', index: 0 }]);
  frames.push(['message_delta', {
    type: 'message_delta',
    delta: { stop_reason: stop, stop_sequence: null },
    usage: { output_tokens: a.outputTokens },
  }]);
  frames.push(['message_stop', { type: 'message_stop' }]);
  return frames.map(([event, f]) => `event: ${event}\ndata: ${JSON.stringify(f)}\n\n`).join('');
}

// ---------------------------------------------------------------- request handling

function bodyHasImage(messages: unknown): boolean {
  if (!Array.isArray(messages)) return false;
  for (const m of messages) {
    const o = m as Record<string, unknown>;
    if (Array.isArray(o.content)) {
      for (const part of o.content) {
        const p = part as Record<string, unknown>;
        if (p.type === 'image_url' || p.type === 'image') return true;
      }
    }
  }
  return false;
}

function deriveRequirements(body: Record<string, unknown>, manifestRow?: ManifestRow): Requirements {
  const req: Requirements = { ...(manifestRow?.requires ?? {}) };
  if (Array.isArray(body.tools) && body.tools.length > 0) req.tools = true;
  if (bodyHasImage(body.messages)) req.vision = true;
  const rf = body.response_format as Record<string, unknown> | undefined;
  if (rf && (rf.type === 'json_object' || rf.type === 'json_schema')) req.json = true;
  return req;
}

async function readBody(req: IncomingMessage, limit = 32 * 1024 * 1024): Promise<string> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > limit) throw new Error('request body too large');
    chunks.push(chunk as Buffer);
  }
  return Buffer.concat(chunks).toString('utf8');
}

function sendJson(res: ServerResponse, status: number, obj: unknown): void {
  const text = JSON.stringify(obj);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

interface GatewayState {
  ctx: SelectionCtx;
  catalog: CatalogProvider[];
  manifestIndex: number;
  inFlight: number;
}

async function handleInference(
  req: IncomingMessage,
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
  state: GatewayState,
): Promise<void> {
  state.inFlight++;
  try {
    const raw = await readBody(req);
    let body: Record<string, unknown>;
    try {
      body = JSON.parse(raw) as Record<string, unknown>;
    } catch {
      if (clientApi === 'openai') sendJson(res, 400, { error: { message: 'invalid JSON body', type: 'invalid_request_error', code: null } });
      else sendJson(res, 400, { type: 'error', error: { type: 'invalid_request_error', message: 'invalid JSON body' } });
      return;
    }
    const manifestRow = state.ctx.brief.manifest[state.manifestIndex];
    state.manifestIndex++;
    const stream = body.stream === true || manifestRow?.stream === true;
    const requirements = deriveRequirements(body, manifestRow);
    const candidates = candidatesFor(state.ctx, requirements);
    if (candidates.length === 0) {
      const msg = 'no free model meets this request';
      if (clientApi === 'openai') sendJson(res, 503, { error: { message: msg, type: 'api_error', code: null } });
      else sendJson(res, 503, { type: 'error', error: { type: 'overloaded_error', message: msg } });
      return;
    }
    let lastError = '';
    for (const cand of candidates) {
      const provider = cand.provider;
      const { url, body: providerBody } = buildProviderBody(clientApi, body, provider, cand.model.id, stream);
      try {
        const upstream = await fetch(url, {
          method: 'POST',
          headers: { 'content-type': 'application/json', ...authHeaders(provider) },
          body: JSON.stringify(providerBody),
          signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
        });
        provider.calls++;
        if (!upstream.ok) {
          lastError = `HTTP ${upstream.status} from ${provider.id}`;
          provider.consecFailures++;
          continue;
        }
        const answer = await parseProviderAnswer(provider.wire, upstream);
        provider.consecFailures = 0;
        const stop = mapStop(answer.stop, clientApi);
        const modelId = cand.model.id;
        log(`${manifestRow?.id ?? '?'} -> ${provider.id}/${modelId} (${answer.inputTokens}+${answer.outputTokens})`);
        if (clientApi === 'openai') {
          if (stream) sendSse(res, renderOpenAiSse(answer, modelId, stop));
          else sendJson(res, 200, renderOpenAiJson(answer, modelId, stop));
        } else {
          if (stream) sendSse(res, renderAnthropicSse(answer, modelId, stop));
          else sendJson(res, 200, renderAnthropicJson(answer, modelId, stop));
        }
        return;
      } catch (err) {
        lastError = String(err);
        provider.consecFailures++;
        continue;
      }
    }
    const msg = `all providers failed (${lastError})`;
    if (clientApi === 'openai') sendJson(res, 503, { error: { message: msg, type: 'api_error', code: null } });
    else sendJson(res, 503, { type: 'error', error: { type: 'overloaded_error', message: msg } });
  } finally {
    state.inFlight--;
  }
}

function handleModels(req: IncomingMessage, res: ServerResponse, state: GatewayState): void {
  const anthropicShape = req.headers['x-api-key'] !== undefined;
  const ids = freeModelIds(state.catalog);
  if (anthropicShape) {
    sendJson(res, 200, {
      data: ids.map((m) => ({
        type: 'model', id: m.id, display_name: m.id, created_at: createdIso(), free: true,
      })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: ids.map((m) => ({
        id: m.id, object: 'model', created: createdSeconds(), owned_by: m.provider, free: true,
      })),
    });
  }
}

// ---------------------------------------------------------------- main

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const listMode = args.includes('--list');

  const promptFile = process.env.MAKEFUN_PROMPT_FILE;
  let brief: Brief = { providers: [], ratings: [], manifest: [], trialId: '' };
  if (promptFile) {
    brief = parseBrief(readFileSync(promptFile, 'utf8'));
    log(`brief: ${brief.providers.length} providers, ${brief.ratings.length} ratings, ${brief.manifest.length} manifest rows, trial ${brief.trialId}`);
  } else {
    log('MAKEFUN_PROMPT_FILE not set; using an empty brief');
  }

  // Set the env base URL before probing (provider directory entries are
  // paths relative to it).
  ENV_URL = process.env.MAKEFUN_ENV_URL ?? '';

  // Probe every provider's catalog route (once, retry once on failure).
  const probed = new Map<string, ProbedModel[]>();
  await Promise.all(brief.providers.map(async (p) => {
    let models = await probeProvider(p);
    if (models.length === 0) {
      log(`probe ${p.id}: empty, retrying once`);
      models = await probeProvider(p);
    }
    probed.set(p.id, models);
    log(`probe ${p.id}: ${models.length} models`);
  }));
  const catalog = buildCatalog(brief, probed);
  const ctx = buildSelectionCtx(brief, probed);

  if (listMode) {
    process.stdout.write(JSON.stringify({ providers: catalog }) + '\n');
    return;
  }

  const envUrl = process.env.MAKEFUN_ENV_URL;
  if (!envUrl) {
    console.error('MAKEFUN_ENV_URL is not set; cannot register with the trial env');
    process.exit(1);
  }

  const state: GatewayState = { ctx, catalog, manifestIndex: 0, inFlight: 0 };

  const server = createServer((req, res) => {
    (async () => {
      try {
        const url = new URL(req.url ?? '/', 'http://localhost');
        if (req.method === 'GET' && url.pathname === '/v1/models') {
          handleModels(req, res, state);
        } else if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
          await handleInference(req, res, 'openai', state);
        } else if (req.method === 'POST' && url.pathname === '/v1/messages') {
          await handleInference(req, res, 'anthropic', state);
        } else {
          sendJson(res, 404, { error: { message: 'not found', type: 'invalid_request_error', code: null } });
        }
      } catch (err) {
        log('handler error:', String(err));
        if (!res.headersSent) sendJson(res, 500, { error: { message: 'internal error', type: 'api_error', code: null } });
      }
    })();
  });

  await new Promise<void>((resolve) => server.listen(0, '0.0.0.0', resolve));
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('could not bind a port');
  const port = address.port;
  if (port === 8080) throw new Error('port 8080 is reserved for the env');

  // Register with the trial env (retry a few times; the env driver starts
  // on the first successful register).
  let registered = false;
  for (let attempt = 0; attempt < 3 && !registered; attempt++) {
    try {
      const res = await fetch(`${envUrl}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: 1, port, catalog: { providers: catalog } }),
        signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
      });
      if (!res.ok) {
        const detail = await res.text().catch(() => '');
        throw new Error(`register HTTP ${res.status}: ${detail}`);
      }
      registered = true;
    } catch (err) {
      log(`register attempt ${attempt + 1} failed:`, String(err));
      if (attempt < 2) await new Promise((r) => setTimeout(r, 500));
    }
  }
  if (!registered) {
    console.error('could not register with the trial env');
    process.exit(1);
  }
  log(`registered on port ${port}; serving`);

  let exiting = false;
  const finish = () => {
    if (exiting) return;
    exiting = true;
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 1000).unref();
  };
  process.on('SIGTERM', finish);
  process.on('SIGINT', finish);

  // Poll /work until the env says the trial is done and no request is in flight.
  for (;;) {
    await new Promise((r) => setTimeout(r, POLL_MS));
    try {
      const res = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
      if (!res.ok) continue;
      const work = await res.json() as { done?: boolean; pending?: number };
      if (work.done === true && state.inFlight === 0) {
        log('work done; exiting 0');
        finish();
        return;
      }
    } catch {
      // Env not reachable yet; keep polling.
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
