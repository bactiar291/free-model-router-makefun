#!/usr/bin/env python3
"""Deterministic free-model router. Python 3.12/3.13, standard library only."""
import argparse
import copy
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import signal
import sys
import threading
from urllib.parse import quote, urlsplit

# The harness uses -P -s: explicitly load only our adjacent authored module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wire import (MAX_BYTES, answer_openai, decode_answer, dumps, loads,
                  render_stream, to_anthropic, to_gemini, to_openai)

MAX_PROMPT = 1024 * 1024
MAX_REQUEST = 8 * 1024 * 1024
NETWORK_TIMEOUT = 2.0


class WireError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def error_body(api, status, code, message):
    kind = ("invalid_request_error" if status in (400, 413) else "authentication_error" if status == 401
            else "not_found_error" if status == 404 else "rate_limit_error" if status == 429
            else "overloaded_error" if status == 503 else "api_error")
    if api == "anthropic":
        return {"type": "error", "error": {"type": kind, "message": message}}
    return {"error": {"message": message, "type": kind, "param": None, "code": code}}


def read_brief(path):
    with Path(path).open("rb") as file:
        raw = file.read(MAX_PROMPT + 1)
    if len(raw) > MAX_PROMPT:
        raise ValueError("brief too large")
    text, docs, decoder, at = raw.decode("utf-8"), {}, json_decoder(), 0
    # raw_decode consumes complete documents, including pretty-printed ones;
    # prose and illustrative JSON without a recognized schema are inert.
    while True:
        at = text.find("{", at)
        if at < 0:
            break
        try:
            value, end = decoder.raw_decode(text, at)
        except ValueError:
            at += 1
            continue
        at = end
        if isinstance(value, dict) and isinstance(value.get("schema"), str):
            schema = value["schema"]
            if schema in docs:
                raise ValueError("duplicate brief schema")
            docs[schema] = value
    prefix = "free-model-aggregator-01/"
    return tuple(docs[prefix + kind + "/v1"] for kind in ("providers", "ratings", "workload"))


def json_decoder():
    import json
    return json.JSONDecoder(parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


class Transport:
    """Only dial the env origin. No proxies, redirects, key environment reads or retries."""
    def __init__(self, env_url):
        self.origin = urlsplit(env_url)
        if self.origin.scheme != "http" or not self.origin.hostname or self.origin.username or self.origin.password:
            raise ValueError("env must be an HTTP origin")
        if self.origin.path not in ("", "/") or self.origin.query or self.origin.fragment:
            raise ValueError("env must be an origin without a path")
        self.lock = threading.Lock()

    def path(self, base):
        parsed = urlsplit(base)
        if parsed.scheme or parsed.netloc:
            if (parsed.scheme, parsed.hostname, parsed.port or 80) != (self.origin.scheme, self.origin.hostname, self.origin.port or 80):
                raise ValueError("provider is outside the simulated env")
        if not parsed.path.startswith("/p/") or parsed.query or parsed.fragment or ".." in parsed.path.split("/"):
            raise ValueError("invalid simulated provider path")
        return parsed.path.rstrip("/")

    def request(self, path, body=None, headers=None):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("invalid env request path")
        payload = None if body is None else dumps(body).encode("utf-8")
        outgoing = {"Accept": "application/json", **(headers or {})}
        if payload is not None:
            outgoing["Content-Type"] = "application/json"
        with self.lock:
            conn = http.client.HTTPConnection(self.origin.hostname, self.origin.port or 80, timeout=NETWORK_TIMEOUT)
            try:
                conn.request("GET" if payload is None else "POST", path, payload, outgoing)
                response = conn.getresponse()
                data = response.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    raise ValueError("upstream response too large")
                return response.status, data
            finally:
                conn.close()

    def json(self, path, body=None, headers=None):
        status, raw = self.request(path, body, headers)
        if status != 200:
            raise WireError(status, "env_error", "environment request failed")
        return loads(raw)


def auth_headers(provider, version="2023-06-01"):
    auth = provider["auth"]
    headers = {}
    if auth == "bearer":
        headers["Authorization"] = "Bearer " + provider["key"]
    elif auth in ("x-api-key", "x-goog-api-key"):
        headers[auth] = provider["key"]
    elif auth != "none":
        raise ValueError("unknown provider authentication scheme")
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = version
    return headers


def free_flag(model, rating):
    if isinstance(model.get("free"), bool):
        return model["free"]
    pricing = model.get("pricing")
    if isinstance(pricing, dict) and pricing:
        try:
            return all(float(value) == 0 for value in pricing.values())
        except (ValueError, TypeError):
            return False
    return rating.get("free") is True


def failure_kind(status, raw, wire):
    """Status alone is ambiguous: even payment/quota codes can be transient here."""
    try:
        error = loads(raw)["error"]
        code = error.get("status") if wire == "gemini" else error.get("code") if wire == "openai" else error.get("type")
        message = str(error.get("message", "")).lower()
    except (ValueError, TypeError, KeyError, AttributeError):
        code, message = "unknown", ""
    if "free-tier budget is spent" in message or code in ("budget_exhausted", "quota_exhausted"):
        return "permanent", code
    if message.endswith(" is out") or code in ("provider_outage", "provider_disabled"):
        return "permanent", code
    if status in (401, 403):
        return "permanent", code
    if code == "concurrent_requests":
        return "temporary", code
    if status in (402, 408, 429) or status >= 500:
        return "temporary", code
    return "request", code


def meets(row, requirements):
    return (all(not requirements.get(cap) or row.get(cap) is True for cap in ("tools", "vision", "json"))
            and (row.get("context_length") or 0) >= requirements.get("min_context", 0))


class Router:
    def __init__(self, transport, brief):
        self.transport = transport
        directory, quality, workload = brief
        self.providers = directory["providers"]
        self.ratings = {(r["provider"], r["model"]): r for r in quality["ratings"]}
        self.manifest = workload["manifest"]
        self.catalog = {"providers": []}
        self.candidates = []
        self.index = 0
        self.dead = set()
        self.calls = {p["id"]: 0 for p in self.providers}
        self.refusals = {p["id"]: 0 for p in self.providers}
        self.serve_lock = threading.Lock()
        self.quality_spent = 0
        self.best = {r['id']: r['rating_milli'] for r in quality.get('best', [])}
        self.quality_bound = sum(r.get('weight', 1) * self.best.get(r.get('id'), 0) for r in self.manifest)

    def load(self):
        for provider in self.providers:
            base = provider.get("path") or provider.get("base_url")
            path = self.transport.path(base)
            provider["dial_path"] = path
            suffix = "/v1beta/models" if provider["wire"] == "gemini" else "/v1/models"
            listing = self.transport.json(path + suffix, headers=auth_headers(provider))
            source = listing["models" if provider["wire"] == "gemini" else "data"]
            models, seen = [], set()
            for model in source:
                model_id = model["name"].removeprefix("models/") if provider["wire"] == "gemini" else model["id"]
                if model_id in seen:
                    raise ValueError("duplicate catalog model")
                seen.add(model_id)
                rating = self.ratings.get((provider["id"], model_id), {})
                free = free_flag(model, rating)
                context = model.get("context_length", model.get("inputTokenLimit", rating.get("context_length")))
                models.append({"id": model_id, "free": free, "context_length": context})
                if free and rating:
                    self.candidates.append({**rating, "free": True, "provider_obj": provider})
            # Catalog identities use paths; ephemeral env ports must never enter exports.
            self.catalog["providers"].append({"id": provider["id"], "base_url": path,
                "key_env": provider["key_env"], "wire": provider["wire"], "models": sorted(models, key=lambda m: m["id"])})

    def model_list(self, api):
        # Model IDs shared by providers appear once in the public union.
        models = {}
        for provider in self.catalog["providers"]:
            for model in provider["models"]:
                if model["free"]:
                    models.setdefault(model["id"], provider["id"])
        ids = sorted(models)
        if api == "anthropic":
            return {"data": [{"type": "model", "id": mid, "display_name": mid, "created_at": "1970-01-01T00:00:00Z"} for mid in ids],
                    "has_more": False, "first_id": ids[0] if ids else None, "last_id": ids[-1] if ids else None}
        return {"object": "list", "data": [{"id": mid, "object": "model", "created": 0, "owned_by": models[mid]} for mid in ids]}

    def eligible(self, requirements, json_mode=False):
        return [r for r in self.candidates if r["provider"] not in self.dead and meets(r, requirements)
                and not (json_mode and r["provider_obj"]["wire"] == "anthropic")]

    def rank(self, candidates, requirements, weight):
        """Rank weighted quality with a conservative future opportunity cost.

        For each future job, measure the quality lost if a provider disappears.
        Weight that advantage by the job's public weight. The largest advantage
        reserves specialists (vision and long context) and premium models for
        heavy work. Recompute after every observed permanent refusal.

        Call counts alone do not establish exhaustion: penalizing successful
        providers spreads traffic into more failure schedules and can lower
        quality. Keep temporary refusals eligible for later requests; retire
        providers only on explicit permanent signals in serve(). Integer
        arithmetic and identity tie breaks make ordering reproducible.
        """
        reserve = {p["id"]: 0 for p in self.providers}
        for future in self.manifest[self.index:]:
            options = self.eligible(future["requires"])
            by_provider = {}
            for row in options:
                pid = row["provider"]
                by_provider[pid] = max(by_provider.get(pid, 0), row["rating_milli"])
            for pid, rating in by_provider.items():
                alternative = max((v for p, v in by_provider.items() if p != pid), default=0)
                advantage = max(0, rating - alternative) * future.get("weight", 1)
                reserve[pid] = max(reserve[pid], advantage)

        def utility(row):
            pid = row["provider"]
            opportunity = min(160, reserve[pid])
            return row["rating_milli"] * weight * 8 - opportunity

        return sorted(candidates, key=lambda r: (-utility(r), -r["rating_milli"], r["provider"], r["model"]))

    # Public distribution bounds from program/generator/draw.mjs, not instance
    # budgets. No seed, failure positions, or outage identity is known here.
    BUDGET_PRIORS = {'groq': (9, 12), 'cerebras': (5, 6), 'gemini': (6, 8),
                     'zai': (4, 5), 'openrouter': (10, 12), 'sambanova': (4, 5),
                     'pollinations': (10, 14), 'cloudflare': (7, 8),
                     'ovh': (5, 6), 'kilol': (12, 16)}

    def slot_risk(self, pid, offset):
        """Conservative expected refusal cost for an additional provider call.

        The first inference has no scheduled failure. Later slots combine the
        public failure-position and outage distributions; cumulative call count
        only estimates risk, never certifies remaining budget. Observed refusals
        reduce the estimated remaining scheduled failures; explicit permanent
        errors remain hard exclusions. This is a prior, not instance knowledge.
        """
        number = self.calls[pid] + offset + 1
        if number == 1:
            return 0
        lo, hi = self.BUDGET_PRIORS.get(pid, (4, 8))
        risk = 0.0
        for budget in range(lo, hi + 1):
            if number > budget:
                chance = 1.0
            else:
                remaining_failures = max(0.25, 1.5 - self.refusals[pid])
                scheduled = remaining_failures / max(1, budget - 2) if number < budget else 0.0
                designated = 0.1 if number == 2 else 0.0
                outage = 0.1 * min(1.0, (number - 1) / max(1, budget - 2))
                chance = 1 - (1 - scheduled) * (1 - designated) * (1 - outage)
            risk += chance / (hi - lo + 1)
        # Convex usage cost prevents a later cheap slot bypassing earlier risk.
        return round(1000 * risk) + offset * 15

    @staticmethod
    def assignment(costs):
        """Rectangular Hungarian assignment with deterministic column ties."""
        n, m = len(costs), len(costs[0])
        u, v, p, way = [0] * (n + 1), [0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
        for i in range(1, n + 1):
            p[0], j0 = i, 0
            minimum, used = [10**18] * (m + 1), [False] * (m + 1)
            while True:
                used[j0] = True
                i0, delta, j1 = p[j0], 10**18, 0
                for j in range(1, m + 1):
                    if not used[j]:
                        cur = costs[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minimum[j]:
                            minimum[j], way[j] = cur, j0
                        if minimum[j] < delta:
                            delta, j1 = minimum[j], j
                for j in range(m + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minimum[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break
        result = [0] * n
        for j in range(1, m + 1):
            if p[j]:
                result[p[j] - 1] = j - 1
        return result

    def planned(self, candidates, requirements, weight, attempted):
        """Receding-horizon risk allocation subject to a real quality floor.

        A Lagrange multiplier finds a low-risk remaining assignment within the
        weighted quality-loss allowance. Only the first assignment is executed;
        future assignments are recomputed from actual responses. The planner
        never changes a request or synthesizes an answer to meet its target.
        """
        if not self.quality_bound or not candidates:
            return None
        current = self.manifest[self.index - 1]
        jobs = [dict(current, requires=requirements, weight=weight)] + self.manifest[self.index:]
        options = []
        for i, job in enumerate(jobs):
            eligible = candidates if i == 0 else self.eligible(job['requires'], job['api'] == 'openai' and job['requires'].get('json', False))
            by_provider = {}
            for row in eligible:
                pid = row['provider']
                if pid in self.dead or (i == 0 and pid in attempted):
                    continue
                previous = by_provider.get(pid)
                if previous is None or (-row['rating_milli'], row['model']) < (-previous['rating_milli'], previous['model']):
                    by_provider[pid] = row
            if not by_provider:
                return None
            options.append(by_provider)
        providers = sorted({pid for option in options for pid in option})
        slots = [(pid, k) for pid in providers for k in range(len(jobs))]
        risks = []
        for pid in providers:
            highest = 0
            for k in range(len(jobs)):
                highest = max(highest, self.slot_risk(pid, k))
                risks.append(highest)
        loss = [[(self.best[job['id']] - option[pid]['rating_milli']) * job['weight']
                 if pid in option else 10**8 for pid, _ in slots] for job, option in zip(jobs, options)]
        # Reserve two tenths of a percentage point for late capability loss.
        allowance = self.quality_bound * 18 // 1000 - self.quality_spent

        def solve(multiplier):
            costs = [[(risks[j] * 100 + value * multiplier) if value < 10**8 else 10**14
                      for j, value in enumerate(row)] for row in loss]
            chosen = self.assignment(costs)
            return chosen, sum(loss[i][j] for i, j in enumerate(chosen))

        chosen, spent = solve(0)
        if spent > allowance:
            lo, hi = 0, 1000000
            chosen, spent = solve(hi)
            if spent <= allowance:
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    proposal, cost = solve(mid)
                    if cost <= allowance:
                        hi, chosen = mid, proposal
                    else:
                        lo = mid
        return options[0].get(slots[chosen[0]][0])

    def serve(self, body, api, version):
        with self.serve_lock:
            manifest = self.manifest[self.index] if self.index < len(self.manifest) else {"requires": {}, "weight": 1}
            self.index += 1
            canonical = to_openai(body, api)
            requirements = dict(manifest["requires"])
            if canonical.get("tools"):
                requirements["tools"] = True
            if any(part.get("type") == "image_url" for msg in canonical["messages"] for part in (msg.get("content") or []) if isinstance(part, dict)):
                requirements["vision"] = True
            json_mode = canonical.get("response_format", {}).get("type") in ("json_object", "json_schema")
            if json_mode:
                requirements["json"] = True
            candidates = self.rank(self.eligible(requirements, json_mode), requirements, manifest.get("weight", 1))
            attempted = set()
            retryable = set()
            recovery = False
            while True:
                available = [r for r in candidates if r['provider'] not in attempted and r['provider'] not in self.dead]
                if not available:
                    if not recovery and retryable - self.dead:
                        recovery = True
                        attempted.difference_update(retryable)
                        continue
                    break
                row = self.planned(available, requirements, manifest.get('weight', 1), attempted) or available[0]
                provider = row["provider_obj"]
                pid, wire = row["provider"], provider["wire"]
                if pid in attempted or pid in self.dead:
                    continue
                attempted.add(pid)
                upstream = copy.deepcopy(canonical)
                upstream["model"] = row["model"]
                stream = body.get("stream") is True
                upstream["stream"] = stream
                if wire == "openai":
                    route = "/v1/chat/completions"
                    if stream:
                        upstream["stream_options"] = {"include_usage": True}
                elif wire == "anthropic":
                    route = "/v1/messages"
                    upstream = copy.deepcopy(body) if api == "anthropic" else to_anthropic(upstream)
                    upstream["model"] = row["model"]
                else:
                    route = "/v1beta/models/" + quote(row["model"], safe="") + (":streamGenerateContent?alt=sse" if stream else ":generateContent")
                    upstream = to_gemini(upstream)
                headers = auth_headers(provider, version)
                if stream:
                    headers["Accept"] = "text/event-stream"
                self.calls[pid] += 1
                try:
                    status, raw = self.transport.request(provider["dial_path"] + route, upstream, headers)
                except (OSError, http.client.HTTPException):
                    self.refusals[pid] += 1
                    continue
                if status != 200:
                    kind, _ = failure_kind(status, raw, wire)
                    self.refusals[pid] += 1
                    if kind == "permanent":
                        self.dead.add(pid)
                    elif kind == "temporary":
                        retryable.add(pid)
                    continue
                # A successful inference is never replaced by a second one. If its
                # wire is corrupt, report the failure instead of inventing content.
                try:
                    answer = decode_answer(raw, wire, stream, row["model"])
                    self.quality_spent += max(0, self.best.get(manifest.get('id'), row['rating_milli']) - row['rating_milli']) * manifest.get('weight', 1)
                    return answer
                except (ValueError, KeyError, TypeError, IndexError):
                    raise WireError(502, "invalid_upstream_response", "provider returned an invalid or incomplete response") from None
            raise WireError(503, "no_available_model", "no available free provider can serve this request")


class Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    request_queue_size = 16


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *args):
        pass

    def respond(self, status, body, content_type="application/json"):
        raw = body if isinstance(body, bytes) else dumps(body).encode("utf-8")
        self.send_response_only(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True

    def api(self):
        return "anthropic" if urlsplit(self.path).path == "/v1/messages" or "x-api-key" in self.headers else "openai"

    def do_GET(self):
        if urlsplit(self.path).path == "/v1/models":
            self.respond(200, self.server.router.model_list(self.api()))
        else:
            self.respond(404, error_body(self.api(), 404, "not_found", "unknown route"))

    def do_POST(self):
        api = self.api()
        try:
            path = urlsplit(self.path).path
            if path not in ("/v1/messages", "/v1/chat/completions"):
                raise WireError(404, "not_found", "unknown route")
            api = "anthropic" if path == "/v1/messages" else "openai"
            if self.headers.get("Transfer-Encoding"):
                raise WireError(400, "invalid_request", "use a Content-Length JSON body")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not lengths[0].isdigit():
                raise WireError(400, "invalid_request", "one Content-Length is required")
            size = int(lengths[0])
            if not 0 < size <= MAX_REQUEST:
                raise WireError(413, "request_too_large", "request exceeds the body limit")
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise WireError(400, "invalid_request", "incomplete request body")
            body = loads(raw)
            if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
                raise WireError(400, "invalid_request", "messages must be a nonempty array")
            if body.get("n", 1) != 1:
                raise WireError(400, "unsupported_parameter", "only one completion is supported")
            answer = self.server.router.serve(body, api, self.headers.get("anthropic-version", "2023-06-01"))
            if body.get("stream") is True:
                raw = "".join(render_stream(answer, api)).encode("utf-8")
                self.respond(200, raw, "text/event-stream; charset=utf-8")
            else:
                self.respond(200, answer if api == "anthropic" else answer_openai(answer))
        except WireError as error:
            self.respond(error.status, error_body(api, error.status, error.code, error.message))
        except (ValueError, KeyError, TypeError, IndexError, RecursionError):
            self.respond(400, error_body(api, 400, "invalid_request", "invalid or unsupported request body"))
        except (OSError, http.client.HTTPException):
            try:
                self.respond(502, error_body(api, 502, "upstream_unreachable", "upstream connection failed"))
            except OSError:
                pass

    def do_PUT(self):
        self.respond(405, error_body(self.api(), 405, "method_not_allowed", "method not allowed"))

    do_DELETE = do_PATCH = do_PUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="probe and print the catalog; empty outside a trial")
    args = parser.parse_args()
    env_url, prompt = os.environ.get("MAKEFUN_ENV_URL", ""), os.environ.get("MAKEFUN_PROMPT_FILE", "")
    if not env_url or not prompt:
        if args.list:
            print('{"providers":[]}')
            return 0
        print("router: MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE are required", file=sys.stderr)
        return 1
    transport = Transport(env_url)
    router = Router(transport, read_brief(prompt))
    router.load()
    if args.list:
        print(dumps(router.catalog))
        return 0
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    with Server(("0.0.0.0", 0), Handler) as server:
        server.router = router
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        try:
            reply = transport.json("/register", {"protocol_version": 1, "port": server.server_port, "catalog": router.catalog})
            if reply.get("ok") is not True:
                raise ValueError("registration refused")
            failures = 0
            while not stop.wait(0.05):
                try:
                    work = transport.json("/work")
                    failures = 0
                    if work.get("done") is True:
                        return 0
                except (OSError, ValueError, WireError, http.client.HTTPException):
                    failures += 1
                    if failures >= 5:
                        raise ValueError("environment stopped responding") from None
            return 1
        finally:
            server.shutdown()
            thread.join(timeout=2)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, WireError, http.client.HTTPException):
        # Never log brief keys, request contents, upstream envelopes or URLs.
        print("router: trial setup or control failed", file=sys.stderr)
        sys.exit(1)
