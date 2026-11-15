"""Focused protocol regressions; fixtures here are test inputs, never trial answers."""
import copy
import http.client
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
import wire


class WireTests(unittest.TestCase):
    def answer(self):
        return {"id": "msg_test", "type": "message", "role": "assistant", "model": "test-model",
                "content": [{"type": "text", "text": "Actual upstream text: café 🧭\n"},
                            {"type": "tool_use", "id": "call_a", "name": "lookup", "input": {"account": "0012", "nested": [True, None]}},
                            {"type": "tool_use", "id": "call_b", "name": "ticket", "input": {"subject": "雪"}}],
                "stop_reason": "tool_use", "stop_sequence": None,
                "usage": {"input_tokens": 123, "output_tokens": 17}}

    def test_stream_roundtrip_both_apis(self):
        for api in ("anthropic", "openai"):
            with self.subTest(api=api):
                answer = self.answer()
                raw = "".join(wire.render_stream(answer, api)).encode()
                actual = wire.decode_answer(raw, api, True, "test-model")
                self.assertEqual(actual, answer)

    def test_anthropic_usage_is_known_at_start(self):
        frames = list(wire.sse_frames("".join(wire.render_stream(self.answer(), "anthropic")).encode()))
        start = json.loads(frames[0][1])
        self.assertEqual(start["message"]["usage"], {"input_tokens": 123, "output_tokens": 0})
        self.assertEqual(frames[-1][0], "message_stop")
        self.assertEqual(frames[-2][0], "message_delta")

    def test_openai_fragmented_interleaved_tools_and_late_usage(self):
        base = {"id": "a", "model": "b", "created": 0}
        def event(delta, finish=None):
            return wire.frame(None, {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
        raw = event({"role": "assistant"})
        raw += event({"tool_calls": [{"index": 0, "id": "one", "function": {"name": "look", "arguments": '{"x":'}},
                                     {"index": 1, "id": "two", "function": {"name": "create", "arguments": '{"y":'}}]})
        raw += event({"tool_calls": [{"index": 1, "function": {"arguments": '"雪"}'}},
                                     {"index": 0, "function": {"name": "up", "arguments": '1}'}}]})
        raw += event({}, "tool_calls")
        raw += wire.frame(None, {**base, "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}})
        raw += "data: [DONE]\n\n"
        answer = wire.decode_answer(raw.encode(), "openai", True, "b")
        self.assertEqual([b["name"] for b in answer["content"] if b["type"] == "tool_use"], ["lookup", "create"])
        self.assertEqual(answer["content"][-1]["input"], {"y": "雪"})
        self.assertEqual(answer["usage"], {"input_tokens": 9, "output_tokens": 4})

    def test_sse_comments_crlf_and_multiline_data(self):
        raw = b': heartbeat\r\nevent: x\r\ndata: {"a":\r\ndata: 1}\r\n\r\n'
        self.assertEqual(list(wire.sse_frames(raw)), [("x", '{"a":\n1}')])

    def test_truncated_streams_are_errors(self):
        for api, ending in (("openai", "data: [DONE]\n\n"),
                            ("anthropic", wire.frame("message_stop", {"type": "message_stop"}))):
            raw = "".join(wire.render_stream(self.answer(), api))
            self.assertTrue(raw.endswith(ending))
            with self.assertRaises(ValueError):
                wire.decode_answer(raw[:-len(ending)].encode(), api, True, "test-model")

    def test_openai_rejects_data_after_done(self):
        raw = "".join(wire.render_stream(self.answer(), "openai")) + wire.frame(None, {"choices": []})
        with self.assertRaises(ValueError):
            wire.openai_stream(raw.encode())

    def test_anthropic_rejects_delta_on_closed_block(self):
        raw = "".join(wire.render_stream(self.answer(), "anthropic"))
        needle = wire.frame("content_block_stop", {"type": "content_block_stop", "index": 0})
        raw = raw.replace(needle, needle + wire.frame("content_block_delta", {
            "type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "bad"}}))
        with self.assertRaises(ValueError):
            wire.anthropic_stream(raw.encode())

    def test_stop_families(self):
        for anthropic, openai in (("end_turn", "stop"), ("stop_sequence", "stop"),
                                  ("max_tokens", "length"), ("tool_use", "tool_calls"), ("refusal", "content_filter")):
            answer = self.answer()
            answer["stop_reason"] = anthropic
            if anthropic == "stop_sequence":
                answer["stop_sequence"] = "END"
            out = wire.answer_openai(answer)
            self.assertEqual(out["choices"][0]["finish_reason"], openai)

    def request(self):
        return {"model": "auto", "max_tokens": 45, "temperature": 0.3, "top_p": 0.8, "stop": ["END"],
                "stream": True, "messages": [{"role": "system", "content": "System Ω"},
                    {"role": "user", "content": [{"type": "text", "text": "Look"},
                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj"}}]},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "call_x", "type": "function", "function": {"name": "lookup", "arguments": '{"id":12}'}}]},
                    {"role": "tool", "tool_call_id": "call_x", "content": '{"found":true}'}],
                "tools": [{"type": "function", "function": {"name": "lookup", "description": "Find it",
                    "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}}}],
                "tool_choice": {"type": "function", "function": {"name": "lookup"}}, "parallel_tool_calls": False}

    def test_request_anthropic_roundtrip(self):
        original = self.request()
        converted = wire.to_anthropic(original)
        self.assertEqual(converted["system"], "System Ω")
        self.assertEqual(converted["messages"][0]["content"][1]["source"]["data"], "YWJj")
        restored = wire.to_openai(converted, "anthropic")
        for key in ("max_tokens", "temperature", "top_p", "stop", "tools", "tool_choice", "parallel_tool_calls", "stream"):
            self.assertEqual(restored[key], original[key])
        self.assertEqual(restored["messages"][-1], original["messages"][-1])
        self.assertEqual(restored["messages"][-2]["tool_calls"], original["messages"][-2]["tool_calls"])

    def test_gemini_request_preserves_tools_results_image_system_and_schema(self):
        body = self.request()
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}}
        out = wire.to_gemini(body)
        self.assertEqual(out["systemInstruction"]["parts"], [{"text": "System Ω"}])
        self.assertEqual(out["contents"][0]["parts"][1], {"inlineData": {"mimeType": "image/png", "data": "YWJj"}})
        self.assertEqual(out["contents"][-1]["parts"][0]["functionResponse"], {"id": "call_x", "name": "lookup", "response": {"found": True}})
        self.assertEqual(out["generationConfig"], {"maxOutputTokens": 45, "temperature": 0.3, "topP": 0.8,
            "stopSequences": ["END"], "responseMimeType": "application/json", "responseJsonSchema": {"type": "object"}})
        self.assertEqual(out["toolConfig"]["functionCallingConfig"], {"mode": "ANY", "allowedFunctionNames": ["lookup"]})

    def test_json_not_silently_dropped_by_anthropic(self):
        body = self.request()
        body["response_format"] = {"type": "json_object"}
        with self.assertRaises(ValueError):
            wire.to_anthropic(body)

    def test_gemini_stream_text_usage_and_tool_reason(self):
        raw = wire.frame(None, {"candidates": [{"content": {"parts": [{"text": "hé"}]}}]})
        raw += wire.frame(None, {"candidates": [{"content": {"parts": [{"text": "llo"},
            {"functionCall": {"id": "c", "name": "f", "args": {"v": 1}}}]}, "finishReason": "STOP"}],
            "modelVersion": "model-g", "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3}})
        answer = wire.decode_answer(raw.encode(), "gemini", True, "model-g")
        self.assertEqual(answer["content"][0]["text"], "héllo")
        self.assertEqual(answer["stop_reason"], "tool_use")
        self.assertEqual(answer["usage"], {"input_tokens": 7, "output_tokens": 3})

    def test_bounds_and_invalid_json(self):
        with self.assertRaises(ValueError):
            list(wire.sse_frames(b"x" * (wire.MAX_BYTES + 1)))
        with self.assertRaises(ValueError):
            wire.loads('{"x":NaN}')
        with self.assertRaises(ValueError):
            wire.object_args("not json")


class RouterTests(unittest.TestCase):
    def test_auth_is_independent_of_wire(self):
        for scheme, expected in (("bearer", {"Authorization": "Bearer test-only"}),
                                 ("x-api-key", {"x-api-key": "test-only"}),
                                 ("x-goog-api-key", {"x-goog-api-key": "test-only"}), ("none", {})):
            headers = main.auth_headers({"auth": scheme, "key": "test-only", "wire": "anthropic"}, "custom-version")
            self.assertEqual(headers, {**expected, "anthropic-version": "custom-version"})

    def test_refusals_distinguish_budget_outage_and_temporary(self):
        for api in ("openai", "anthropic", "gemini"):
            for message, expected in (("Provider refused this call", "temporary"),
                                      ("Provider's free-tier budget is spent", "permanent"),
                                      ("Provider is out", "permanent")):
                key = {"openai": "code", "anthropic": "type", "gemini": "status"}[api]
                raw = wire.dumps({"error": {key: "insufficient_quota", "message": message}}).encode()
                self.assertEqual(main.failure_kind(429, raw, api)[0], expected)
        raw = b'{"error":{"code":"payment_required","message":"refused this call"}}'
        self.assertEqual(main.failure_kind(402, raw, "openai")[0], "temporary")

    def test_provider_urls_cannot_escape_env(self):
        transport = main.Transport("http://127.0.0.1:12345")
        self.assertEqual(transport.path("/p/test"), "/p/test")
        self.assertEqual(transport.path("http://127.0.0.1:12345/p/test"), "/p/test")
        for bad in ("https://example.com/p/test", "//example.com/p/test", "/register", "/p/../register"):
            with self.assertRaises(ValueError):
                transport.path(bad)

    def test_explicit_paid_catalog_overrides_rating(self):
        self.assertFalse(main.free_flag({"free": False}, {"free": True}))
        self.assertFalse(main.free_flag({"pricing": {"prompt": "0.001"}}, {"free": True}))
        self.assertTrue(main.free_flag({"pricing": {"prompt": "0", "completion": "0"}}, {}))
        self.assertFalse(main.free_flag({}, {}))

    def test_outside_trial_catalog(self):
        out = subprocess.run([sys.executable, "-P", "-s", "main.py", "--list"],
                             cwd=Path(__file__).resolve().parents[1], env={}, capture_output=True, check=True)
        self.assertEqual(out.stdout, b'{"providers":[]}\n')
        self.assertEqual(out.stderr, b"")

    def test_ranking_does_not_treat_call_count_as_hidden_budget(self):
        providers = [{"id": "a"}, {"id": "b"}]
        router = main.Router(None, ({"providers": providers}, {"ratings": []}, {"manifest": []}))
        candidates = [{"provider": "a", "model": "strong", "rating_milli": 1500},
                      {"provider": "b", "model": "other", "rating_milli": 1490}]
        router.calls["a"] = 100
        router.refusals["a"] = 20
        self.assertEqual(router.rank(candidates, {}, 1)[0]["model"], "strong")

    def test_future_specialist_reservation_and_current_weight(self):
        providers = [{"id": "a", "wire": "openai"}, {"id": "b", "wire": "openai"}]
        router = main.Router(None, ({"providers": providers}, {"ratings": []}, {
            "manifest": [{"requires": {"vision": True}, "weight": 2}]}))
        router.candidates = [{"provider": p["id"], "provider_obj": p, "model": p["id"],
            "rating_milli": rating, "vision": vision} for p, rating, vision in
            ((providers[0], 1500, True), (providers[1], 1485, False))]
        self.assertEqual(router.rank(router.candidates, {}, 1)[0]["provider"], "b")
        self.assertEqual(router.rank(router.candidates, {}, 2)[0]["provider"], "a")
        router.manifest = []
        self.assertEqual(router.rank(router.candidates, {}, 1)[0]["provider"], "a")

    def test_full_error_envelopes(self):
        self.assertEqual(main.error_body("openai", 503, "no_available_model", "unavailable"),
            {"error": {"message": "unavailable", "type": "overloaded_error", "param": None, "code": "no_available_model"}})
        self.assertEqual(main.error_body("anthropic", 400, "bad", "invalid"),
            {"type": "error", "error": {"type": "invalid_request_error", "message": "invalid"}})


class PlannerTests(unittest.TestCase):
    def test_assignment_matches_exhaustive_small_problem(self):
        import itertools
        matrices = [[[3, 1, 5], [2, 4, 1]], [[9, 2, 7, 8], [6, 4, 3, 7], [5, 8, 1, 8]],
                    [[0, 0, 0], [0, 0, 0]]]
        for costs in matrices:
            assigned = main.Router.assignment(costs)
            self.assertEqual(len(set(assigned)), len(costs))
            actual = sum(costs[i][j] for i, j in enumerate(assigned))
            expected = min(sum(costs[i][j] for i, j in enumerate(columns))
                           for columns in itertools.permutations(range(len(costs[0])), len(costs)))
            self.assertEqual(actual, expected)
            self.assertEqual(assigned, main.Router.assignment(costs))

    def router(self):
        providers = [{'id': p, 'wire': 'openai', 'auth': 'none', 'dial_path': '/p/' + p} for p in ('a', 'b')]
        jobs = [{'id': 'r1', 'weight': 1, 'api': 'openai', 'requires': {}},
                {'id': 'r2', 'weight': 2, 'api': 'openai', 'requires': {'vision': True}}]
        router = main.Router(None, ({'providers': providers}, {'ratings': [], 'best': [
            {'id': 'r1', 'rating_milli': 1500}, {'id': 'r2', 'rating_milli': 1500}]}, {'manifest': jobs}))
        router.candidates = [{'provider': p['id'], 'provider_obj': p, 'model': p['id'],
                             'rating_milli': rating, 'vision': vision} for p, rating, vision in
                            ((providers[0], 1500, True), (providers[1], 1490, False))]
        return router

    def test_plan_reserves_unique_vision_and_respects_attempts(self):
        router = self.router()
        router.index = 1
        self.assertEqual(router.planned(router.candidates, {}, 1, set())['provider'], 'b')
        self.assertEqual(router.planned(router.candidates, {}, 1, {'b'})['provider'], 'a')
        router.quality_spent = router.quality_bound * 18 // 1000
        self.assertEqual(router.planned(router.candidates, {}, 1, set())['provider'], 'a')

    def test_temporary_recovery_never_repeats_success_or_permanent_error(self):
        for permanent in (False, True):
            router = self.router()
            router.manifest = []
            router.quality_bound = 0
            class Replies:
                def __init__(self):
                    self.paths = []
                def request(self, path, body, headers):
                    self.paths.append(path)
                    if len(self.paths) == 1 and permanent:
                        return 429, b'{"error":{"code":"rate_limit","message":"Provider is out"}}'
                    if len(self.paths) <= 2:
                        return 429, b'{"error":{"code":"rate_limit","message":"Provider refused this call"}}'
                    value = {'id': 'real-upstream-fixture', 'model': body['model'],
                             'choices': [{'message': {'role': 'assistant', 'content': 'upstream answer'}, 'finish_reason': 'stop'}],
                             'usage': {'prompt_tokens': 3, 'completion_tokens': 4}}
                    return 200, wire.dumps(value).encode()
            transport = Replies()
            router.transport = transport
            answer = router.serve({'messages': [{'role': 'user', 'content': 'question'}]}, 'openai', '2023-06-01')
            self.assertEqual(answer['content'], [{'type': 'text', 'text': 'upstream answer'}])
            self.assertEqual(len(transport.paths), 3)
            self.assertEqual(transport.paths[-1], '/p/b/v1/chat/completions' if permanent else '/p/a/v1/chat/completions')

    def test_first_slot_prior_and_observed_failure_update(self):
        router = self.router()
        self.assertEqual(router.slot_risk('a', 0), 0)
        router.calls['a'] = 2
        before = router.slot_risk('a', 0)
        router.refusals['a'] = 1
        self.assertLess(router.slot_risk('a', 0), before)


class HttpTests(unittest.TestCase):
    def test_invalid_requests_have_wire_errors_and_no_inference(self):
        class NoInference:
            def serve(self, *args):
                raise AssertionError("invalid request reached inference")

        with main.Server(("127.0.0.1", 0), main.Handler) as server:
            server.router = NoInference()
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
            thread.start()
            try:
                for path, body in (("/v1/messages", b"not JSON"), ("/v1/chat/completions", b"[]")):
                    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
                    connection.request("POST", path, body, {"Content-Type": "application/json"})
                    response = connection.getresponse()
                    value = json.loads(response.read())
                    self.assertEqual(response.status, 400)
                    self.assertIn("message", value["error"])
                    self.assertIn("type", value["error"])
                    if path == "/v1/messages":
                        self.assertEqual(value["type"], "error")
                    else:
                        self.assertIn("param", value["error"])
                        self.assertIn("code", value["error"])
                    connection.close()
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
                connection.request("POST", "/v1/messages", headers={"Content-Length": str(main.MAX_REQUEST + 1)})
                response = connection.getresponse()
                self.assertEqual(response.status, 413)
                self.assertEqual(json.loads(response.read())["type"], "error")
                connection.close()
            finally:
                server.shutdown()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
