"""Lossless translations for the competition's OpenAI, Anthropic and Gemini wires.

Streams are bounded and assembled before translation. This deliberately trades
first-token latency for accurate usage at Anthropic message_start and atomic errors.
No answer text or usage is inferred from a request.
"""
import copy
import json
import re

MAX_BYTES = 2 * 1024 * 1024


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def loads(value):
    return json.loads(value, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def text_of(content):
    if isinstance(content, str):
        return content
    return "\n".join(b["text"] for b in (content or []) if b.get("type") == "text")


def blocks(content):
    if content is None:
        return []
    return [{"type": "text", "text": content}] if isinstance(content, str) else content


def object_args(value):
    result = loads(value) if isinstance(value, str) else value
    if not isinstance(result, dict):
        raise ValueError("tool arguments must be an object")
    return result


def image_to_openai(source):
    if source["type"] == "base64":
        return {"type": "image_url", "image_url": {"url": "data:" + source["media_type"] + ";base64," + source["data"]}}
    if source["type"] == "url":
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    raise ValueError("unsupported image source")


def image_from_openai(part):
    url = part["image_url"]["url"]
    match = re.fullmatch(r"data:([^;,]+);base64,(.*)", url, re.S)
    source = ({"type": "base64", "media_type": match[1], "data": match[2]} if match
              else {"type": "url", "url": url})
    return {"type": "image", "source": source}


def to_openai(body, api):
    """Use Chat Completions as the request interchange; preserve same-wire fields."""
    if api == "openai":
        return copy.deepcopy(body)
    out = {k: copy.deepcopy(body[k]) for k in ("model", "max_tokens", "temperature", "top_p", "stream") if k in body}
    messages = []
    if "system" in body:
        messages.append({"role": "system", "content": text_of(body["system"])})
    for message in body["messages"]:
        content = message["content"]
        if isinstance(content, str):
            messages.append(copy.deepcopy(message))
            continue
        parts, calls = [], []
        for block in content:
            kind = block["type"]
            if kind == "text":
                parts.append(copy.deepcopy(block))
            elif kind == "image":
                parts.append(image_to_openai(block["source"]))
            elif kind == "tool_use":
                calls.append({"id": block["id"], "type": "function", "function": {
                    "name": block["name"], "arguments": dumps(block["input"])}})
            elif kind == "tool_result":
                result = block.get("content", "")
                if isinstance(result, list):
                    result = [image_to_openai(b["source"]) if b["type"] == "image" else copy.deepcopy(b) for b in result]
                messages.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": result})
            else:
                raise ValueError("unsupported Anthropic content block")
        if parts or calls:
            msg = {"role": message["role"], "content": parts or None}
            if calls:
                msg["tool_calls"] = calls
            messages.append(msg)
    out["messages"] = messages
    if "stop_sequences" in body:
        out["stop"] = copy.deepcopy(body["stop_sequences"])
    if "tools" in body:
        out["tools"] = [{"type": "function", "function": {
            "name": t["name"], **({"description": t["description"]} if "description" in t else {}),
            "parameters": copy.deepcopy(t["input_schema"])}} for t in body["tools"]]
    if "tool_choice" in body:
        choice = body["tool_choice"]
        out["tool_choice"] = ({"type": "function", "function": {"name": choice["name"]}}
                              if choice["type"] == "tool" else {"any": "required", "auto": "auto", "none": "none"}[choice["type"]])
        if "disable_parallel_tool_use" in choice:
            out["parallel_tool_calls"] = not choice["disable_parallel_tool_use"]
    # Some compatible Messages clients carry this extension. Never invent it.
    if "response_format" in body:
        out["response_format"] = copy.deepcopy(body["response_format"])
    return out


def to_anthropic(body):
    if body.get("response_format", {}).get("type") in ("json_object", "json_schema"):
        raise ValueError("this Anthropic wire cannot carry JSON mode")
    out = {k: copy.deepcopy(body[k]) for k in ("model", "max_tokens", "temperature", "top_p", "stream") if k in body}
    if "max_completion_tokens" in body:
        out["max_tokens"] = body["max_completion_tokens"]
    system, messages = [], []
    for msg in body["messages"]:
        role = msg["role"]
        if role in ("system", "developer"):
            system.append(text_of(msg["content"]))
            continue
        if role == "tool":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = [image_from_openai(p) if p["type"] == "image_url" else copy.deepcopy(p) for p in content]
            parts = [{"type": "tool_result", "tool_use_id": msg["tool_call_id"], "content": content}]
            role = "user"
        else:
            parts = []
            for part in blocks(msg.get("content")):
                if part["type"] == "image_url":
                    parts.append(image_from_openai(part))
                elif part["type"] == "text":
                    parts.append(copy.deepcopy(part))
                else:
                    raise ValueError("unsupported OpenAI content part")
            for call in msg.get("tool_calls", []):
                parts.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"],
                              "input": object_args(call["function"]["arguments"])})
        # Anthropic groups adjacent messages with the same role.
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(parts)
        else:
            messages.append({"role": role, "content": parts})
    out["messages"] = messages
    if system:
        out["system"] = "\n".join(system)
    if "stop" in body:
        out["stop_sequences"] = [body["stop"]] if isinstance(body["stop"], str) else copy.deepcopy(body["stop"])
    if "tools" in body:
        out["tools"] = [{"name": t["function"]["name"],
                         **({"description": t["function"]["description"]} if "description" in t["function"] else {}),
                         "input_schema": copy.deepcopy(t["function"]["parameters"])} for t in body["tools"]]
    if "tool_choice" in body:
        choice = body["tool_choice"]
        out["tool_choice"] = ({"type": "tool", "name": choice["function"]["name"]} if isinstance(choice, dict)
                              else {"type": {"required": "any", "auto": "auto", "none": "none"}[choice]})
        if "parallel_tool_calls" in body:
            out["tool_choice"]["disable_parallel_tool_use"] = not body["parallel_tool_calls"]
    return out


def to_gemini(body):
    out, system, contents, tool_names = {}, [], [], {}
    for message in body["messages"]:
        role = message["role"]
        if role in ("system", "developer"):
            system.extend({"text": b["text"]} for b in blocks(message.get("content")))
            continue
        parts = []
        if role == "tool":
            name = tool_names.get(message["tool_call_id"])
            if name is None:
                raise ValueError("tool result has no preceding function call")
            result = message.get("content", "")
            if isinstance(result, str):
                try:
                    result = loads(result)
                except ValueError:
                    result = {"result": result}
            if not isinstance(result, dict):
                result = {"result": result}
            parts.append({"functionResponse": {"id": message["tool_call_id"], "name": name, "response": result}})
        else:
            for part in blocks(message.get("content")):
                if part["type"] == "text":
                    parts.append({"text": part["text"]})
                elif part["type"] == "image_url":
                    source = image_from_openai(part)["source"]
                    if source["type"] == "base64":
                        parts.append({"inlineData": {"mimeType": source["media_type"], "data": source["data"]}})
                    else:
                        parts.append({"fileData": {"fileUri": source["url"]}})
                else:
                    raise ValueError("unsupported Gemini content part")
            for call in message.get("tool_calls", []):
                tool_names[call["id"]] = call["function"]["name"]
                parts.append({"functionCall": {"id": call["id"], "name": call["function"]["name"],
                                               "args": object_args(call["function"]["arguments"])}})
        contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    out["contents"] = contents
    if system:
        out["systemInstruction"] = {"parts": system}
    config = {}
    for source, target in (("max_tokens", "maxOutputTokens"), ("max_completion_tokens", "maxOutputTokens"),
                           ("temperature", "temperature"), ("top_p", "topP")):
        if source in body:
            config[target] = body[source]
    if "stop" in body:
        config["stopSequences"] = [body["stop"]] if isinstance(body["stop"], str) else copy.deepcopy(body["stop"])
    fmt = body.get("response_format", {})
    if fmt.get("type") in ("json_object", "json_schema"):
        config["responseMimeType"] = "application/json"
        if fmt["type"] == "json_schema":
            config["responseJsonSchema"] = copy.deepcopy(fmt["json_schema"]["schema"])
    out["generationConfig"] = config
    if body.get("tools"):
        out["tools"] = [{"functionDeclarations": [copy.deepcopy(t["function"]) for t in body["tools"]]}]
    if "tool_choice" in body:
        choice = body["tool_choice"]
        config = ({"mode": "ANY", "allowedFunctionNames": [choice["function"]["name"]]} if isinstance(choice, dict)
                  else {"mode": {"auto": "AUTO", "none": "NONE", "required": "ANY"}[choice]})
        out["toolConfig"] = {"functionCallingConfig": config}
    return out


STOP_TO_ANTHROPIC = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
                     "function_call": "tool_use", "content_filter": "refusal"}
STOP_TO_OPENAI = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls", "refusal": "content_filter"}
GEMINI_STOP = {"STOP": "end_turn", "MAX_TOKENS": "max_tokens", "SAFETY": "refusal", "RECITATION": "refusal",
               "BLOCKLIST": "refusal", "PROHIBITED_CONTENT": "refusal", "SPII": "refusal"}


def sse_frames(raw):
    if len(raw) > MAX_BYTES:
        raise ValueError("upstream stream too large")
    text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    for frame in text.split("\n\n"):
        event, data = None, []
        for line in frame.split("\n"):
            if line.startswith(":"):
                continue
            field, sep, value = line.partition(":")
            if sep and value.startswith(" "):
                value = value[1:]
            if field == "event":
                event = value
            elif field == "data":
                data.append(value)
        if data:
            yield event, "\n".join(data)


def openai_stream(raw):
    answer = {"choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": None}]}
    message = answer["choices"][0]["message"]
    calls, finished, done = {}, False, False
    for _, data in sse_frames(raw):
        if done:
            raise ValueError("event after OpenAI DONE")
        if data == "[DONE]":
            done = True
            continue
        chunk = loads(data)
        if "error" in chunk:
            raise ValueError("upstream stream error")
        for key in ("id", "model", "created", "usage"):
            if key in chunk:
                answer[key] = chunk[key]
        choices = chunk.get("choices", [])
        if not choices:
            continue
        choice = choices[0]
        if choice.get("index", 0) != 0:
            raise ValueError("multiple choices are unsupported")
        delta = choice.get("delta", {})
        if finished and delta:
            raise ValueError("delta after finish")
        message["content"] += delta.get("content") or ""
        if delta.get("refusal"):
            message["refusal"] = message.get("refusal", "") + delta["refusal"]
        for call in delta.get("tool_calls", []):
            idx = call["index"]
            dest = calls.setdefault(idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            dest["id"] += call.get("id", "")
            for key in ("name", "arguments"):
                dest["function"][key] += call.get("function", {}).get(key, "")
        if choice.get("finish_reason") is not None:
            if finished:
                raise ValueError("duplicate finish")
            answer["choices"][0]["finish_reason"] = choice["finish_reason"]
            finished = True
    if not done or not finished:
        raise ValueError("truncated OpenAI stream")
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    return answer


def anthropic_stream(raw):
    answer, open_blocks, closed, partial = None, {}, {}, {}
    stopped = delta_seen = False
    for event, data in sse_frames(raw):
        if stopped:
            raise ValueError("event after message_stop")
        obj = loads(data)
        if obj.get("type") != event:
            raise ValueError("Anthropic event/type mismatch")
        if event == "ping":
            continue
        if event == "message_start":
            if answer is not None:
                raise ValueError("duplicate message_start")
            answer = copy.deepcopy(obj["message"])
            answer["content"] = []
        elif answer is None:
            raise ValueError("missing message_start")
        elif event == "content_block_start":
            i = obj["index"]
            if i in open_blocks or i in closed or delta_seen:
                raise ValueError("invalid block start")
            open_blocks[i] = copy.deepcopy(obj["content_block"])
        elif event == "content_block_delta":
            i, delta = obj["index"], obj["delta"]
            if i not in open_blocks:
                raise ValueError("delta outside block")
            if delta["type"] == "text_delta":
                open_blocks[i]["text"] += delta["text"]
            elif delta["type"] == "input_json_delta":
                partial[i] = partial.get(i, "") + delta["partial_json"]
            else:
                raise ValueError("unsupported content delta")
        elif event == "content_block_stop":
            i = obj["index"]
            block = open_blocks.pop(i)
            if i in partial:
                block["input"] = object_args(partial[i])
            closed[i] = block
        elif event == "message_delta":
            if open_blocks or delta_seen:
                raise ValueError("invalid message_delta order")
            delta_seen = True
            answer.update(obj["delta"])
            answer.setdefault("usage", {}).update(obj.get("usage", {}))
        elif event == "message_stop":
            if not delta_seen or open_blocks:
                raise ValueError("premature message_stop")
            stopped = True
        else:
            raise ValueError("upstream error or unsupported event")
    if not stopped or answer is None:
        raise ValueError("truncated Anthropic stream")
    answer["content"] = [closed[i] for i in sorted(closed)]
    return answer


def gemini_stream(raw):
    parts, answer, finished = [], {}, False
    for _, data in sse_frames(raw):
        chunk = loads(data)
        if "error" in chunk:
            raise ValueError("upstream Gemini stream error")
        for key in ("usageMetadata", "modelVersion", "responseId"):
            if key in chunk:
                answer[key] = chunk[key]
        candidates = chunk.get("candidates", [])
        if not candidates:
            continue
        candidate = candidates[0]
        incoming = candidate.get("content", {}).get("parts", [])
        if finished and incoming:
            raise ValueError("Gemini content after finish")
        for part in incoming:
            if "text" in part and parts and "text" in parts[-1]:
                parts[-1]["text"] += part["text"]
            else:
                parts.append(copy.deepcopy(part))
        if "finishReason" in candidate:
            if finished:
                raise ValueError("duplicate Gemini finish")
            finished = True
            answer["candidates"] = [{"finishReason": candidate["finishReason"], "content": {"parts": parts, "role": "model"}}]
    if not finished:
        raise ValueError("truncated Gemini stream")
    return answer


def normalize_answer(value, wire, model):
    """Anthropic-shaped response interchange, derived only from upstream fields."""
    if wire == "anthropic":
        answer = copy.deepcopy(value)
    elif wire == "openai":
        choice = value["choices"][0]
        message, content = choice["message"], []
        if message.get("content") is not None:
            content.extend(copy.deepcopy(blocks(message["content"])))
        if message.get("refusal"):
            content.append({"type": "text", "text": message["refusal"]})
        for call in message.get("tool_calls", []):
            content.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"],
                            "input": object_args(call["function"]["arguments"])})
        usage = value["usage"]
        answer = {"id": value["id"], "type": "message", "role": "assistant", "model": value["model"],
                  "content": content, "stop_reason": STOP_TO_ANTHROPIC[choice["finish_reason"]], "stop_sequence": None,
                  "usage": {"input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"]}}
    else:
        candidate, content = value["candidates"][0], []
        for i, part in enumerate(candidate.get("content", {}).get("parts", [])):
            if "text" in part:
                content.append({"type": "text", "text": part["text"]})
            elif "functionCall" in part:
                fn = part["functionCall"]
                content.append({"type": "tool_use", "id": fn.get("id", "call_" + str(i)), "name": fn["name"], "input": fn["args"]})
            else:
                raise ValueError("unsupported Gemini answer part")
        reason = GEMINI_STOP[candidate["finishReason"]]
        if reason == "end_turn" and any(b["type"] == "tool_use" for b in content):
            reason = "tool_use"
        usage = value["usageMetadata"]
        answer = {"id": value.get("responseId", "msg_gemini"), "type": "message", "role": "assistant",
                  "model": value.get("modelVersion", model), "content": content, "stop_reason": reason, "stop_sequence": None,
                  "usage": {"input_tokens": usage["promptTokenCount"], "output_tokens": usage["candidatesTokenCount"]}}
    if answer.get("role") != "assistant" or answer.get("stop_reason") not in STOP_TO_OPENAI:
        raise ValueError("invalid upstream answer")
    for key in ("input_tokens", "output_tokens"):
        if not isinstance(answer["usage"][key], int) or answer["usage"][key] < 0:
            raise ValueError("invalid upstream usage")
    return answer


def answer_openai(answer):
    text, calls = [], []
    for block in answer["content"]:
        if block["type"] == "text":
            text.append(block["text"])
        elif block["type"] == "tool_use":
            calls.append({"id": block["id"], "type": "function", "function": {"name": block["name"], "arguments": dumps(block["input"])}})
        else:
            raise ValueError("unsupported answer block")
    message = {"role": "assistant", "content": "".join(text) if text else None}
    if calls:
        message["tool_calls"] = calls
    usage = answer["usage"]
    return {"id": answer["id"], "object": "chat.completion", "created": 0, "model": answer["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": STOP_TO_OPENAI[answer["stop_reason"]], "logprobs": None}],
            "usage": {"prompt_tokens": usage["input_tokens"], "completion_tokens": usage["output_tokens"],
                      "total_tokens": usage["input_tokens"] + usage["output_tokens"]}}


def frame(event, value):
    return (("event: " + event + "\n") if event else "") + "data: " + dumps(value) + "\n\n"


def render_stream(answer, api):
    if api == "anthropic":
        start = {**answer, "content": [], "stop_reason": None, "stop_sequence": None,
                 "usage": {**answer["usage"], "output_tokens": 0}}
        yield frame("message_start", {"type": "message_start", "message": start})
        for i, block in enumerate(answer["content"]):
            if block["type"] == "text":
                initial = {"type": "text", "text": ""}
                delta = {"type": "text_delta", "text": block["text"]}
            elif block["type"] == "tool_use":
                initial = {**block, "input": {}}
                delta = {"type": "input_json_delta", "partial_json": dumps(block["input"])}
            else:
                raise ValueError("unsupported streaming block")
            yield frame("content_block_start", {"type": "content_block_start", "index": i, "content_block": initial})
            yield frame("content_block_delta", {"type": "content_block_delta", "index": i, "delta": delta})
            yield frame("content_block_stop", {"type": "content_block_stop", "index": i})
        yield frame("message_delta", {"type": "message_delta", "delta": {
            "stop_reason": answer["stop_reason"], "stop_sequence": answer.get("stop_sequence")},
            "usage": {"output_tokens": answer["usage"]["output_tokens"]}})
        yield frame("message_stop", {"type": "message_stop"})
    else:
        result = answer_openai(answer)
        base = {k: result[k] for k in ("id", "created", "model")}
        base["object"] = "chat.completion.chunk"
        def chunk(delta, reason=None):
            return frame(None, {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]})
        yield chunk({"role": "assistant", "content": ""})
        message = result["choices"][0]["message"]
        if message["content"] is not None:
            yield chunk({"content": message["content"]})
        for i, call in enumerate(message.get("tool_calls", [])):
            yield chunk({"tool_calls": [{"index": i, **call, "function": {"name": call["function"]["name"], "arguments": ""}}]})
            yield chunk({"tool_calls": [{"index": i, "function": {"arguments": call["function"]["arguments"]}}]})
        yield chunk({}, result["choices"][0]["finish_reason"])
        yield frame(None, {**base, "choices": [], "usage": result["usage"]})
        yield "data: [DONE]\n\n"


def decode_answer(raw, wire, streaming, model):
    if len(raw) > MAX_BYTES:
        raise ValueError("response too large")
    value = ({"openai": openai_stream, "anthropic": anthropic_stream, "gemini": gemini_stream}[wire](raw)
             if streaming else loads(raw))
    return normalize_answer(value, wire, model)
