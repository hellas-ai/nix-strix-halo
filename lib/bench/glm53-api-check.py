#!/usr/bin/env python3
"""Exercise real generation, streamed tool calls, and tool-result continuation.

This is a protocol/coherence gate, not coding or long-context qualification.
The file tool is a deterministic fixture and never reads the real filesystem.
"""

import argparse
import json
import time
import urllib.request


def chat(base, messages, **options):
    payload = {
        "model": "glm-5.3-flash",
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1024,
        "reasoning_effort": "low",
        "chat_template_kwargs": {"clear_thinking": True},
        **options,
    }
    request = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=900) as response:
        if not payload.get("stream"):
            result = json.load(response)
        else:
            calls, content, reasoning, first, usage, finish = {}, [], [], None, None, None
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if any(delta.get(k) for k in ("content", "reasoning_content", "tool_calls")) and first is None:
                        first = time.monotonic() - started
                    content.append(delta.get("content") or "")
                    reasoning.append(delta.get("reasoning_content") or "")
                    finish = choice.get("finish_reason") or finish
                    for part in delta.get("tool_calls") or []:
                        call = calls.setdefault(part["index"], {
                            "id": "", "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
                        if part.get("id"):
                            call["id"] += part["id"]
                        for key in ("name", "arguments"):
                            call["function"][key] += part.get("function", {}).get(key) or ""
            message = {"role": "assistant", "content": "".join(content),
                       "reasoning_content": "".join(reasoning)}
            if calls:
                message["tool_calls"] = [calls[i] for i in sorted(calls)]
            result = {"choices": [{"message": message, "finish_reason": finish}],
                      "usage": usage, "first_event_seconds": first}
    result["wall_seconds"] = time.monotonic() - started
    print(json.dumps(result), flush=True)
    assert result["choices"][0]["finish_reason"] != "length", "Output truncated"
    return result["choices"][0]["message"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30053/v1")
    args = parser.parse_args()
    message = chat(args.base_url, [{"role": "user", "content":
                    "Calculate 137 * 29. Give only the integer as your final answer."}])
    assert message["content"].strip() == "3973", message

    tools = [{"type": "function", "function": {
        "name": "read_file", "description": "Read the text of a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"], "additionalProperties": False},
    }}]
    messages = [{"role": "user", "content":
                 "Use read_file to read marker.txt. Then reply with the exact marker inside, and nothing else."}]
    message = chat(args.base_url, messages, tools=tools, tool_choice="auto",
                   stream=True, stream_options={"include_usage": True})
    calls = message.get("tool_calls", [])
    assert len(calls) == 1, message
    call = calls[0]
    assert call["id"] and call["function"]["name"] == "read_file", call
    assert json.loads(call["function"]["arguments"]) == {"path": "marker.txt"}, call
    messages.extend([message, {"role": "tool", "tool_call_id": call["id"],
                               "content": "GLM_AGENT_ACCEPTANCE_53"}])
    message = chat(args.base_url, messages, tools=tools, stream=True,
                   stream_options={"include_usage": True})
    assert message["content"].strip() == "GLM_AGENT_ACCEPTANCE_53", message
    print(json.dumps({"protocol_and_coherence": "passed"}), flush=True)


if __name__ == "__main__":
    main()
