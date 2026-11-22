"""CodeFlow's native Anthropic Messages transport for the shared concrete policy."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib.metadata import version
from urllib.parse import urlsplit

from llm_agent import ARRAY_STR, MAX_TOKENS, STRING, AgentError, choose_candidates


SONNET_MODEL = "claude-sonnet-5-5"
FINAL_PROPERTIES = {"candidate_ids": ARRAY_STR, "critic": STRING, "selection_rationale": STRING}
FINAL_SCHEMA = {
    "type": "object",
    "properties": FINAL_PROPERTIES,
    "required": list(FINAL_PROPERTIES),
    "additionalProperties": False,
}


@lru_cache(maxsize=1)
def _client(endpoint):
    from anthropic import Anthropic

    secret = os.environ.get("CODEFLOW_SONNET_API_KEY")
    if not secret:
        raise AgentError("Set CODEFLOW_SONNET_API_KEY for the live CodeFlow Sonnet experiment")
    return Anthropic(base_url=endpoint, api_key=secret, max_retries=0, timeout=90.0)


def _native_messages(messages, native_continuation):
    system_parts, native = [], []
    for message in messages:
        role = message["role"]
        if role == "system":
            system_parts.append(message["content"])
            continue
        if role == "tool":
            role = "user"
            blocks = [{"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": message["content"]}]
        elif role == "assistant" and "tool_calls" in message:
            if native_continuation is None:
                raise AgentError("Sonnet continuation must include its original assistant content blocks")
            blocks = native_continuation
        else:
            blocks = [{"type": "text", "text": message["content"]}]
        if native and native[-1]["role"] == role:
            native[-1]["content"].extend(blocks)
        else:
            native.append({"role": role, "content": list(blocks)})
    return "\n".join(system_parts), native


class SonnetTransport:
    model = SONNET_MODEL
    config = {
        "backend": "CodeFlow Anthropic Messages", "max_tokens": MAX_TOKENS,
        "thinking": {"type": "between_tools"}, "effort": "low", "proposal_tool_choice": "auto",
    }

    @property
    def endpoint(self):
        endpoint = os.environ.get("CODEFLOW_SONNET_BASE_URL", "https://codeflow.sh").rstrip("/")
        parts = urlsplit(endpoint)
        if parts.netloc != "codeflow.sh" or parts.scheme != "https" or parts.path not in {"", "/v1", "/v1/messages"} or parts.query or parts.fragment:
            raise AgentError("Sonnet requires the CodeFlow HTTPS Messages endpoint without credentials or query text")
        return "https://codeflow.sh"

    def encode(self, request, *, native_continuation):
        system, messages = _native_messages(request["messages"], native_continuation)
        final = request["tool_choice"] == "none"
        options = {"effort": self.config["effort"]}
        if final:
            options["format"] = {"type": "json_schema", "schema": FINAL_SCHEMA}
        tool_specs = [tool["function"] for tool in request["tools"]]
        return {
            "model": request["model"], "messages": messages, "system": system,
            "max_tokens": request["max_tokens"], "thinking": self.config["thinking"], "output_config": options,
            "tool_choice": {"type": "none"} if final else {"type": "auto", "disable_parallel_tool_use": True},
            "tools": [
                {"name": tool["name"], "description": tool["description"], "input_schema": tool["parameters"], "strict": True}
                for tool in tool_specs
            ],
        }

    def generate(self, endpoint, request):
        from anthropic import APIError, APIStatusError

        try:
            message = _client(endpoint).messages.create(**request)
        except APIStatusError as exc:
            error_type = getattr(exc, "type", None)
            detail = f" ({error_type})" if error_type in {
                "invalid_request_error", "authentication_error", "permission_error", "not_found_error",
                "rate_limit_error", "overloaded_error", "api_error",
            } else ""
            raise AgentError(f"CodeFlow Sonnet HTTP status {exc.status_code}{detail}") from None
        except APIError as exc:
            raise AgentError(f"CodeFlow Sonnet connection failed ({type(exc).__name__})") from None
        usage = message.usage.model_dump(exclude_none=True)
        blocks = [block.model_dump(mode="json", exclude_none=True, exclude_unset=True) for block in message.content]
        if message.model != request["model"] or not message.id or "input_tokens" not in usage or "output_tokens" not in usage:
            raise AgentError("CodeFlow response omitted requested Sonnet identity, response ID or usage")
        calls = [
            {"id": block["id"], "type": "function", "function": {
                "name": block["name"], "arguments": json.dumps(block["input"], ensure_ascii=True, sort_keys=True),
            }}
            for block in blocks if block["type"] == "tool_use"
        ]
        normalized_stops = {"tool_use": "tool_calls", "end_turn": "stop", "max_tokens": "length"}
        input_tokens = sum(usage.get(field, 0) for field in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        return {
            "text": "".join(block["text"] for block in blocks if block["type"] == "text"),
            "tool_calls": calls, "finish_reason": normalized_stops.get(message.stop_reason, message.stop_reason),
            "native_content": blocks, "provider_stop_reason": message.stop_reason,
            "model": message.model, "response_id": message.id, "request_id": getattr(message, "_request_id", None),
            "usage": {"input_tokens": input_tokens, "output_tokens": usage["output_tokens"]},
            "provider_usage": usage, "runtime": {"anthropic": version("anthropic")},
        }


def choose_sonnet_candidates(**kwargs):
    """Use the native Sonnet transport with the frozen shared decision policy."""
    return choose_candidates(**kwargs, transport=SonnetTransport())
