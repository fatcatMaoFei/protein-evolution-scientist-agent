"""Native CodeFlow OpenAI Responses transport for the shared concrete policy."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from importlib.metadata import version
from urllib.parse import urlsplit

from llm_agent import ARRAY_STR, MAX_TOKENS, STRING, AgentError, choose_candidates

GPT_MODEL = "gpt-6-luna"
DEFAULT_BASE = "https://codeflow.sh/v1"
FINAL_PROPERTIES = {"candidate_ids": ARRAY_STR, "critic": STRING, "selection_rationale": STRING}
FINAL_SCHEMA = {
    "type": "object",
    "properties": FINAL_PROPERTIES,
    "required": list(FINAL_PROPERTIES),
    "additionalProperties": False,
}


@lru_cache(maxsize=1)
def _client(endpoint: str):
    from openai import OpenAI

    secret = os.environ.get("CODEFLOW_GPT_API_KEY")
    if not secret:
        raise AgentError("Set CODEFLOW_GPT_API_KEY for the live CodeFlow GPT experiment")
    return OpenAI(base_url=endpoint, api_key=secret, max_retries=0, timeout=90.0)


def _response_input(messages: list[dict], native_continuation: list[dict] | None) -> tuple[str, list[dict]]:
    instructions, inputs = [], []
    for message in messages:
        role = message["role"]
        if role == "system":
            instructions.append(message["content"])
        elif role == "assistant" and "tool_calls" in message:
            expected_ids = [call["id"] for call in message["tool_calls"]]
            actual_ids = [
                item["call_id"] for item in native_continuation or [] if item["type"] == "function_call"
            ]
            if expected_ids != actual_ids:
                raise AgentError("GPT continuation must contain its original native function call")
            inputs.extend(native_continuation)
        elif role == "tool":
            inputs.append({
                "type": "function_call_output", "call_id": message["tool_call_id"], "output": message["content"],
            })
        else:
            inputs.append({"role": role, "content": message["content"]})
    return "\n".join(instructions), inputs


def _matches_model(requested: str, reported: str | None) -> bool:
    return bool(reported) and (
        reported == requested or re.fullmatch(re.escape(requested) + r"-\d{4}-\d{2}-\d{2}", reported) is not None
    )


class GPTTransport:
    model = GPT_MODEL
    config = {
        "backend": "CodeFlow OpenAI Responses",
        "max_output_tokens": MAX_TOKENS,
        "reasoning": {"effort": "none"},
        "store": False,
        "parallel_tool_calls": False,
        "final_format": "strict_json_schema",
    }

    @property
    def endpoint(self) -> str:
        endpoint = os.environ.get("CODEFLOW_GPT_BASE_URL", DEFAULT_BASE).rstrip("/")
        parts = urlsplit(endpoint)
        if (
            parts.netloc != "codeflow.sh"
            or parts.scheme != "https"
            or parts.path not in {"/v1", "/v1/responses"}
            or parts.query
            or parts.fragment
        ):
            raise AgentError("GPT requires the CodeFlow HTTPS Responses endpoint without credentials or query text")
        return DEFAULT_BASE

    def encode(self, request: dict, *, native_continuation: list[dict] | None) -> dict:
        instructions, inputs = _response_input(request["messages"], native_continuation)
        final = request["tool_choice"] == "none"
        choice = "none" if final else {
            "type": "function",
            "name": request["tool_choice"]["function"]["name"],
        }
        payload = {
            "model": request["model"],
            "instructions": instructions,
            "input": inputs,
            "tools": [
                {"type": "function", **tool["function"], "strict": True}
                for tool in request["tools"]
            ],
            "tool_choice": choice,
            "parallel_tool_calls": self.config["parallel_tool_calls"],
            "max_output_tokens": request["max_tokens"],
            "reasoning": self.config["reasoning"],
            "store": self.config["store"],
        }
        if self.config["reasoning"]["effort"] != "none":
            payload["include"] = ["reasoning.encrypted_content"]
        if final:
            payload["text"] = {
                "format": {"type": "json_schema", "name": "concrete_final", "schema": FINAL_SCHEMA, "strict": True},
            }
        return payload

    def generate(self, endpoint: str, request: dict) -> dict:
        from openai import APIError, APIStatusError

        try:
            response = _client(endpoint).responses.create(**request)
        except APIStatusError as exc:
            error_code = getattr(exc, "code", None)
            code = f" ({error_code})" if isinstance(error_code, str) and error_code in {
                "invalid_api_key", "model_not_found", "invalid_request_error", "rate_limit_exceeded",
                "insufficient_quota", "server_error", "permission_denied", "unsupported_parameter",
            } else ""
            raise AgentError(f"CodeFlow GPT HTTP status {exc.status_code}{code}") from None
        except APIError as exc:
            raise AgentError(f"CodeFlow GPT connection failed ({type(exc).__name__})") from None
        if not _matches_model(request["model"], response.model) or not response.id or response.usage is None:
            raise AgentError("CodeFlow response omitted the requested GPT identity, response ID or token usage")
        usage = response.usage.model_dump(mode="json", exclude_none=True)
        if not all(isinstance(usage.get(field), int) for field in ("input_tokens", "output_tokens")):
            raise AgentError("CodeFlow GPT response did not report input and output tokens")
        items = [item.model_dump(mode="json", exclude_none=True, exclude_unset=True) for item in response.output]
        calls = [
            {
                "id": item["call_id"],
                "type": "function",
                "function": {"name": item["name"], "arguments": item["arguments"]},
            }
            for item in items if item["type"] == "function_call"
        ]
        status = response.status
        if status == "completed":
            finish_reason = "tool_calls" if calls else "stop"
        elif status == "incomplete" and getattr(response.incomplete_details, "reason", None) == "max_output_tokens":
            finish_reason = "length"
        else:
            finish_reason = status or "unknown"
        return {
            "text": response.output_text,
            "tool_calls": calls,
            "finish_reason": finish_reason,
            "native_content": items,
            "provider_status": status,
            "model": response.model,
            "response_id": response.id,
            "request_id": getattr(response, "_request_id", None),
            "usage": {field: usage[field] for field in ("input_tokens", "output_tokens")},
            "provider_usage": usage,
            "runtime": {"openai": version("openai")},
        }


def choose_gpt_candidates(**kwargs):
    """Use the native GPT transport with the frozen shared decision policy."""
    return choose_candidates(**kwargs, transport=GPTTransport())
