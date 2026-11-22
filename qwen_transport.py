"""Pinned local MLX transport for the shared concrete decision experiment."""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from importlib.metadata import version

from llm_agent import MAX_TOKENS, AgentError, choose_candidates


QWEN_MODEL = "mlx-community/Qwen3-4B-4bit"
QWEN_REVISION = "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25"
LOCAL_ENDPOINT = "local:mlx-lm"
INFERENCE = ThreadPoolExecutor(max_workers=1, thread_name_prefix="concrete-mlx")
TOOL_CALL = re.compile(r"\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*", re.DOTALL)
TOOL_PREFILL = '<tool_call>\n{"name":"evaluate_concrete_shortlist","arguments":'


@lru_cache(maxsize=1)
def _load_model():
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from mlx_lm import load

    options = {
        "revision": QWEN_REVISION, "token": False,
        "allow_patterns": ["*.json", "*.jinja", "*.safetensors", "*.txt"],
    }
    try:
        snapshot = snapshot_download(QWEN_MODEL, local_files_only=True, **options)
    except LocalEntryNotFoundError:
        snapshot = snapshot_download(QWEN_MODEL, **options)
    return load(snapshot, tokenizer_config={"trust_remote_code": False})


def _rendered_messages(encoded):
    messages = []
    instruction = next(message["content"] for message in encoded["messages"] if message["role"] == "user")
    budget = json.loads(instruction.split("\n", 1)[1])["budget"]
    for message in encoded["messages"]:
        item = dict(message)
        if item["role"] == "assistant" and "tool_calls" in item:
            item["content"] = item["content"] or ""
            item["tool_calls"] = [
                {"type": "function", "function": {
                    "name": tool["function"]["name"], "arguments": json.loads(tool["function"]["arguments"]),
                }} for tool in item["tool_calls"]
            ]
        messages.append(item)
        if item["role"] == "tool":
            messages.append({"role": "user", "content": f"Use the tool result for the final decision. Return the required JSON object with exactly {budget} distinct evaluated candidate aliases, without Markdown or a tool call."})
    return messages


def _generate_local(encoded):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = _load_model()
    rendered = tokenizer.apply_chat_template(
        _rendered_messages(encoded), tools=encoded["tools"],
        add_generation_prompt=True, tokenize=False, enable_thinking=encoded["enable_thinking"],
    )
    rendered += encoded["assistant_prefill"]
    prompt = tokenizer.encode(rendered, add_special_tokens=False)
    parts = []
    for piece in stream_generate(
        model, tokenizer, prompt, max_tokens=encoded["max_tokens"], sampler=make_sampler(temp=encoded["temperature"]),
    ):
        parts.append(piece.text)
    return rendered, "".join(parts), piece


def _parse_native_call(raw):
    match = TOOL_CALL.fullmatch(raw)
    if match is None:
        return []
    try:
        command = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    if not isinstance(command, dict) or set(command) != {"name", "arguments"} or not isinstance(command["arguments"], dict) or not isinstance(command["name"], str):
        return []
    return [{
        "id": "qwen-" + hashlib.sha256(raw.encode()).hexdigest()[:16], "type": "function",
        "function": {"name": command["name"], "arguments": json.dumps(command["arguments"], ensure_ascii=True, sort_keys=True)},
    }]


class QwenTransport:
    model = QWEN_MODEL
    endpoint = LOCAL_ENDPOINT
    config = {
        "backend": "local MLX via mlx_lm.stream_generate", "revision": QWEN_REVISION,
        "max_tokens": MAX_TOKENS, "enable_thinking": False, "temperature": 0.0,
        "proposal_tool_choice": "native named tool-call prefill, model-generated arguments schema validated in Python",
        "continuation_tool_choice": "tools omitted, JSON validated in Python",
    }

    def encode(self, request, *, native_continuation):
        return {
            "model": request["model"], "revision": QWEN_REVISION, "messages": request["messages"],
            "tools": request["tools"] if request["tool_choice"] != "none" else None,
            "assistant_prefill": TOOL_PREFILL if request["tool_choice"] != "none" else "",
            "enable_thinking": self.config["enable_thinking"], "temperature": self.config["temperature"],
            "max_tokens": request["max_tokens"], "tool_choice": request["tool_choice"],
        }

    def generate(self, endpoint, request):
        if endpoint != LOCAL_ENDPOINT or request["model"] != QWEN_MODEL or request["revision"] != QWEN_REVISION:
            raise AgentError("Local Qwen execution requires the registered MLX snapshot")
        rendered_prompt, raw, piece = INFERENCE.submit(_generate_local, request).result()
        tool_calls = _parse_native_call(request["assistant_prefill"] + raw) if request["tools"] is not None else []
        finish_reason = "tool_calls" if tool_calls and piece.finish_reason == "stop" else piece.finish_reason
        usage = {"input_tokens": piece.prompt_tokens, "output_tokens": piece.generation_tokens}
        identity = hashlib.sha256(json.dumps([QWEN_REVISION, rendered_prompt, raw], ensure_ascii=True).encode()).hexdigest()
        return {
            "text": "" if tool_calls else raw.strip(), "tool_calls": tool_calls, "finish_reason": finish_reason,
            "provider_stop_reason": piece.finish_reason, "raw_output": raw, "rendered_prompt": rendered_prompt,
            "model": QWEN_MODEL, "revision": QWEN_REVISION, "response_id": "local-" + identity, "request_id": None,
            "usage": usage, "provider_usage": usage,
            "runtime": {name: version(name) for name in ("mlx-lm", "mlx", "transformers", "huggingface-hub")},
        }


def choose_qwen_candidates(**kwargs):
    """Select concrete mixtures with the pinned local Qwen language model."""
    return choose_candidates(**kwargs, transport=QwenTransport())
