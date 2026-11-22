"""Opt-in Chat Completions model profiles for the GFP workflow."""

from __future__ import annotations

from getpass import GetPassWarning, getpass
from importlib.metadata import version
import ipaddress
import json
import os
from pathlib import Path
import re
from secrets import token_hex
import stat
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPHandler, HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener
import warnings

from llm_agent import AgentError


DEFAULT_PROFILE_DIR = Path(__file__).resolve().parent / ".gfp-local" / "models"
BUILTIN_PROFILES = frozenset({"deepseek", "sonnet", "gpt", "qwen"})
_PROFILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_MODEL_NAME = re.compile(r"[A-Za-z0-9._:/@-]{1,160}")
_LOCAL_ENV_NAME = "GFP_PROFILE_API_KEY"
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def _profile_name(name: str) -> str:
    if not isinstance(name, str) or not _PROFILE_NAME.fullmatch(name):
        raise AgentError("Choose a profile name consisting of 1–40 letters, digits, hyphens or underscores")
    if name.lower() in BUILTIN_PROFILES:
        raise AgentError("Choose a name other than a built-in model profile")
    return name


def normalize_base_url(url: str) -> str:
    """Accept an HTTPS base URL or HTTP to a literal loopback address."""
    if not isinstance(url, str) or not url:
        raise AgentError("Enter an HTTPS API base URL")
    if any(char.isspace() or ord(char) < 32 for char in url):
        raise AgentError("The API base URL must not contain whitespace or control characters")
    if "?" in url or "#" in url or "\\" in url:
        raise AgentError("The API base URL must not contain a query, fragment or backslash")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        raise AgentError("The API base URL is invalid") from None
    if (not parsed.hostname or parsed.username is not None or parsed.password is not None
            or "@" in parsed.netloc or "%" in parsed.netloc or port == 0):
        raise AgentError("Use an API base URL without embedded credentials")
    if parsed.scheme != "https":
        try:
            is_loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            is_loopback = False
        if parsed.scheme != "http" or not is_loopback:
            raise AgentError("Use HTTPS for a remote API base URL")
    path = parsed.path.rstrip("/").removesuffix("/chat/completions").rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _validate_key(api_key: str) -> str:
    if not isinstance(api_key, str) or not api_key or any(
        not char.isascii() or char.isspace() or not char.isprintable() for char in api_key
    ):
        raise AgentError("Provide a nonempty API key without whitespace or control characters")
    return api_key


class OpenAICompatibleTransport:
    """Chat Completions adapter with a bound credential, separate from its serialized wire."""

    def __init__(self, *, model_id: str, base_url: str, profile: str, api_key: str | None):
        if not isinstance(model_id, str) or not _MODEL_NAME.fullmatch(model_id):
            raise AgentError("Enter a valid model ID without spaces")
        self.model = model_id
        self.endpoint = normalize_base_url(base_url)
        self.profile = _profile_name(profile)
        self._api_key = None if api_key is None else _validate_key(api_key)
        if self._api_key is not None and (
            self._api_key in unquote(self.endpoint) or self._api_key in self.model or self._api_key in self.profile
        ):
            raise AgentError("Keep API credentials separate from the endpoint, model ID and profile name")
        self.config = {
            "backend": "OpenAI-compatible Chat Completions",
            "protocol": "openai",
            "profile": self.profile,
            "proposal_tool_choice": "forced_native",
            "final_format": "json_object",
        }

    def encode(self, request: dict, *, native_continuation=None) -> dict:
        """Keep to the common Chat Completions schema."""
        fields = ("model", "messages", "tools", "tool_choice", "temperature", "max_tokens", "response_format")
        return {name: request[name] for name in fields if name in request}

    def _request(self, endpoint: str, wire: dict):
        from openai.types.chat import ChatCompletion

        # The standard-library handlers run at debuglevel zero. The optional SDK
        # debug logger can emit raw response headers, so only reuse its schema.
        handlers = [_NoRedirect(), HTTPHandler(debuglevel=0), HTTPSHandler(debuglevel=0)]
        if endpoint.startswith("http:"):
            handlers.append(ProxyHandler({}))
        request = Request(
            f"{endpoint}/chat/completions",
            data=json.dumps(wire, ensure_ascii=True, allow_nan=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": f"OpenAI/Python {version('openai')}",
                "X-Stainless-Lang": "python",
                "X-Stainless-Package-Version": version("openai"),
            },
            method="POST",
        )
        try:
            with build_opener(*handlers).open(request, timeout=90) as received:
                body = received.read(_MAX_RESPONSE_BYTES + 1)
                request_id = received.headers.get("x-request-id")
        except HTTPError as exc:
            exc.close()
            raise AgentError(f"OpenAI-compatible HTTP status {exc.code}") from None
        except OSError:
            raise AgentError("OpenAI-compatible connection failed") from None
        if len(body) > _MAX_RESPONSE_BYTES:
            raise AgentError("OpenAI-compatible response exceeded the size limit")
        try:
            completion = ChatCompletion.model_validate_json(body)
        except (TypeError, ValueError):
            raise AgentError("Provider returned an invalid Chat Completions response") from None
        object.__setattr__(completion, "_request_id", request_id)
        return completion

    def generate(self, endpoint: str, wire: dict) -> dict:
        from openai.types.chat import ChatCompletion

        if self._api_key is None:
            raise AgentError("This profile was loaded for replay without a live API key")
        if endpoint != self.endpoint or wire.get("model") != self.model:
            raise AgentError("Live request endpoint or model does not match the selected profile")
        completion = self._request(endpoint, wire)
        if not isinstance(completion, ChatCompletion):
            raise AgentError("Provider returned an invalid Chat Completions response")
        choice = completion.choices[0] if len(completion.choices) == 1 else None
        usage = completion.usage
        if choice is None or usage is None or not completion.model or not completion.id:
            raise AgentError("Provider response must include one choice, token usage, ID and served model")
        if not all(isinstance(n, int) and n >= 0 for n in (usage.prompt_tokens, usage.completion_tokens)):
            raise AgentError("Provider omitted valid prompt or completion token counts")
        response = {
            "text": choice.message.content or "",
            "finish_reason": choice.finish_reason,
            "tool_calls": [tool.model_dump(exclude_none=True) for tool in choice.message.tool_calls or []],
            "model": completion.model,
            "response_id": completion.id,
            "id": completion.id,
            "request_id": getattr(completion, "_request_id", None),
            "created": completion.created,
            "usage": {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens},
            "provider_usage": usage.model_dump(exclude_none=True),
            "runtime": {"openai": version("openai")},
        }
        key_wire = json.dumps(self._api_key, ensure_ascii=True)[1:-1]
        if key_wire in json.dumps(response, ensure_ascii=True):
            raise AgentError("Provider returned credential-bearing content, which was withheld")
        return response


def probe_openai_transport(adapter: OpenAICompatibleTransport) -> dict:
    """Exercise two native requests using only throwaway setup data."""
    name = "gfp_setup_probe"
    challenge = token_hex(8)
    receipt = token_hex(12)
    tools = [{
        "type": "function",
        "function": {
            "name": name,
            "description": "Returns a receipt used to verify the tool result is received.",
            "parameters": {
                "type": "object",
                "properties": {"challenge": {"type": "string"}},
                "required": ["challenge"],
                "additionalProperties": False,
            },
        },
    }]
    messages = [
        {"role": "system", "content": (
            "Use the supplied function once. After the tool result, return exactly one JSON object "
            "with key receipt and the literal value from the result."
        )},
        {"role": "user", "content": f"Call {name} with challenge {challenge}."},
    ]
    request = {
        "model": adapter.model, "messages": messages, "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": name}},
        "temperature": 0.0, "max_tokens": 200,
    }
    first = adapter.generate(adapter.endpoint, adapter.encode(request))
    calls = first["tool_calls"]
    if first["finish_reason"] != "tool_calls" or len(calls) != 1:
        raise AgentError("Model setup needs exactly one native tool call in the first response")
    call = calls[0]
    function = call.get("function", {})
    try:
        args = json.loads(function.get("arguments", ""))
    except (TypeError, json.JSONDecodeError):
        raise AgentError("Model setup received malformed native function arguments") from None
    if (call.get("type") != "function" or not call.get("id") or function.get("name") != name
            or args != {"challenge": challenge}):
        raise AgentError("Model setup received an incorrect native function call")

    continuation = [
        *messages,
        {"role": "assistant", "content": first["text"] or None, "tool_calls": calls},
        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"receipt": receipt})},
    ]
    request = {
        **request, "messages": continuation, "tool_choice": "none",
        "response_format": {"type": "json_object"},
    }
    second = adapter.generate(adapter.endpoint, adapter.encode(request))
    try:
        result = json.loads(second["text"])
    except (TypeError, json.JSONDecodeError):
        raise AgentError("Model setup follow-up must return a JSON object") from None
    if second["finish_reason"] != "stop" or second["tool_calls"] or result != {"receipt": receipt}:
        raise AgentError("Model setup follow-up did not reproduce the actual tool result")
    return {
        "native_tool_call": True,
        "tool_result_consumed": True,
        "json_object": True,
        "served_models": [first["model"], second["model"]],
        "response_ids": [first["response_id"], second["response_id"]],
    }


def _prompt_key() -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("error", GetPassWarning)
        try:
            return _validate_key(getpass("Model API key (hidden): "))
        except (GetPassWarning, EOFError):
            raise AgentError("No hidden terminal is available. Use --key-env instead.") from None


def _from_env(name: str) -> str:
    if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
        raise AgentError("Enter a valid API key environment variable name")
    value = os.environ.get(name)
    if not value:
        raise AgentError("The selected API key environment variable is not set")
    return _validate_key(value)


def _write_new(path: Path, text: str) -> None:
    """Publish one private file atomically without replacing a prior file."""
    stream = NamedTemporaryFile(dir=path.parent, prefix=".model-profile-", delete=False)
    pending = Path(stream.name)
    try:
        with stream:
            stream.write(text.encode("utf-8"))
        os.link(pending, path)
    finally:
        pending.unlink()


def _read_local_key(path: Path) -> str:
    try:
        handle = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(handle, encoding="utf-8") as stream:
            metadata = os.fstat(stream.fileno())
            if (not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_mode & 0o077
                    or metadata.st_uid != os.getuid()):
                raise AgentError("The profile key file must be a private regular file owned by the current user")
            data = stream.read(8193)
    except (OSError, UnicodeError):
        raise AgentError("Could not read the private profile key file") from None
    prefix = f"{_LOCAL_ENV_NAME}="
    if len(data) > 8192 or not data.startswith(prefix) or not data.endswith("\n"):
        raise AgentError("The profile key file is invalid")
    return _validate_key(data[len(prefix):-1])


def setup_openai_profile(
    name: str, *, base_url: str, model_id: str, key_env: str | None = None,
    profile_dir: Path = DEFAULT_PROFILE_DIR,
) -> dict:
    """Probe a provider before saving a new profile. Existing files are preserved."""
    name = _profile_name(name)
    profile_dir = Path(profile_dir)
    profile_path = profile_dir / f"{name}.json"
    key_path = profile_dir / f"{name}.env"
    if os.path.lexists(profile_path) or os.path.lexists(key_path):
        raise AgentError("A profile file already exists for that name. Choose a different name.")
    if key_env is None and profile_dir.resolve() != DEFAULT_PROFILE_DIR:
        raise AgentError("Use --key-env with a custom profile directory")
    # Validate public settings before prompting for an API key or calling the provider.
    preview = OpenAICompatibleTransport(model_id=model_id, base_url=base_url, profile=name, api_key=None)
    api_key = _from_env(key_env) if key_env is not None else _prompt_key()
    adapter = OpenAICompatibleTransport(model_id=preview.model, base_url=preview.endpoint,
                                        profile=name, api_key=api_key)
    verification = probe_openai_transport(adapter)
    key_source = {"mode": "env", "name": key_env} if key_env is not None else {"mode": "private_env_file"}
    profile = {
        "name": name, "protocol": "openai", "base_url": adapter.endpoint, "model_id": model_id,
        "credential": key_source, "verification": verification,
    }
    encoded = json.dumps(profile, indent=2, ensure_ascii=True) + "\n"
    if json.dumps(api_key, ensure_ascii=True)[1:-1] in encoded:
        raise AgentError("Detected a credential in public profile data")
    local_key_written = False
    try:
        profile_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if key_env is None:
            _write_new(key_path, f"{_LOCAL_ENV_NAME}={api_key}\n")
            local_key_written = True
        _write_new(profile_path, encoded)
    except OSError:
        if local_key_written:
            key_path.unlink()
        raise AgentError("Could not save the verified profile without overwriting files") from None
    return profile


def _read_profile(name: str, profile_dir: Path) -> dict:
    name = _profile_name(name)
    try:
        value = json.loads((profile_dir / f"{name}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise AgentError("Could not read the selected OpenAI-compatible profile") from None
    valid_probe = ("native_tool_call", "tool_result_consumed", "json_object")
    if (
        not isinstance(value, dict) or value.get("protocol") != "openai" or value.get("name") != name
        or not isinstance(value.get("verification"), dict)
        or not all(value["verification"].get(key) is True for key in valid_probe)
    ):
        raise AgentError("The selected profile has no compatible native-tool verification record")
    return value


def load_openai_profile(
    name: str, *, profile_dir: Path = DEFAULT_PROFILE_DIR, require_key: bool = True
) -> OpenAICompatibleTransport:
    """Read a verified profile and bind its key only for live generation."""
    profile_dir = Path(profile_dir)
    value = _read_profile(name, profile_dir)
    credential = value.get("credential")
    key = None
    if require_key:
        if not isinstance(credential, dict):
            raise AgentError("The profile has no usable credential reference")
        if credential.get("mode") == "env":
            key = _from_env(credential.get("name"))
        elif credential.get("mode") == "private_env_file":
            key = _read_local_key(profile_dir / f"{name}.env")
        else:
            raise AgentError("The profile has no usable credential source")
    return OpenAICompatibleTransport(
        model_id=value.get("model_id"), base_url=value.get("base_url"), profile=name, api_key=key
    )


def list_openai_profiles(*, profile_dir: Path = DEFAULT_PROFILE_DIR) -> list[dict]:
    """List only public metadata for verified local profiles."""
    result = []
    profile_dir = Path(profile_dir)
    for path in sorted(profile_dir.glob("*.json")):
        adapter = load_openai_profile(path.stem, profile_dir=profile_dir, require_key=False)
        result.append({"name": adapter.profile, "protocol": "openai",
                       "base_url": adapter.endpoint, "model_id": adapter.model})
    return result
