from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import http.client
import json
from threading import Thread
from urllib.error import HTTPError

import pytest
from openai.types.chat import ChatCompletion

from gfp_setup import (
    OpenAICompatibleTransport,
    list_openai_profiles,
    load_openai_profile,
    normalize_base_url,
    setup_openai_profile,
)
from llm_agent import AgentError


TEST_KEY = "sk-local-setup-fixture-d16f28630"


def _completion(message, stop, identifier):
    return ChatCompletion.model_validate({
        "object": "chat.completion", "id": identifier, "created": 1, "model": "fixture/text",
        "choices": [{"index": 0, "message": message, "finish_reason": stop}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38},
    })


def _native_probe_provider(events):
    def request(self, endpoint, wire):
        assert self._api_key == TEST_KEY
        assert endpoint == "https://models.example/v1"
        assert TEST_KEY not in repr(wire)
        events.append(wire)
        if len(events) == 1:
            assert wire["tool_choice"] == {"type": "function", "function": {"name": "gfp_setup_probe"}}
            challenge = wire["messages"][-1]["content"].split()[-1].rstrip(".")
            return _completion({
                "role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_fixture",
                    "type": "function",
                    "function": {"name": "gfp_setup_probe", "arguments": json.dumps({"challenge": challenge})},
                }],
            }, "tool_calls", "response_1")
        assert wire["tool_choice"] == "none"
        assert wire["response_format"] == {"type": "json_object"}
        assert wire["messages"][-1]["role"] == "tool"
        assert wire["messages"][-1]["tool_call_id"] == "call_fixture"
        receipt = json.loads(wire["messages"][-1]["content"])
        return _completion({"role": "assistant", "content": json.dumps(receipt)}, "stop", "response_2")
    return request


def test_endpoint_validation():
    assert normalize_base_url("https://models.example/v1/chat/completions/") == "https://models.example/v1"
    assert normalize_base_url("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080/v1"
    assert normalize_base_url("http://[::1]:8080") == "http://[::1]:8080"
    for url in (
        "http://models.example/v1",
        "https://name:secret@models.example/v1",
        "https://@models.example/v1",
        "https://models.example/v1?key=secret",
        "https://models.example/v1#anchor",
        "https://models.example/\n/path",
    ):
        with pytest.raises(AgentError):
            normalize_base_url(url)


def test_env_profile_requires_actual_tool_and_replays_without_key(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(OpenAICompatibleTransport, "_request", _native_probe_provider(events))
    monkeypatch.setenv("GFP_PROFILE_FIXTURE_KEY", TEST_KEY)
    profile = setup_openai_profile("fixture", base_url="https://models.example/v1", model_id="fixture/text",
                                   key_env="GFP_PROFILE_FIXTURE_KEY", profile_dir=tmp_path)
    assert len(events) == 2
    assert profile["verification"]["response_ids"] == ["response_1", "response_2"]
    assert profile["verification"]["tool_result_consumed"]
    assert TEST_KEY not in (tmp_path / "fixture.json").read_text()
    assert not (tmp_path / "fixture.env").exists()
    model = load_openai_profile("fixture", profile_dir=tmp_path)
    assert model.model == "fixture/text"
    assert TEST_KEY not in repr(model.config)
    monkeypatch.delenv("GFP_PROFILE_FIXTURE_KEY")
    assert list_openai_profiles(profile_dir=tmp_path) == [{
        "name": "fixture", "protocol": "openai", "base_url": "https://models.example/v1", "model_id": "fixture/text"
    }]
    replay = load_openai_profile("fixture", profile_dir=tmp_path, require_key=False)
    with pytest.raises(AgentError, match="replay"):
        replay.generate(replay.endpoint, {"model": replay.model})
    with pytest.raises(AgentError, match="not set"):
        load_openai_profile("fixture", profile_dir=tmp_path)


def test_local_profile_uses_private_file_and_never_overwrites(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr("gfp_setup.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("gfp_setup._prompt_key", lambda: TEST_KEY)
    monkeypatch.setattr(OpenAICompatibleTransport, "_request", _native_probe_provider(events))
    kwargs = dict(base_url="https://models.example/v1", model_id="fixture/text", profile_dir=tmp_path)
    setup_openai_profile("saved", **kwargs)
    keyfile = tmp_path / "saved.env"
    assert keyfile.stat().st_mode & 0o077 == 0
    assert TEST_KEY not in (tmp_path / "saved.json").read_text()
    assert load_openai_profile("saved", profile_dir=tmp_path)._api_key == TEST_KEY
    with pytest.raises(AgentError, match="already exists"):
        setup_openai_profile("saved", **kwargs)
    assert len(events) == 2
    keyfile.chmod(0o644)
    with pytest.raises(AgentError, match="private"):
        load_openai_profile("saved", profile_dir=tmp_path)


def test_plain_text_fallback_does_not_save_or_pass(monkeypatch, tmp_path):
    monkeypatch.setenv("GFP_PROFILE_FIXTURE_KEY", TEST_KEY)
    monkeypatch.setattr(
        OpenAICompatibleTransport, "_request",
        lambda self, endpoint, wire: _completion({"role": "assistant", "content": "OK"}, "stop", "response_1"),
    )
    with pytest.raises(AgentError, match="exactly one native tool call"):
        setup_openai_profile("broken", base_url="https://models.example/v1", model_id="fixture/text",
                             key_env="GFP_PROFILE_FIXTURE_KEY", profile_dir=tmp_path)
    assert not (tmp_path / "broken.json").exists()
    assert not (tmp_path / "broken.env").exists()


def test_custom_location_requires_env_and_provider_errors_are_sanitized(monkeypatch, tmp_path):
    with pytest.raises(AgentError, match="--key-env"):
        setup_openai_profile("fixture", base_url="https://models.example/v1", model_id="fixture/text",
                             profile_dir=tmp_path)
    adapter = OpenAICompatibleTransport(model_id="fixture/text", base_url="https://models.example/v1",
                                        profile="fixture", api_key=TEST_KEY)

    class ErrorOpener:
        def open(self, request, timeout):
            assert request.get_header("Authorization") == f"Bearer {TEST_KEY}"
            raise HTTPError(request.full_url, 401, "Unauthorized", {"X-Echo": TEST_KEY},
                            BytesIO(f"provider body {TEST_KEY}".encode()))

    monkeypatch.setattr("gfp_setup.build_opener", lambda *handlers: ErrorOpener())
    with pytest.raises(AgentError, match="HTTP status 401") as exc:
        adapter.generate(adapter.endpoint, {"model": adapter.model})
    assert TEST_KEY not in str(exc.value)
    monkeypatch.setattr(
        OpenAICompatibleTransport, "_request",
        lambda self, endpoint, wire: _completion({"role": "assistant", "content": TEST_KEY}, "stop", "response_1"),
    )
    with pytest.raises(AgentError, match="withheld") as exc:
        adapter.generate(adapter.endpoint, {"model": adapter.model})
    assert TEST_KEY not in str(exc.value)


def test_live_http_transport_withholds_echoed_headers_even_when_debug_enabled(monkeypatch, capsys):
    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            assert self.headers["Authorization"] == f"Bearer {TEST_KEY}"
            length = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(length))
            assert request["model"] == "fixture/text"
            response = _completion(
                {"role": "assistant", "content": "public response"}, "stop", "response_1"
            ).model_dump_json()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("X-Reflected-Header", TEST_KEY)
            self.end_headers()
            self.wfile.write(response.encode())

        def log_message(self, *args):
            return None

    monkeypatch.setattr(http.client.HTTPConnection, "debuglevel", 1)
    monkeypatch.setenv("OPENAI_LOG", "debug")
    with ThreadingHTTPServer(("127.0.0.1", 0), Provider) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_port}/v1"
            adapter = OpenAICompatibleTransport(model_id="fixture/text", base_url=endpoint,
                                                profile="fixture", api_key=TEST_KEY)
            outcome = adapter.generate(endpoint, {"model": adapter.model, "messages": []})
        finally:
            server.shutdown()
            worker.join()
    assert outcome["text"] == "public response"
    assert TEST_KEY not in repr(outcome)
    capture = capsys.readouterr()
    assert TEST_KEY not in capture.out + capture.err
