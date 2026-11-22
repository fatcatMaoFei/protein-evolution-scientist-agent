"""Recover interrupted GPT rounds from authenticated, byte-checked responses."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from factorial_runner import run
from gpt_transport import GPT_MODEL, GPTTransport, choose_gpt_candidates
from llm_agent import AgentError, _digest, choose_candidates
from nonbio import write_json


class RecoveryTransport(GPTTransport):
    def __init__(self, archived, records):
        self.responses = {
            call["request_sha256"]: call for call in archived["calls"] if "response" in call
        }
        self.records = records

    def generate(self, endpoint, request):
        digest = _digest(request)
        if digest in self.responses:
            original = self.responses.pop(digest)
            response = original["response"]
            if _digest(original["request"]) != digest or _digest(response) != original["response_sha256"]:
                raise AgentError("Archived GPT request or response digest does not match")
            self.records.append({"event": "reused_response", "response_id": response["response_id"], "request_sha256": digest})
            return response
        for delay in (12, 24, 48, None):
            try:
                response = super().generate(endpoint, request)
                self.records.append({"event": "live_response", "response_id": response["response_id"], "request_sha256": digest})
                return response
            except AgentError as error:
                self.records.append({"event": "failure", "error": str(error), "request_sha256": digest})
                if delay is None or "HTTP status 429" not in str(error):
                    raise
                time.sleep(delay)
        raise AssertionError("unreachable")


def recover(output, archive, baseline, protocol):
    if json.loads((archive / "status.json").read_text())["status"] != "failed":
        raise ValueError("Recovery expects a failed formal GPT run")

    def selector(**kwargs):
        target = kwargs["log_path"]
        relative = target.relative_to(output.resolve())
        earlier = archive / relative
        raw = earlier.read_bytes() if earlier.is_file() else None
        trace = json.loads(raw) if raw is not None else {"status": "missing", "calls": []}
        digest = hashlib.sha256(raw).hexdigest() if raw is not None else None
        stage_log = output / "recovery" / relative
        if trace["status"] == "ok":
            ids = choose_gpt_candidates(**{**kwargs, "log_path": stage_log}, replay_path=earlier)
            with target.open("xb") as stream:
                stream.write(raw)
            write_json(stage_log.with_suffix(".source.json"), {
                "event": "reused_complete_trace", "archive_trace_sha256": digest, "new_calls": 0,
            })
            return ids
        events = []
        try:
            return choose_candidates(**kwargs, transport=RecoveryTransport(trace, events))
        finally:
            write_json(stage_log.with_suffix(".source.json"), {
                "event": "completed_partial_trace" if raw is not None else "completed_missing_trace",
                "archive_trace_sha256": digest, "events": events,
            })

    return run(output, baseline=baseline, selector=selector, treatment_model=GPT_MODEL,
               freeze_path=protocol, smoke=False, workers=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("evidence/concrete"))
    parser.add_argument("--protocol", type=Path, default=Path("protocol-four-model.md"))
    args = parser.parse_args()
    recover(args.output, args.archive, args.baseline, args.protocol)
