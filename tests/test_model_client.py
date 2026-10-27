"""Tests for the model client.

Every request goes through the real Anthropic SDK, but into a fake network
that records what was sent and returns a canned reply. No API key, network
access or money is needed.
"""

import asyncio
import json

import anthropic
import httpx2
import pytest

from xml_proofreader.findings import Finding
from xml_proofreader.model_client import FALLBACK_BETA, ClaudeProofreader, ProofreadError

TEH = Finding(original="teh", correction="the", type="spelling", reason="Misspelled word.")


def message(text: str | None = '{"findings": []}', stop_reason: str = "end_turn", **extra) -> dict:
    """A reply in the shape the Messages API returns. A text of None means no content at all."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [] if text is None else [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 20,
            "output_tokens": 50,
            "cache_read_input_tokens": 3600,
            "cache_creation_input_tokens": 0,
        },
        **extra,
    }


class FakeAPI:
    """Stands in for the Anthropic API: it records each request and returns one canned reply."""

    def __init__(self, status: int = 200, body: dict | None = None) -> None:
        self.status = status
        self.body = body if body is not None else message()
        self.requests: list[httpx2.Request] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return httpx2.Response(self.status, json=self.body)

    @property
    def last_body(self) -> dict:
        return json.loads(self.requests[-1].content)


def proofread(fake: FakeAPI, texts: list[str], **options) -> tuple[list[list[Finding]], ClaudeProofreader]:
    """Proofread each text through the real SDK, connected to the fake API."""

    async def run() -> tuple[list[list[Finding]], ClaudeProofreader]:
        transport = httpx2.MockTransport(fake.handle)
        http_client = anthropic.DefaultAsyncHttpxClient(transport=transport)
        async with anthropic.AsyncAnthropic(
            api_key="sk-test", http_client=http_client, max_retries=0
        ) as client:
            proofreader = ClaudeProofreader(client, "STYLE GUIDE TEXT", "en", **options)
            results = [await proofreader.proofread(text) for text in texts]
            return results, proofreader

    return asyncio.run(run())


# --- A normal answer


def test_findings_come_back_as_validated_objects():
    fake = FakeAPI(body=message(json.dumps({"findings": [TEH.model_dump()]})))
    results, _ = proofread(fake, ["teh cat"])
    assert results == [[TEH]]


def test_the_request_carries_the_cached_system_prompt_the_paragraph_and_the_schema():
    fake = FakeAPI()
    proofread(fake, ["teh cat"])
    body = fake.last_body
    assert body["model"] == "claude-opus-5-5"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "STYLE GUIDE TEXT" in body["system"][0]["text"]
    assert body["messages"] == [{"role": "user", "content": "<paragraph>teh cat</paragraph>"}]
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "thinking" not in body
    assert "temperature" not in body


def test_server_side_fallbacks_are_requested_with_their_beta_header():
    fake = FakeAPI()
    proofread(fake, ["teh cat"])
    assert fake.last_body["fallbacks"] == "default"
    assert fake.requests[-1].headers["anthropic-beta"] == FALLBACK_BETA


def test_model_and_effort_can_be_changed():
    fake = FakeAPI()
    proofread(fake, ["teh cat"], model="claude-sonnet-5-5", effort="high")
    assert fake.last_body["model"] == "claude-sonnet-5-5"
    assert fake.last_body["output_config"]["effort"] == "high"


def test_the_system_prompt_is_the_same_for_every_paragraph():
    fake = FakeAPI()
    proofread(fake, ["first paragraph", "second paragraph"])
    first, second = (json.loads(request.content) for request in fake.requests)
    assert first["system"] == second["system"]


def test_token_usage_is_added_up_across_calls():
    fake = FakeAPI()
    _, proofreader = proofread(fake, ["one", "two"])
    usage = proofreader.usage
    assert (usage.calls, usage.input_tokens, usage.output_tokens) == (2, 40, 100)
    assert (usage.cache_read_tokens, usage.cache_write_tokens) == (7200, 0)


# --- Problems that affect only this paragraph


def test_a_refusal_is_reported_for_this_paragraph_only():
    details = {"type": "refusal", "category": "cyber", "explanation": "declined"}
    fake = FakeAPI(body=message(None, "refusal", stop_details=details))
    with pytest.raises(ProofreadError, match="declined this paragraph"):
        proofread(fake, ["teh cat"])


def test_an_answer_cut_off_at_the_token_limit_is_reported():
    fake = FakeAPI(body=message('{"findings": [{"original": "te', "max_tokens"))
    with pytest.raises(ProofreadError, match="could not be read as findings"):
        proofread(fake, ["teh cat"])


@pytest.mark.parametrize("status", [429, 500, 529])
def test_errors_the_sdk_retries_are_reported_for_this_paragraph_only(status):
    fake = FakeAPI(status, {"type": "error", "error": {"type": "api_error", "message": "busy"}})
    with pytest.raises(ProofreadError, match=f"kept failing \\({status}\\)"):
        proofread(fake, ["teh cat"])


def test_a_network_failure_is_reported_for_this_paragraph_only():
    def fail(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("no route to host", request=request)

    fake = FakeAPI()
    fake.handle = fail
    with pytest.raises(ProofreadError, match="Could not reach the API"):
        proofread(fake, ["teh cat"])


# --- Problems that would affect every paragraph


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_errors_that_would_hit_every_paragraph_stop_the_run(status):
    fake = FakeAPI(status, {"type": "error", "error": {"type": "invalid_request_error", "message": "no"}})
    with pytest.raises(anthropic.APIStatusError) as raised:
        proofread(fake, ["teh cat"])
    assert raised.value.status_code == status
