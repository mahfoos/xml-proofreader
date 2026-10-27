"""Ask Claude to proofread one paragraph and return validated findings.

Most of each request is identical across a run. The system prompt holds the
instructions and the whole style guide, and it carries a cache marker, so
after the first call the API reads it from cache. Only the short paragraph
message changes from one call to the next.

The answer is constrained to the ParagraphFindings schema, so the SDK returns
Finding objects instead of text that would need parsing by hand.
"""

from dataclasses import dataclass

import anthropic
import pydantic
from anthropic.types.beta import BetaUsage

from .findings import Finding, ParagraphFindings
from .prompt import build_system_prompt, build_user_message

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
MAX_TOKENS = 16000
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ProofreadError(Exception):
    """One paragraph could not be proofread. The rest of the run can continue."""


@dataclass
class TokenUsage:
    """Running totals of the tokens used, for the end-of-run summary."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, usage: BetaUsage) -> None:
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += usage.cache_read_input_tokens or 0
        self.cache_write_tokens += usage.cache_creation_input_tokens or 0


class ClaudeProofreader:
    """Proofreads paragraphs with Claude, using one fixed system prompt for the whole run."""

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        style_guide: str,
        language: str,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
    ) -> None:
        self.client = client
        self.model = model
        self.effort = effort
        self.system = [
            {
                "type": "text",
                "text": build_system_prompt(style_guide, language),
                "cache_control": {"type": "ephemeral"},
            }
        ]
        self.usage = TokenUsage()

    async def proofread(self, text: str) -> list[Finding]:
        """Return the findings for one paragraph's text.

        Raises ProofreadError when only this paragraph is affected: the API kept
        failing after the SDK's own retries, the model declined, or its answer
        was cut off. Any other API error means every call would fail, such as a
        bad key or an unknown model, so it is not caught and stops the run.
        """
        try:
            response = await self.client.beta.messages.parse(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=self.system,
                messages=[{"role": "user", "content": build_user_message(text)}],
                output_format=ParagraphFindings,
                output_config={"effort": self.effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.APIStatusError as error:
            if not _is_transient(error):
                raise
            raise ProofreadError(f"The API kept failing ({error.status_code}): {error.message}") from error
        except anthropic.APIConnectionError as error:
            raise ProofreadError(f"Could not reach the API: {error}") from error
        except pydantic.ValidationError as error:
            raise ProofreadError(
                "The answer could not be read as findings, so it was probably cut off or declined."
            ) from error

        self.usage.add(response.usage)
        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            raise ProofreadError(f"The model declined this paragraph (category: {category}).")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise ProofreadError(f"No usable answer (stop reason: {response.stop_reason}).")
        return response.parsed_output.findings


def _is_transient(error: anthropic.APIStatusError) -> bool:
    """True for the statuses the SDK retries itself: 408, 409, 429 and every 5xx."""
    return error.status_code in (408, 409, 429) or error.status_code >= 500
