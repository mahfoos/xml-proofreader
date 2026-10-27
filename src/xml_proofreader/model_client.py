"""Ask Claude to proofread one paragraph and return validated findings.

Most of each request is identical across a run. The system prompt holds the
instructions and the whole style guide, and it carries a cache marker, so
after the first call the API reads it from cache. Only the short paragraph
message changes from one call to the next.

The answer is constrained to the ParagraphFindings schema, so the SDK returns
Finding objects instead of text that would need parsing by hand.
"""

from dataclasses import dataclass
from typing import Literal

import anthropic
import pydantic
from anthropic.types.beta import BetaTextBlockParam, BetaUsage

from .findings import Finding, ParagraphFindings
from .prompt import build_system_prompt, build_user_message

Effort = Literal["low", "medium", "high", "xhigh", "max"]

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT: Effort = "high"
MAX_TOKENS = 16000
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ProofreadError(Exception):
    """One paragraph could not be proofread. The rest of the run can continue."""


@dataclass
class TokenUsage:
    """Running totals of the tokens used, for the end-of-run summary.

    Attributes:
        calls: How many replies the API returned.
        input_tokens: Input tokens billed at the normal price.
        output_tokens: Output tokens, including the model's thinking.
        cache_read_tokens: Input tokens read from the prompt cache.
        cache_write_tokens: Input tokens written to the prompt cache.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, usage: BetaUsage) -> None:
        """Add one reply's token counts to the totals.

        Args:
            usage: The usage block from one API reply. The cache counts can
                be missing, so a missing count is added as 0.
        """
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
        effort: Effort = DEFAULT_EFFORT,
    ) -> None:
        """Build the system prompt once, for every paragraph of the run.

        Args:
            client: An open API client. The caller controls its lifetime, and
                tests pass one connected to a fake network.
            style_guide: The style guide text, from load_style_guide.
            language: The BCP 47 language tag from the command line.
            model: The Claude model to call.
            effort: How much thinking the model may do.
        """
        self.client = client
        self.model = model
        self.effort: Effort = effort
        self.system: list[BetaTextBlockParam] = [
            {
                "type": "text",
                "text": build_system_prompt(style_guide, language),
                "cache_control": {"type": "ephemeral"},
            }
        ]
        self.usage = TokenUsage()

    async def proofread(self, text: str) -> list[Finding]:
        """Proofread one paragraph's text.

        Args:
            text: The paragraph's text, exactly as the locator found it.

        Returns:
            The model's findings for the paragraph, already validated
            against the schema.

        Raises:
            ProofreadError: If only this paragraph is affected: the API kept
                failing after the SDK's own retries, the model declined, or
                its answer was cut off.
            anthropic.APIStatusError: For any other API error, such as a bad
                key or an unknown model. It would fail for every paragraph,
                so it is not caught and stops the run.
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
            raise ProofreadError(
                f"The API kept failing ({error.status_code}): {describe_api_error(error)}"
            ) from error
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


def describe_api_error(error: anthropic.APIStatusError) -> str:
    """Extract the API's own explanation from an error.

    Args:
        error: An error the SDK raised for an HTTP error status.

    Returns:
        The message from the API's error body, or the SDK's own text when
        the body is not in the usual shape, for example from a proxy.
    """
    body = error.body
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return str(body["error"].get("message", error.message))
    return error.message


def _is_transient(error: anthropic.APIStatusError) -> bool:
    """Decide whether an error status is temporary.

    Args:
        error: An error the SDK raised for an HTTP error status.

    Returns:
        True for the statuses the SDK retries itself: 408, 409, 429 and
        every 5xx.
    """
    return error.status_code in (408, 409, 429) or error.status_code >= 500
