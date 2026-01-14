"""Token counter utility for tracking LLM usage.

Wraps litellm to track token counts and costs across all LLM calls.
"""

from dataclasses import dataclass, field
from typing import Optional
import litellm


@dataclass
class UsageStats:
    """Statistics for a single LLM call."""
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float  # USD


@dataclass
class TokenCounter:
    """Track token usage and costs across multiple LLM calls."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    call_count: int = 0
    calls: list[UsageStats] = field(default_factory=list)

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: Optional[float] = None,
    ) -> UsageStats:
        """
        Record a single LLM call.

        Args:
            model: Model name (e.g., "gpt-4o", "claude-3-opus")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Cost in USD (if None, estimated from litellm)

        Returns:
            UsageStats for this call
        """
        total_tokens = input_tokens + output_tokens

        # Estimate cost if not provided
        if cost is None:
            cost = self._estimate_cost(model, input_tokens, output_tokens)

        stats = UsageStats(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
        )

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.call_count += 1
        self.calls.append(stats)

        return stats

    def record_response(self, response) -> UsageStats:
        """
        Record usage from a litellm response object.

        Args:
            response: Response from litellm.completion()

        Returns:
            UsageStats for this call
        """
        usage = response.usage
        model = response.model

        return self.record(
            model=model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost=getattr(response, "_hidden_params", {}).get("response_cost"),
        )

    def _estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost based on model pricing."""
        try:
            cost = litellm.completion_cost(
                model=model,
                prompt="x" * input_tokens,  # Dummy prompt
                completion="x" * output_tokens,  # Dummy completion
            )
            return cost
        except Exception:
            # Fallback to rough estimate if litellm can't calculate
            # Using approximate GPT-4 pricing as default
            input_cost = input_tokens * 0.00003  # $0.03 per 1K
            output_cost = output_tokens * 0.00006  # $0.06 per 1K
            return input_cost + output_cost

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    def summary(self) -> str:
        """Return a summary string of usage."""
        return (
            f"Calls: {self.call_count} | "
            f"Tokens: {self.total_tokens:,} "
            f"(in: {self.total_input_tokens:,}, out: {self.total_output_tokens:,}) | "
            f"Cost: ${self.total_cost:.4f}"
        )

    def reset(self) -> None:
        """Reset all counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
        self.calls = []


# Global counter for convenience
_global_counter: Optional[TokenCounter] = None


def get_global_counter() -> TokenCounter:
    """Get or create the global token counter."""
    global _global_counter
    if _global_counter is None:
        _global_counter = TokenCounter()
    return _global_counter


def reset_global_counter() -> None:
    """Reset the global token counter."""
    global _global_counter
    _global_counter = TokenCounter()
