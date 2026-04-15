"""Language model wrappers following the Concordia LanguageModel interface.

Concordia defines an abstract LanguageModel with two methods:
    sample_text(prompt, ...) -> str
    sample_choice(prompt, choices, ...) -> (index, choice, metadata)

Implementations:
    AnthropicLanguageModel — wraps the Anthropic Messages API
    OpenAILanguageModel    — wraps the OpenAI Chat Completions API

Use make_language_model() to instantiate by provider name.

Note: Concordia requires Python >=3.12. This module is designed to work with
Python 3.10+ so the full CCSM pipeline can run without the Concordia dependency.
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional


class LanguageModel:
    """Abstract base matching the Concordia LanguageModel interface."""

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        seed: Optional[int] = None,
        terminators: Optional[list[str]] = None,
    ) -> str:
        raise NotImplementedError

    def sample_choice(
        self,
        prompt: str,
        choices: list[str],
        *,
        seed: Optional[int] = None,
    ) -> tuple[int, str, dict]:
        """Return (index, chosen_text, metadata)."""
        raise NotImplementedError


class AnthropicLanguageModel(LanguageModel):
    """Concordia-compatible wrapper around the Anthropic Messages API.

    Args:
        model: Anthropic model ID, e.g. "claude-sonnet-4-20250514".
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        temperature: Default sampling temperature (overridden per call).
        max_retries: Number of times to retry on transient API errors.
        retry_delay: Seconds to wait between retries.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        import anthropic  # imported here so the module can be imported without anthropic
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self._call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        terminators: Optional[list[str]] = None,
    ) -> str:
        """Generate text from a prompt string."""
        temp = temperature if temperature is not None else self._temperature
        stop_seqs = terminators or []

        for attempt in range(self._max_retries):
            try:
                kwargs: dict = dict(
                    model=self._model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                )
                if stop_seqs:
                    kwargs["stop_sequences"] = stop_seqs

                response = self._client.messages.create(**kwargs)
                self._call_count += 1
                self._total_input_tokens += response.usage.input_tokens
                self._total_output_tokens += response.usage.output_tokens
                return response.content[0].text

            except Exception as exc:
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (2 ** attempt))
                else:
                    raise RuntimeError(
                        f"Anthropic API call failed after {self._max_retries} attempts: {exc}"
                    ) from exc

        return ""  # unreachable

    def sample_choice(
        self,
        prompt: str,
        choices: list[str],
        *,
        seed: Optional[int] = None,
    ) -> tuple[int, str, dict]:
        """Ask the model to pick from a list of choices.

        Appends a formatted choice list to the prompt, then parses the response
        to identify which option was selected. Falls back to choice 0 on failure.
        """
        choices_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
        full_prompt = (
            f"{prompt}\n\n"
            f"Please select one of the following options by responding with "
            f"only the number:\n{choices_text}"
        )

        response = self.sample_text(
            full_prompt, max_tokens=64, temperature=0.0, seed=seed
        )

        # Parse: look for a digit 1-N
        match = re.search(r"\b([1-9]\d*)\b", response.strip())
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(choices):
                return idx, choices[idx], {"raw_response": response}

        # Fallback: check if any choice text appears in the response
        for i, choice in enumerate(choices):
            if choice.lower() in response.lower():
                return i, choice, {"raw_response": response}

        return 0, choices[0], {"raw_response": response, "parse_failed": True}

    def stats(self) -> dict:
        return {
            "call_count": self._call_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }


class OpenAILanguageModel(LanguageModel):
    """Concordia-compatible wrapper around the OpenAI Chat Completions API.

    Args:
        model: OpenAI model ID, e.g. "gpt-4o" or "gpt-4o-mini".
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        base_url: Optional base URL override (for Azure or compatible endpoints).
        temperature: Default sampling temperature (overridden per call).
        max_retries: Number of times to retry on transient API errors.
        retry_delay: Seconds to wait between retries.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        import openai  # imported here so the module can be imported without openai
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        kwargs: dict = {"api_key": api_key or os.environ.get("OPENAI_API_KEY", "")}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)
        self._call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        terminators: Optional[list[str]] = None,
    ) -> str:
        """Generate text from a prompt string."""
        temp = temperature if temperature is not None else self._temperature
        stop_seqs = terminators or None

        # Newer OpenAI models (o-series, gpt-5.x) use max_completion_tokens
        _use_completion_tokens = any(
            self._model.startswith(p) for p in ("o1", "o3", "o4", "gpt-5")
        )
        tokens_key = "max_completion_tokens" if _use_completion_tokens else "max_tokens"

        for attempt in range(self._max_retries):
            try:
                kwargs: dict = dict(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    **{tokens_key: max_tokens},
                )
                # o-series models don't support temperature
                if not _use_completion_tokens:
                    kwargs["temperature"] = temp
                if stop_seqs:
                    kwargs["stop"] = stop_seqs
                if seed is not None:
                    kwargs["seed"] = seed

                response = self._client.chat.completions.create(**kwargs)
                self._call_count += 1
                if response.usage:
                    self._total_input_tokens += response.usage.prompt_tokens
                    self._total_output_tokens += response.usage.completion_tokens
                return response.choices[0].message.content or ""

            except Exception as exc:
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (2 ** attempt))
                else:
                    raise RuntimeError(
                        f"OpenAI API call failed after {self._max_retries} attempts: {exc}"
                    ) from exc

        return ""  # unreachable

    def sample_choice(
        self,
        prompt: str,
        choices: list[str],
        *,
        seed: Optional[int] = None,
    ) -> tuple[int, str, dict]:
        """Ask the model to pick from a list of choices."""
        choices_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
        full_prompt = (
            f"{prompt}\n\n"
            f"Please select one of the following options by responding with "
            f"only the number:\n{choices_text}"
        )

        response = self.sample_text(
            full_prompt, max_tokens=64, temperature=0.0, seed=seed
        )

        match = re.search(r"\b([1-9]\d*)\b", response.strip())
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(choices):
                return idx, choices[idx], {"raw_response": response}

        for i, choice in enumerate(choices):
            if choice.lower() in response.lower():
                return i, choice, {"raw_response": response}

        return 0, choices[0], {"raw_response": response, "parse_failed": True}

    def stats(self) -> dict:
        return {
            "call_count": self._call_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }


def make_language_model(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs,
) -> LanguageModel:
    """Factory: create a LanguageModel by provider name.

    Args:
        provider: "anthropic" or "openai".
        model: Model ID. Defaults to "claude-sonnet-4-20250514" for Anthropic,
               "gpt-4o" for OpenAI.
        api_key: API key. Falls back to ANTHROPIC_API_KEY / OPENAI_API_KEY env vars.
        temperature: Default sampling temperature.
        **kwargs: Extra kwargs passed to the underlying model class
                  (e.g. base_url for OpenAI).

    Returns:
        An AnthropicLanguageModel or OpenAILanguageModel instance.
    """
    if provider == "anthropic":
        return AnthropicLanguageModel(
            model=model or "claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=temperature,
            **kwargs,
        )
    elif provider == "openai":
        return OpenAILanguageModel(
            model=model or "gpt-4o",
            api_key=api_key,
            temperature=temperature,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown provider {provider!r}. Must be 'anthropic' or 'openai'."
        )
