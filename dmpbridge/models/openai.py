"""OpenAI model backend."""
from ..exceptions import ConfigurationError
from ..logging_setup import get_logger

logger = get_logger(__name__)


class OpenAIModel:
    """Call the OpenAI chat completions API with ``json_object`` output mode.

    The system prompt is automatically extended to instruct the model to wrap
    its label array in a JSON object — a requirement of OpenAI's
    ``response_format={"type": "json_object"}`` mode.

    Parameters
    ----------
    model:
        OpenAI model ID, e.g. ``"gpt-4o"`` or ``"gpt-4o-mini"``.
    api_key:
        OpenAI API key.
    """

    def __init__(self, model: str, api_key: str) -> None:
        if not api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is not set.\n"
                "Add it to your .env file:  OPENAI_API_KEY=sk-..."
            )
        try:
            import openai as _openai
        except ImportError:
            raise ImportError(
                "The openai package is not installed.\n"
                "Install it with:  pip install openai"
            )
        self.model   = model
        self._client = _openai.OpenAI(api_key=api_key)

    def complete(self, system: str, prompt: str) -> str:
        """Send *system* + *prompt* to OpenAI and return the raw text response."""
        # json_object mode requires the system prompt to explicitly ask for JSON.
        system_json = system + '\n\nWrap your array in a JSON object: {"labels": [...]}'
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_json},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return resp.choices[0].message.content or ""
