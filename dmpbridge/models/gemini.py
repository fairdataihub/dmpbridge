"""Google Gemini model backend."""
from ..exceptions import ConfigurationError
from ..logging_setup import get_logger

logger = get_logger(__name__)


class GeminiModel:
    """Call the Google Gemini API with JSON mime-type enforcement.

    Parameters
    ----------
    model:
        Gemini model ID, e.g. ``"gemini-2.0-flash"`` or ``"gemini-1.5-pro"``.
    api_key:
        Google AI API key.
    """

    def __init__(self, model: str, api_key: str) -> None:
        if not api_key:
            raise ConfigurationError(
                "GEMINI_API_KEY is not set.\n"
                "Add it to your .env file:  GEMINI_API_KEY=AIza..."
            )
        try:
            import google.generativeai as _genai
        except ImportError:
            raise ImportError(
                "The google-generativeai package is not installed.\n"
                "Install it with:  pip install google-generativeai"
            )
        _genai.configure(api_key=api_key)
        self._genai     = _genai
        self.model_name = model

    def complete(self, system: str, prompt: str) -> str:
        """Send *system* + *prompt* to Gemini and return the raw text response."""
        model = self._genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system,
            generation_config=self._genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )
        resp = model.generate_content(prompt)
        return resp.text or ""
