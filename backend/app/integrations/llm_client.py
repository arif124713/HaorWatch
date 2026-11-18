import httpx

from app.core.config import Settings


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return (
            self.settings.llm_provider == "gemini"
            and bool(self.settings.gemini_api_key)
            and bool(self.settings.gemini_model)
        )

    def generate(self, prompt: str) -> str | None:
        if not self.enabled:
            return None
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.settings.gemini_model}:generateContent"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
        }
        headers = {"x-goog-api-key": self.settings.gemini_api_key}
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return None
        return sanitize_text(text)


def sanitize_text(text: str) -> str | None:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = cleaned.replace("<", "‹").replace(">", "›")
    cleaned = cleaned.strip()
    return cleaned or None
