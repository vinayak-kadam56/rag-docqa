"""Answer generation. Pick a provider with LLM_PROVIDER:

  none       (default) no LLM - the app returns the best matching passages
  anthropic  Claude via the Anthropic API (set ANTHROPIC_API_KEY, optional LLM_MODEL)
  ollama     a local model served by Ollama (LLM_MODEL, OLLAMA_HOST)
"""
import json
import os
import urllib.request

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided by the user. "
    "If the passages do not contain the answer, say you could not find it in the documents. "
    "Be concise and cite the passages you used like [1] or [2]."
)


class LLMError(Exception):
    """Raised when the language-model call fails."""


def build_user_prompt(question: str, passages: list[dict]) -> str:
    blocks = [f"[{i}] (source: {p['filename']})\n{p['text']}" for i, p in enumerate(passages, 1)]
    return "Context:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


class AnthropicLLM:
    name = "anthropic"

    def __init__(self):
        import anthropic  # optional dependency
        self.client = anthropic.Anthropic()               # reads ANTHROPIC_API_KEY
        self.model = os.getenv("LLM_MODEL") or "claude-sonnet-5-5"

    def generate(self, system: str, user: str) -> str:
        try:
            msg = self.client.messages.create(
                model=self.model, max_tokens=600, system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        except Exception as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc


class OllamaLLM:
    name = "ollama"

    def __init__(self):
        self.model = os.getenv("LLM_MODEL") or "llama3.2"
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

    def generate(self, system: str, user: str) -> str:
        body = json.dumps({
            "model": self.model, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())["message"]["content"].strip()
        except Exception as exc:
            raise LLMError(f"Ollama error (is it running at {self.host}?): {exc}") from exc


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "none").lower()
    if provider == "anthropic":
        return AnthropicLLM()
    if provider == "ollama":
        return OllamaLLM()
    return None
