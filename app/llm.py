import threading

import ollama


DEFAULT_MODEL = "qwen3-coder:latest"


class LLMTimeoutError(TimeoutError):
    """Structured failure raised when one LLM call exceeds its deadline."""

    def __init__(self, model, timeout, stage):
        self.model = model
        self.timeout = timeout
        self.stage = stage
        super().__init__(
            f"{stage} LLM call timed out after {timeout:g}s (model={model})"
        )


def call_llm(
    prompt,
    system_prompt="You are an AI Technical Analyst. You break down complex technical concepts into structured, easy-to-understand explanations.",
    model=DEFAULT_MODEL,
    format=None,
    timeout=None,
    stage="llm",
):
    result = {}
    failure = {}

    def run_chat():
        try:
            chat = (
                ollama.Client(timeout=timeout).chat
                if timeout is not None
                else ollama.chat
            )
            result["response"] = chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                **({'format': format} if format else {}),
            )
        except BaseException as error:
            failure["error"] = error

    worker = threading.Thread(target=run_chat, daemon=True)
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        raise LLMTimeoutError(model, timeout, stage)
    if "error" in failure:
        raise failure["error"]

    return result["response"]["message"]["content"]