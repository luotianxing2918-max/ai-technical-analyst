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


CUDA_FAILURE_MARKERS = (
    "cuda error",
    "shared object initialization failed",
    "llama-server process has terminated",
    "0xc0000409",
)


def _is_cuda_failure(error):
    message = str(error).casefold()
    if any(marker in message for marker in CUDA_FAILURE_MARKERS):
        return True
    return "status code: 500" in message and any(
        marker in message
        for marker in ("cuda", "llama-server", "shared object", "ollama")
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
            chat_options = {"format": format} if format else {}
            if stage == "final":
                chat_options["options"] = {"num_ctx": 2048}
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            try:
                result["response"] = chat(
                    model=model,
                    messages=messages,
                    **chat_options,
                )
            except BaseException as error:
                if stage not in {"router", "final"} or not _is_cuda_failure(error):
                    raise
                fallback_options = {"options": {"num_gpu": 0}}
                if stage == "final":
                    fallback_options["options"]["num_ctx"] = 2048
                if format:
                    fallback_options["format"] = format
                result["response"] = chat(
                    model=model,
                    messages=messages,
                    **fallback_options,
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