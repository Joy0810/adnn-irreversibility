"""Load a model, and turn chat messages into text, token ids, greedy replies and log-probabilities.

GPU machine only. Never load two models at once.
"""
from __future__ import annotations

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
}


def _dtype_kwarg() -> str:
    # transformers renamed `torch_dtype` to `dtype` in 4.56; older versions would
    # silently ignore `dtype` and load in float32, which does not fit.
    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def load(key: str, device_map: str = "auto"):
    name = MODELS.get(key, key)
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, device_map=device_map, **{_dtype_kwarg(): torch.bfloat16})
    model.eval()
    return model, tokenizer


def chat_text(tokenizer, messages: list, add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def _ids(tokenizer, text: str) -> list:
    # The chat template already contains any BOS token, so add no special tokens.
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def generate(model, tokenizer, messages: list, max_new_tokens: int = 192) -> str:
    """Greedy reply to `messages`."""
    ids = torch.tensor([_ids(tokenizer, chat_text(tokenizer, messages, True))], device=model.device)
    # Qwen's generation_config sets repetition_penalty 1.05, which applies even without
    # sampling; override it so decoding is truly greedy for every model.
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                         do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                         pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def make_generate(model, tokenizer, max_new_tokens: int = 192):
    return lambda messages: generate(model, tokenizer, messages, max_new_tokens)


@torch.no_grad()
def continuation_logprob(model, tokenizer, messages: list, continuation: str) -> float:
    """Total log-probability of `continuation` as the start of the assistant's reply."""
    prompt_ids = _ids(tokenizer, chat_text(tokenizer, messages, True))
    full_ids = _ids(tokenizer, chat_text(tokenizer, messages, True) + continuation)
    if full_ids[:len(prompt_ids)] != prompt_ids:  # a token merged across the boundary
        full_ids = prompt_ids + _ids(tokenizer, continuation)
    ids = torch.tensor([full_ids], device=model.device)
    n = len(prompt_ids)
    logprobs = torch.log_softmax(model(ids).logits[0, n - 1:-1].float(), dim=-1)
    return logprobs.gather(1, ids[0, n:, None]).sum().item()


def versions() -> dict:
    import numpy, sklearn
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": numpy.__version__, "sklearn": sklearn.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
