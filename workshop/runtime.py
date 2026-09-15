"""Pinned local model and optional, explicitly selected remote reference."""

import os
import time

from workshop.domain import messages_for
from workshop.store import MODEL_ID, MODEL_REVISION


class LocalModel:
    def __init__(self, adapter_path=None):
        from huggingface_hub import snapshot_download
        from mlx_lm import load
        path = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
        self.model, self.tokenizer = load(path, adapter_path=adapter_path)

    def predict(self, skill, text):
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        mx.random.seed(17)
        prompt = self.tokenizer.apply_chat_template(messages_for(skill, text), tokenize=False,
                                                   add_generation_prompt=True)
        tokens = len(self.tokenizer.encode(prompt))
        if tokens > 6000:
            raise ValueError("Input exceeds the local 6000-token limit; provide a smaller excerpt.")
        start = time.perf_counter()
        raw = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=220,
                       sampler=make_sampler(temp=0), verbose=False)
        return raw, round((time.perf_counter() - start) * 1000), tokens


class RemoteModel:
    def __init__(self):
        self.url = os.environ.get("WORKSHOP_REFERENCE_URL", "")
        self.model = os.environ.get("WORKSHOP_REFERENCE_MODEL", "")
        self.key = os.environ.get("WORKSHOP_REFERENCE_KEY", "")
        if not self.url or not self.model:
            raise ValueError("Reference provider is not configured.")

    def predict(self, skill, text):
        import httpx
        start = time.perf_counter()
        with httpx.Client(timeout=90) as client:
            response = client.post(self.url.rstrip("/") + "/chat/completions",
                                   headers={"Authorization": f"Bearer {self.key}"},
                                   json=dict(model=self.model, messages=messages_for(skill, text),
                                             temperature=0, max_tokens=220))
            response.raise_for_status()
            result = response.json()
        return (result["choices"][0]["message"]["content"],
                round((time.perf_counter() - start) * 1000),
                result.get("usage", {}).get("prompt_tokens"))
