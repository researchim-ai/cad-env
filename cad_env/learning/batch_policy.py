"""One VLM instance serving greedy requests from independent CAD episodes."""

import inspect
import queue
import threading
import time
from concurrent.futures import Future

from .policy import HFPolicy


class BatchedHFPolicy(HFPolicy):
    def generate_batch(self, requests):
        import torch
        from PIL import Image

        if not requests:
            return []
        if self.temperature != 0:
            raise ValueError("Batched evaluation currently requires greedy decoding")
        self.tokenizer.padding_side = "left"
        texts, images = [], []
        for messages, paths in requests:
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )
            )
            if self.visual:
                for path in paths:
                    with Image.open(path) as im:
                        images.append(im.convert("RGB"))
        if self.visual:
            inputs = self.processor(
                text=texts, images=images or None, return_tensors="pt", padding=True
            )
        else:
            inputs = self.processor(texts, return_tensors="pt", padding=True)
        width = inputs["input_ids"].shape[-1]
        if width + self.max_new_tokens > self.max_context:
            raise ValueError(
                "Batched context exceeds fixed budget; no truncation allowed"
            )
        if "token_type_ids" not in inspect.signature(self.model.forward).parameters:
            inputs.pop("token_type_ids", None)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.batch_decode(output[:, width:], skip_special_tokens=True)


class PolicyBatcher:
    """Serialize GPU access, microbatch queued requests, propagate every failure."""

    def __init__(self, policy, batch_size=4, wait_ms=40):
        if batch_size < 1 or wait_ms < 0:
            raise ValueError("Positive batch size and nonnegative wait required")
        self.policy = policy
        self.identity = policy.identity
        self.batch_size, self.wait = batch_size, wait_ms / 1000
        self.queue = queue.Queue()
        self._state_lock = threading.Lock()
        self.closed = False
        self.stats = {
            "calls": 0,
            "requests": 0,
            "batch_sizes": [],
            "generation_seconds": 0.0,
        }
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def __call__(self, messages, image_paths=()):
        result = Future()
        with self._state_lock:
            if self.closed:
                raise RuntimeError("Batcher is closed")
            self.queue.put(((messages, image_paths), result))
        return result.result(timeout=600)

    def _run(self):
        while True:
            first = self.queue.get()
            if first is None:
                return
            batch = [first]
            deadline = time.monotonic() + self.wait
            while len(batch) < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self.queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self.queue.put(None)
                    break
                batch.append(item)
            started = time.monotonic()
            try:
                outputs = self.policy.generate_batch([item[0] for item in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError("Policy returned a different batch length")
                for (_, result), output in zip(batch, outputs):
                    result.set_result(output)
            except Exception as exc:  # noqa: BLE001 -- every waiting caller must receive GPU failures
                for _, result in batch:
                    result.set_exception(exc)
            finally:
                self.stats["calls"] += 1
                self.stats["requests"] += len(batch)
                self.stats["batch_sizes"].append(len(batch))
                self.stats["generation_seconds"] += time.monotonic() - started

    def close(self):
        with self._state_lock:
            if not self.closed:
                self.closed = True
                self.queue.put(None)
        self.thread.join()
