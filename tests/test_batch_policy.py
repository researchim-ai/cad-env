from concurrent.futures import ThreadPoolExecutor

import pytest

from cad_env.learning.batch_policy import PolicyBatcher


class FakePolicy:
    def __init__(self):
        self.identity = {"model": "fake"}

    def generate_batch(self, requests):
        return [str(messages) for messages, images in requests]


def test_batcher_preserves_routing_and_collects_requests():
    batcher = PolicyBatcher(FakePolicy(), batch_size=4, wait_ms=100)
    try:
        with ThreadPoolExecutor(4) as pool:
            values = list(pool.map(lambda i: batcher(i), range(4)))
        assert values == ["0", "1", "2", "3"]
    finally:
        batcher.close()
    assert batcher.stats["requests"] == 4
    assert max(batcher.stats["batch_sizes"]) == 4
    with pytest.raises(RuntimeError, match="closed"):
        batcher(1)


def test_batcher_propagates_failure_to_all_waiters_and_survives():
    class Broken(FakePolicy):
        def generate_batch(self, requests):
            raise ValueError("failed generation")

    batcher = PolicyBatcher(Broken(), batch_size=2, wait_ms=100)
    try:
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(batcher, i) for i in range(2)]
            for f in futures:
                with pytest.raises(ValueError, match="failed generation"):
                    f.result(timeout=2)
    finally:
        batcher.close()


def test_close_drains_accepted_requests_and_is_idempotent():
    import threading

    entered, release = threading.Event(), threading.Event()

    class Blocking(FakePolicy):
        def generate_batch(self, requests):
            entered.set()
            assert release.wait(2)
            return super().generate_batch(requests)

    batcher = PolicyBatcher(Blocking(), batch_size=1, wait_ms=0)
    with ThreadPoolExecutor(2) as pool:
        pending = pool.submit(batcher, "accepted")
        assert entered.wait(2)
        closer = pool.submit(batcher.close)
        release.set()
        assert pending.result(timeout=2) == "accepted"
        closer.result(timeout=2)
    batcher.close()
