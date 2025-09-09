import pytest

from app.utils.resilience.dead_letter.queue import DeadLetterQueue


class FakeRedis:
    def __init__(self):
        self.lists = {}

    async def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)

    async def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    async def rpop(self, key):
        lst = self.lists.get(key) or []
        if not lst:
            return None
        return lst.pop()


pytestmark = pytest.mark.asyncio


async def test_dlq_send_and_process():
    fake = FakeRedis()
    dlq = DeadLetterQueue(redis_client=fake, max_retries=2)

    msg = {"audit_run_id": "abc123"}
    await dlq.send_to_dlq(msg, original_queue="audit:tasks", error=RuntimeError("boom"))

    processed_msgs = []

    async def processor(payload):
        processed_msgs.append(payload)
        return True

    count = await dlq.process_dlq_messages("audit:tasks", processor, max_messages=10)
    assert count == 1
    assert processed_msgs[0]["audit_run_id"] == "abc123"
