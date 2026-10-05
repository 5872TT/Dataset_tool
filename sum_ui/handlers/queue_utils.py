"""Small helpers for bounded, latest-value UI progress queues."""
import queue


def put_latest(target: queue.Queue, item) -> None:
    """Keep only the newest UI event when the consumer is temporarily slower."""
    try:
        target.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        target.get_nowait()
    except queue.Empty:
        pass
    try:
        target.put_nowait(item)
    except queue.Full:
        pass
