from typing import List

from commonlib.task_queue import PikaConnection

pika_connection = None


def init_pika_connection(
    broker_url: str, queues: List[str], max_queue_length: int
) -> PikaConnection:
    """Initialize RabbitMQ (pika) connection and store it as a global module variable."""
    global pika_connection
    if pika_connection is None:
        pika_connection = PikaConnection(broker_url=broker_url)
        for queue_name in queues:
            pika_connection.declare_queue(queue_name, max_queue_length)
    return get_pika_connection()


def get_pika_connection() -> PikaConnection:
    """Get RabbitMQ (pika) connection."""
    assert pika_connection is not None, "Pika connection was not initialized."
    return pika_connection


def close_pika_connection():
    """Close RabbitMQ (pika) connection."""
    if pika_connection is not None:
        pika_connection.close()
