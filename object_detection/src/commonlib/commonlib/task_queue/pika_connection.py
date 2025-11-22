import logging

import pika
from pika.exceptions import AMQPConnectionError

from commonlib.error_handling import retry

logger = logging.getLogger("app")


class PikaConnection:
    """Wrapper that establishes `pika.BlockingConnection` and channel."""

    def __init__(self, broker_url: str):
        self.broker_url = broker_url
        self.connection = None
        self.channel = None
        self.connect()

    def declare_queue(self, queue_name: str, max_queue_length: int = 1000):
        """Create queue if it does not exist."""
        assert self.channel is not None, "Pika connection was not initialized."
        self.channel.queue_declare(
            queue=queue_name,
            durable=True,
            arguments={
                "x-max-length": max_queue_length,
                "x-overflow": "reject-publish",
            },
        )

    @retry(
        num_retries=5,
        exceptions=(AMQPConnectionError,),
        delay=10.0,
        init_message="Creating pika connection.",
        retry_message="Failed to connect to the RabbitMQ broker. Retrying...",
        fail_message="Failed to connect to the RabbitMQ broker",
    )
    def connect(self):
        """Establish connection to RabbitMQ broker."""
        params = pika.URLParameters(self.broker_url)
        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()

    def check_and_reconnect(self):
        """Check if the connection is open, if not establish a new connection."""
        if self.connection.is_closed:
            self.connect()

    def close(self):
        """Close pika connection to RabbitMQ."""
        logger.info("Closing pika connection.")
        if self.channel:
            self.channel.stop_consuming()
        if self.connection:
            self.connection.close()
