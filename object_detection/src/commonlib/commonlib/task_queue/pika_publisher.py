import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

import pika
from pika.exceptions import StreamLostError

from commonlib.error_handling import async_retry

from .pika_connection import PikaConnection

logger = logging.getLogger("app")


class NoConsumerException(Exception):
    """Exception raised when no RabbitMQ consumer is available."""


class WorkerTimeoutException(Exception):
    """Exception raised when worker takes too much time to process the request."""


@async_retry(
    num_retries=3,
    exceptions=(StreamLostError,),
    delay=0.5,
    retry_message="Failed to reconnect to the RabbitMQ broker. Retrying...",
    fail_message="Failed to reconnect to the RabbitMQ broker",
)
def send_message(
    pika_connection: PikaConnection, queue_name: str, msg: dict, message_ttl_seconds: int = 60
):
    """Send message to RabbitMQ queue.

    :param pika_connection: Pika connection instance.
    :param queue_name: Name of the message queue.
    :param message_ttl_seconds: Message time to live (in seconds).
    :param msg: A dictionary with a message to send.
    """
    # check pika connection and reconnect if needed
    pika_connection.check_and_reconnect()
    channel = pika_connection.channel
    queue_state = channel.queue_declare(queue=queue_name, passive=True)

    # check if any consumer is available
    logger.debug(f"Number of available workers: {queue_state.method.consumer_count}")
    if queue_state.method.consumer_count == 0:
        logger.warning("Inference worker is not available.")
        raise NoConsumerException

    # send message
    msg_str = json.dumps(msg, ensure_ascii=True)
    logger.debug(f"Sending message to RabbitMQ broker: {msg_str}")
    channel.basic_publish(
        exchange="",
        routing_key=queue_name,
        properties=pika.BasicProperties(
            content_type="application/json",
            # convert seconds to milliseconds
            expiration=str(message_ttl_seconds * 1000),
        ),
        body=bytes(msg_str, "utf-8"),
    )


class RpcPublisher:
    """Message publisher that implements Remote Procedure Call (RPC) pattern.

    The RPC publisher sends a message and awaits a result from the consumer.
    Apart from the main queue, the publisher creates a callback queue
    exclusive to the connection channel.
    """

    def __init__(self, pika_connection: PikaConnection):
        self.request_id = f"request_id={datetime.now().timestamp()}"
        self.pika_connection = pika_connection
        self.response = None

    @async_retry(
        num_retries=3,
        exceptions=(StreamLostError,),
        delay=0.5,
        retry_message="Failed to reconnect to the RabbitMQ broker. Retrying...",
        fail_message="Failed to reconnect to the RabbitMQ broker",
    )
    def establish_callback_queue(self, queue_name: str):
        """Reconnect to RabbitMQ broker if needed and create a callback queue.

        The method uses @async_retry decorator to catch `StreamLostError`.
        This is needed because `check_and_reconnect` method may not detect
        that the connection is lost.
        """
        # check pika connection and reconnect if needed
        self.pika_connection.check_and_reconnect()
        connection = self.pika_connection.connection
        channel = self.pika_connection.channel
        queue_state = channel.queue_declare(queue=queue_name, passive=True)

        # create a response queue
        callback_queue_state = channel.queue_declare(queue="", exclusive=True)
        callback_queue_name = callback_queue_state.method.queue
        channel.basic_consume(
            queue=callback_queue_name,
            on_message_callback=self.on_response,
        )

        return connection, channel, queue_state, callback_queue_name

    def on_response(
        self,
        channel: pika.adapters.blocking_connection.BlockingChannel,
        method: pika.spec.Basic.Deliver,
        props: pika.spec.BasicProperties,
        body: bytes,
    ):
        """A callback function that stores response from the consumer.

        The response is returned by `send_message` function.
        """
        if self.request_id == props.correlation_id:
            self.response = json.loads(body)
        channel.basic_ack(delivery_tag=method.delivery_tag)

    async def _await_response(self, connection) -> Optional[dict]:
        """Wait for a response from the consumer.

        :return: A message received from the consumer.
        """
        while self.response is None:
            connection.process_data_events()
            # allow asyncio to run other tasks in the event loop
            await asyncio.sleep(0.5)
        return self.response

    async def send_message(
        self, queue_name: str, msg: dict, timeout_seconds: float, message_ttl_seconds: int = 60
    ) -> dict:
        """Send a message to the consumer and awaits for the response message.

        :param queue_name: Name of the message queue.
        :param msg: A message send to the consumer.
        :param timeout_seconds: Time limit to wait for a response (in seconds).
        :param message_ttl_seconds: Message time to live (in seconds).
        :return: A message received from the consumer.
        """
        # check pika connection and reconnect if needed
        (
            connection,
            channel,
            queue_state,
            callback_queue_name,
        ) = await self.establish_callback_queue(queue_name)

        # check if any consumer is available
        logger.debug(f"Number of available workers: {queue_state.method.consumer_count}")
        if queue_state.method.consumer_count == 0:
            logger.warning("Inference worker is not available.")
            raise NoConsumerException

        # send message
        msg_str = json.dumps(msg, ensure_ascii=True)
        logger.debug(f"Sending message to RabbitMQ broker: {msg_str}")
        channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            properties=pika.BasicProperties(
                content_type="application/json",
                reply_to=callback_queue_name,
                correlation_id=self.request_id,
                # convert seconds to milliseconds
                expiration=str(message_ttl_seconds * 1000),
            ),
            body=bytes(msg_str, "utf-8"),
        )

        # await response
        try:
            response = await asyncio.wait_for(
                self._await_response(connection), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.warning("Inference worker timed out.")
            raise WorkerTimeoutException

        return response
