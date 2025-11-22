import json
import logging
import os
import time
import traceback

import numpy as np
import pika
from pika.exceptions import AMQPChannelError, AMQPConnectionError, ConnectionClosedByBroker
from worker.georegistration import GeoRegTracker

from commonlib.log import RequestLogger, setup_logging
from commonlib.task_queue import PikaConnection
from commonlib.task_queue.models import (
    TrackingWorkerRequest,
    TrackingWorkerResponse,
    WorkerPredictionStatus,
)

# URLs and connection setting to other system applications (message broker, database, shared volume)
RABBITMQ_BROKER_URL = os.environ["RABBITMQ_BROKER_URL"]
RABBITMQ_QUEUE = os.environ["RABBITMQ_INFERENCE_QUEUE"]
MAX_QUEUE_LENGTH = int(os.environ["RABBITMQ_MAX_QUEUE_LENGTH"])
# number of tasks one worker can get at the same time
NUM_TASKS_PER_WORKER = int(os.environ["RABBITMQ_NUM_TASKS_PER_WORKER"])

# setup logging
setup_logging("logging.yaml")
logger = logging.getLogger("app")


YOLO_MODEL_PATH = os.environ["YOLO_MODEL_PATH"]
YOLO_BATCH_SIZE = int(os.environ["YOLO_BATCH_SIZE"])

tracker: GeoRegTracker


def run_inference(
    req_logger: RequestLogger,
    request_id: str,
    video_id: str,
    video_path: str,
    data_calibration: list,
    data_homography: list,
) -> TrackingWorkerResponse:
    """Run tracking on video.

    :param req_logger: Request logger which includes request id in log messages.
    :param request_id: Request ID passed as an input to the worker.
    :param video_id: ID of the video file.
    :param video_path: Path to video file.
    :param data_calibration: [X, y] lists of camara calibration points.
    :param data_homography: [video_points, map_points]
    :return: A WorkerResponse instance with output or error information.
    """
    try:
        # prep data
        data_calibration = [np.array(value) for value in data_calibration]
        data_homography = [np.array(value) for value in data_homography]

        # run video tracking
        start_time = time.time()
        req_logger.info("Running video tracking.")

        predictions = tracker.process_video(
            req_logger=req_logger,
            video_path=video_path,
            data_calibration=data_calibration,
            data_homography=data_homography,
        )
        frame_rate, end_frame = tracker.get_video_info()
        tracking_time = time.time() - start_time
        req_logger.info(f"Tracking Done in {tracking_time}. Output predictions: {len(predictions)}")
        # create response
        out = TrackingWorkerResponse(
            request_id=request_id,
            video_id=video_id,
            status=WorkerPredictionStatus.DONE,
            frame_rate=frame_rate,
            end_frame=end_frame,
            predictions=predictions,
            error=None,
        )

    except Exception as e:
        out = TrackingWorkerResponse(
            request_id=request_id,
            video_id=video_id,
            status=WorkerPredictionStatus.ERROR,
            error=f"Tracking Exception: {e}: {traceback.format_exc()}",
        )

    return out


def on_request(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    method: pika.spec.Basic.Deliver,
    props: pika.spec.BasicProperties,
    body: bytes,
):
    """A callback invoked when a new message is received, runs inference."""
    start_time = time.time()
    request_id = None
    req_logger = RequestLogger(request_id)
    try:
        # decode message
        msg = json.loads(body)
        logger.info(f"Received message: '{msg}' with delivery tag: {method.delivery_tag}")
        msg = TrackingWorkerRequest(**msg)

        # run inference
        request_id = msg.request_id
        req_logger = RequestLogger(request_id)
        response = run_inference(
            req_logger,
            request_id,
            msg.video_id,
            msg.video_path,
            msg.data_calibration,
            msg.data_homography,
        )
    except Exception:
        error = traceback.format_exc()
        req_logger.critical(f"Returning unexpected error output: '{error}'.")
        response = TrackingWorkerResponse(
            request_id=request_id,
            video_id="error",
            status=WorkerPredictionStatus.ERROR,
            error={"internal_error": "Unexpected Error", "traceback": error},
        )

    # send a response and acknowledge the message
    response_str = json.dumps(response.dict(), ensure_ascii=True)
    channel.basic_publish(
        exchange="",
        routing_key=props.reply_to,
        properties=pika.BasicProperties(
            content_type="application/json",
            correlation_id=props.correlation_id,
        ),
        body=bytes(response_str, "utf-8"),
    )
    channel.basic_ack(delivery_tag=method.delivery_tag)
    elapsed_time = time.time() - start_time
    req_logger.info(f"Processed message with ID={request_id} in {elapsed_time:.2f} seconds.")


def consume_pika_messages():
    """Create a message consumer and run `on_request` callback on received messages."""
    # create a connection to rabbitmq
    pika_connection = PikaConnection(broker_url=RABBITMQ_BROKER_URL)
    pika_connection.declare_queue(queue_name=RABBITMQ_QUEUE, max_queue_length=MAX_QUEUE_LENGTH)

    while True:
        try:
            # consume messages
            logger.info("Starting consuming.")
            channel = pika_connection.channel
            channel.basic_qos(prefetch_count=NUM_TASKS_PER_WORKER)
            channel.basic_consume(queue=RABBITMQ_QUEUE, on_message_callback=on_request)
            channel.start_consuming()
        except (ConnectionClosedByBroker, AMQPChannelError):
            # do not recover if connection was closed by broker or on channel errors
            error = traceback.format_exc()
            logger.error(f"Exception while consuming messages:\n{error}")
            break
        except AMQPConnectionError:
            # recover on all other connection errors
            logger.warning("Received AMQPConnectionError, re-establishing connection.")
            pika_connection.connect()
            continue
        finally:
            pika_connection.close()


if __name__ == "__main__":
    logger.info("Starting worker.")
    tracker = GeoRegTracker(model_path=YOLO_MODEL_PATH, batch_frames=YOLO_BATCH_SIZE)
    consume_pika_messages()
