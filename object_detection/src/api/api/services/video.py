import os
import traceback
from datetime import timedelta

from commonlib.log import simplify_object
from commonlib.task_queue import NoConsumerException, RpcPublisher, WorkerTimeoutException
from commonlib.task_queue.models import (
    TrackingWorkerRequest,
    TrackingWorkerResponse,
    WorkerPredictionStatus,
)

from ..config import RABBITMQ_QUEUES, WORKER_TIMEOUT_SECONDS
from ..database import VideoStatus, get_db_connection
from .base import BaseService
from .workers import get_pika_connection


class VideoService(BaseService):
    async def run_remote_inference(self, msg: TrackingWorkerRequest) -> object:
        """Send tracking request to RabbitMQ queue, catch and report exceptions.

        :param msg: A dictionary with a message to send.
        """
        db_connection = get_db_connection()
        video_id = msg.video_id
        video_path = msg.video_path
        self.req_logger.info(
            "Sending inference request to the worker for Video with id={video_id}."
        )
        frame_rate, end_frame, predictions, prediction_error = None, None, None, None
        try:
            # send rabbitmq message to inference worker
            pika_connection = get_pika_connection()
            rpc_publisher = RpcPublisher(pika_connection)
            response = await rpc_publisher.send_message(
                queue_name=RABBITMQ_QUEUES["inference"],
                msg=msg.dict(),
                timeout_seconds=WORKER_TIMEOUT_SECONDS,
                message_ttl_seconds=WORKER_TIMEOUT_SECONDS,
            )
            self.req_logger.info(f"Retrieved response from worker: {simplify_object(response)}.")
            response = TrackingWorkerResponse(**response)

            # process output
            if response.status == WorkerPredictionStatus.DONE:
                frame_rate = response.frame_rate
                end_frame = response.end_frame
                predictions = response.predictions
            else:
                prediction_error = response.error
        except (WorkerTimeoutException, NoConsumerException):
            prediction_error = "Inference Worker Not Available"
        except Exception:
            # add extra exception handling because of asyncio.create_task method
            error = traceback.format_exc()
            prediction_error = f"Unexpected Error:\n{error}"
            self.req_logger.critical(prediction_error)

        # save results to the database
        prediction_state = VideoStatus.FINISHED if prediction_error is None else VideoStatus.CRASHED
        self.req_logger.info(
            "Saving inference result to the database: "
            f"state={prediction_state}; error={prediction_error}"
        )

        video_service = db_connection.video
        try:
            video_data = video_service.get_video_data(video_id)
            video_datetime = video_data["video_datetime"]

            for pred in predictions:
                prediction_frame_number = pred["frame_number"]
                delta_seconds = prediction_frame_number / frame_rate
                pred["prediction_datetime"] = video_datetime + timedelta(seconds=delta_seconds)

            video_service.create_predictions(video_id=video_id, predictions=predictions)

            video_service.update_video(
                video_id, status=VideoStatus.FINISHED, fps=frame_rate, end_frame=end_frame
            )
        except Exception:
            video_service.update_video(video_id, status=VideoStatus.CRASHED)
            self.req_logger.info(f"Tracking of video with id={video_id} crashed.")
        finally:
            os.remove(video_path)

    async def process_video(
        self,
        video_id: str,
        video_path: str,
        data_calibration: list,
        data_homography: list,
    ) -> str:
        """Remote inference wrapper."""
        # create prediction task in rabbitmq
        # run task using asyncio and don't wait for a response
        # here, asyncio is used for reconnecting to the message broker
        await self.run_remote_inference(
            msg=TrackingWorkerRequest(
                request_id=f"video_id={video_id}",
                video_id=video_id,
                video_path=video_path,
                data_calibration=data_calibration,
                data_homography=data_homography,
            )
        )
        return video_id
