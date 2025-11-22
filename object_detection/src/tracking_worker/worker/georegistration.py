import gc
import logging
import os
import time
import traceback
from functools import wraps
from multiprocessing import Process, Queue
from typing import List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from commonlib.log import RequestLogger

from .calibration import (
    KalmanFilter,
    apply_homography,
    calibrate_camera,
    metric2pixels,
    pixels2metric,
)

logger = logging.getLogger("app")

YOLO_IMAGE_RESOLUTION = os.environ["YOLO_IMAGE_RESOLUTION"]
N_JOBS = os.environ["NUM_CPU_JOBS"]
TRACKING_TIMEOUT_SECS = float(os.environ["WORKER_TIMEOUT_SECS"])
TRACK_HISTORY_LENGTH = int(os.environ["TRACK_HISTORY_LENGTH"])

RECOGNIZED_CLASSES = ["person", "bicycle", "car", "motorcycle"]


def timeit(func):
    """Run time measuring decorator."""

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.debug(f'Function "{func.__name__}" Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def init_yolo_model(model_path: str) -> YOLO:
    """Creates new instance of YOLO model on a specific device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)
    logger.info(f"Moving YOLO model: {model_path} to {device.type} device.")
    model.to(device)
    return model


@timeit
def yolo_track(model: YOLO, frames: List[np.ndarray], scale: float) -> List[dict]:
    """Detects objects in the video frames and tracks them.

    Frames are processed as a whole batch.

    :param model: Instance of an YOLO model.
    :param frames: List of images/video frames.
    :param scale: Scaling factor to rescale predictions back to the original size.
    :return: List of predictions.
    """
    results = model.track(
        frames,
        imgsz=YOLO_IMAGE_RESOLUTION,  # must be a multiple of stride 32
        verbose=False,
        tracker="bytetrack.yaml",
        persist=True,
    )

    if not results:
        return []

    id2label = results[0].names

    preds = []
    for x in results:
        if not x.boxes.is_track:
            logger.warning("Bbox is_track is false! Skipping prediction.")
            continue
        class_ids = x.boxes.cls.cpu().numpy().astype(int).tolist()
        preds.append(
            {
                # "full_image_path": x.path,
                # "image_path": os.path.relpath(x.path, dataset_dir),
                "bboxes": (x.boxes.xyxy.cpu().numpy() / scale).tolist(),
                "class_ids": class_ids,
                "labels": [id2label[x] for x in class_ids],
                "track_ids": x.boxes.id.numpy().tolist(),
                "confs": x.boxes.conf.cpu().numpy().tolist(),
                "speed": x.speed,
            }
        )
    return preds


def read_batch_resized_frames(capture: cv2.VideoCapture, batch_size: int, new_size: tuple = None):
    """Read batch of frames from capture and optionally resizes them."""
    frames = []
    for frame_id in range(0, batch_size):
        # get image from video
        ret, image = capture.read()
        if image is None:
            continue

        if new_size is not None:
            image = resize_image(image, new_size=new_size)
        # process i-th frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
    return frames


def resize_image(image: np.ndarray, new_size: tuple) -> np.ndarray:
    """Resize image, keep aspect ratio."""
    resized_frame = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_frame


def get_resize_scale(width: int, height: int, max_size: int) -> Tuple[float, Tuple]:
    """Get scaling factor and new dimensions.

    No new dimension is larger than max_size.
    """
    scale = min(max_size / height, max_size / width)
    if scale > 1:
        return 1.0, (width, height)
    new_size = (int(width * scale), int(height * scale))
    return scale, new_size


class DetectedObject:
    video_bbox: tuple
    video_height: float
    real_height: float
    estimated_video_bbox: tuple = None
    estimated_video_height: float = None
    map_xy: tuple = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        video_bbox = self.video_bbox
        video_height = self.video_height
        real_height = self.real_height
        return f"{self.__class__.__name__}({video_bbox=}, {video_height=}, {real_height=})"

    @property
    def video_xy(self):
        """Returns center position of the bbox."""
        xmin, ymin, xmax, ymax = self.video_bbox
        xcenter = int(xmin + (xmax - xmin) / 2)
        return int(xcenter), int(ymin)

    @property
    def estimated_video_xy(self):
        """Returns estimated center position of the bbox."""
        xmin, ymin, xmax, ymax = self.estimated_video_bbox
        xcenter = int(xmin + (xmax - xmin) / 2)
        return int(xcenter), int(ymax)


class TrackHistory:
    DEFAULT_REAL_HEIGHT = 180  # centimeters

    def __init__(
        self, image_size: tuple, calibration_matrix: np.ndarray, top_n: int = TRACK_HISTORY_LENGTH
    ):
        self.image_size = image_size  # width, height
        self.calibration_matrix = calibration_matrix
        self.top_n = top_n

        self.track_history = {}
        self.mean_real_height = {}
        self.mean_video_width = {}

        # create kalman filter for smoothing bounding boxes
        self.kf = KalmanFilter(
            transition_covariance=np.eye(4) * 5,
            observation_covariance=np.eye(2) * 5,
        )

    def add_object(self, track_id: int, video_bbox: tuple):
        """Adds new bbox object to tracked history."""
        if track_id not in self.track_history:
            self.track_history[track_id] = []

        # compute real height
        xmin, ymin, xmax, ymax = video_bbox
        xcenter = xmin + (xmax - xmin) / 2
        video_height = ymax - ymin
        real_height = pixels2metric(
            np.array([[xcenter, ymin]]), self.calibration_matrix, np.array([video_height])
        )[0]

        # add to track history
        detected_object = DetectedObject(
            video_bbox=video_bbox, video_height=video_height, real_height=real_height
        )
        self.track_history[track_id].append(detected_object)

        # evaluate average height
        if len(self.track_history[track_id]) > 10:
            track = self.track_history[track_id]
            if track_id not in self.mean_real_height or track_id not in self.mean_video_width:
                self.mean_real_height[track_id] = np.mean([x.real_height for x in track])
                self.mean_video_width[track_id] = np.mean(
                    [x.video_bbox[2] - x.video_bbox[0] for x in track[-30:]]
                )
            else:  # Switch to incremental update of mean values
                self.mean_real_height[track_id] = self.mean_real_height[track_id] + (
                    track[-1].real_height - self.mean_real_height[track_id]
                ) / len(track)
                self.mean_video_width[track_id] = self.mean_video_width[track_id] + (
                    (track[-1].video_bbox[2] - track[-1].video_bbox[0])
                    - self.mean_video_width[track_id]
                ) / len(track)

        # estimate video height
        mean_real_height = self.mean_real_height.get(track_id, self.DEFAULT_REAL_HEIGHT)
        estimated_video_height = metric2pixels(
            np.array([detected_object.video_xy]),
            self.calibration_matrix,
            np.array([[mean_real_height]]),
        )[0, 0]
        detected_object.estimated_video_height = estimated_video_height

        # fix height of occluded bounding box
        if video_height / estimated_video_height < 0.9:
            # object got occluded, fix height
            new_ymax = min(self.image_size[1], ymin + estimated_video_height)
            new_xmin, new_xmax = xmin, xmax
            if track_id in self.mean_video_width:
                mean_video_width = self.mean_video_width[track_id]
                video_width = xmax - xmin
                if video_width < mean_video_width:
                    delta = mean_video_width - video_width
                    new_xmin, new_xmax = xmin - delta / 2, xmax + delta / 2
            detected_object.estimated_video_bbox = (new_xmin, ymin, new_xmax, new_ymax)
        else:
            # keep detected bounding box
            detected_object.estimated_video_bbox = detected_object.video_bbox

        # fix bounding box using kalman filter
        track_objects = self.get_track(track_id, top_n=self.top_n)
        if len(track_objects) > 5:
            assert track_objects[-1] is detected_object
            new_xmin, new_ymin = self.kf.smooth(
                np.array([x.estimated_video_bbox[:2] for x in track_objects])
            )
            new_xmax, new_ymax = self.kf.smooth(
                np.array([x.estimated_video_bbox[2:] for x in track_objects])
            )
            detected_object.estimated_video_bbox = (new_xmin, new_ymin, new_xmax, new_ymax)

    def get_current_object(self, track_id: int):
        """Returns last object from history based on the id."""
        track_objects = self.track_history.get(track_id)
        return track_objects[-1] if len(track_objects) > 0 else None

    def get_track(self, track_id: int, top_n: 20):
        """Returns last n objects from history based on the id."""
        track_objects = self.track_history.get(track_id, [])
        if top_n is not None:
            track_objects = track_objects[-top_n:]
        return track_objects

    def get_track_history(self) -> dict:
        """Returns the whole history of all tracked objects."""
        return self.track_history


class GeoRegTracker:
    def __init__(
        self,
        model_path: str,
        batch_frames: int = 50,
    ):
        self.model_path = model_path
        self.batch_frames = batch_frames

        self.model = init_yolo_model(self.model_path)

        self.track_history = None
        self.fps, self.end_frame, self.image_size = None, None, None
        self.A, self.M = None, None

        self.scale, self.new_size = 1.0, None  # Frames within YOLO resolution are not resized.

    @timeit
    def process_video(
        self,
        req_logger: RequestLogger,
        video_path: str,
        data_calibration: List[np.ndarray],
        data_homography: List[np.ndarray],
    ) -> list:
        """Track objects in a local video file.

        Loads local video file and inferences on the video frames.
        This is done in batches of frames depending on the size of memory,
        must be set manually in .env.

        Returns a list of predictions that compose of class info, object bbox,
        map info, and track id.

        :param req_logger: Logger for a specific request.
        :param video_path: Path to a video file.
        :param data_calibration: [X, y] lists of calibration points.
        :param data_homography: [video_points, map_points].
        :return: List of processed predictions.
        """
        capture = None
        try:
            self.calibrate(
                req_logger=req_logger,
                data_calibration=data_calibration,
                data_homography=data_homography,
            )

            capture = self.open_video_capture(req_logger, video_path)

            self.track_history = TrackHistory(image_size=self.image_size, calibration_matrix=self.A)

            preds = self.track_parallel(
                req_logger=req_logger,
                capture=capture,
                batch_size=self.batch_frames,
                track_history=self.track_history,
                M=self.M,
                scale=self.scale,
            )

        except Exception as e:
            req_logger.warning(traceback.format_exc())
            raise e
        finally:
            # Clean up
            self.model.predictor.trackers[0].reset()  # Reset model track history
            if capture:
                capture.release()

        return preds

    def calibrate(
        self,
        req_logger: RequestLogger,
        data_calibration: List[np.ndarray],
        data_homography: List[np.ndarray],
    ):
        """Create calibration and homography matrices."""
        # [x_im, y_im] * [a1, a2]^T = y
        self.A = calibrate_camera(*data_calibration)

        # estimate homography
        self.M, _ = cv2.findHomography(*data_homography)

        errors = np.linalg.norm(
            apply_homography(self.M, np.array(data_homography[0])) - np.array(data_homography[1]),
            axis=1,
        )
        req_logger.debug(f"Estimation error (max): {errors.max()}")
        req_logger.debug(f"Estimation error (mean): {errors.mean()}")

    def open_video_capture(self, req_logger: RequestLogger, video_path: str) -> cv2.VideoCapture:
        """Creates video capture and loads video parameters.

        :param req_logger: Logger for a specific request.
        :param video_path: Path to the video file.
        :return: Opened video capture.
        """
        capture = cv2.VideoCapture(video_path)
        # Get video info
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        self.end_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.image_size = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        max_size = int(YOLO_IMAGE_RESOLUTION)
        # Resize only if larger than YOLO input resolution
        if any([size > max_size for size in self.image_size]):
            self.scale, self.new_size = get_resize_scale(*self.image_size, max_size)
        req_logger.info(f"Resize: {self.image_size} >> {self.new_size}")
        return capture

    def get_video_info(self) -> tuple:
        """Return stats about the processed video."""
        return self.fps, self.end_frame

    @timeit
    def track_parallel(
        self,
        req_logger: RequestLogger,
        capture: cv2.VideoCapture,
        batch_size: int,
        track_history: TrackHistory,
        M: np.ndarray,
        scale: float,
        timeout: int = TRACKING_TIMEOUT_SECS,
    ) -> list:
        """Inference on images. Runs postprocessing in a parallel process.

        Loads images in parallel in batches, runs yolo inference and puts predictions into
        working queue, where the object smoothening and homography proceeds.
        The resulting predictions are harvested from the output queue in
        the end and returned.

        :param req_logger: Logger for a specific request.
        :param capture: Opened video capture.
        :param batch_size: Image batch size.
        :param track_history: History of tracked objects.
        :param M: Homography matrix.
        :param scale: Scaling factor to rescale predictions back to the original size.
        :param timeout: Max output queue timeout.
        :return: List of processed predictions.
        """
        tracked_predictions = []
        working_queue, output_queue = Queue(), Queue()

        tracker = Process(
            target=postprocessing_worker,
            args=(req_logger, working_queue, output_queue, track_history, M),
        )

        for batch_offset in range(self.end_frame // batch_size + 1):
            req_logger.debug(
                f"Inference batch: {batch_offset + 1}, {self.end_frame // batch_size + 1}"
            )
            images = read_batch_resized_frames(capture, batch_size, new_size=self.new_size)

            yolo_predictions = yolo_track(self.model, images, scale)
            del images
            gc.collect()

            working_queue.put(yolo_predictions)
            if batch_offset == 0:
                tracker.start()

        if tracker:  # Wait for tracking process to finish.
            working_queue.put("KILL")
            tracked_predictions, track_history = output_queue.get(block=True, timeout=timeout)
            tracker.terminate()

        return tracked_predictions


@timeit
def postprocessing_worker(
    req_logger: RequestLogger,
    working_queue: Queue,
    output_queue: Queue,
    track_history: TrackHistory,
    M: np.ndarray,
    timeout: int = 120,
) -> None:
    """Postprocessing Worker with input and output queues.

    Smoothens tracked YOLO predictions and applies homography to them.
    When "KILL" command is sent, return all tracked predictions.

    :param req_logger: Request logger instance.
    :param working_queue: Queue for retrieving YOLO predictions.
    :param output_queue: Queue to return all processed predictions.
    :param track_history: History of tracked objects.
    :param M: Homography matrix.
    :param timeout: Max output queue timeout.
    :return: None
    """
    start_time = time.time()
    tracking_predictions = []
    frame_index = 0

    try:
        while True:
            yolo_predictions = working_queue.get(timeout=timeout)
            if yolo_predictions == "KILL":
                break

            frame_index, _tracking_predictions, track_history = run_smoothening_homography(
                yolo_predictions, track_history, M, frame_index
            )
            tracking_predictions.extend(_tracking_predictions)

    except ImportError:
        req_logger.warning("Tracking video timeout")

    req_logger.debug(f"Tracking video: {time.time() - start_time}")
    output_queue.put((tracking_predictions, track_history))
    return None


@timeit
def run_smoothening_homography(
    yolo_predictions: List[dict], track_history: TrackHistory, M: np.ndarray, frame_number: int = 0
) -> Tuple[int, list, TrackHistory]:
    """Filters tracked predictions. Projects to 2D map with homography matrix.

    :param yolo_predictions: Predictions from YOLO model
    :param track_history: History of tracked objects.
    :param M: Homography matrix.
    :param frame_number: Frame number of the first prediction.
    :return: Last frame number, processed predictions, track history.
    """
    tracking_predictions = []

    for pred in yolo_predictions:
        for bbox, label, class_id, conf, track_id in zip(
            pred["bboxes"], pred["labels"], pred["class_ids"], pred["confs"], pred["track_ids"]
        ):
            # visualize only car and person objects
            if label not in RECOGNIZED_CLASSES:
                continue

            # update track history
            track_history.add_object(track_id, bbox)

            # get current track object with fixed bounding box
            detected_object = track_history.get_current_object(track_id)
            xmin, ymin, xmax, ymax = detected_object.estimated_video_bbox

            # estimate position on the map
            x_t, y_t = apply_homography(
                M, np.array([[(xmin + xmax) / 2, (ymin + (ymax - ymin) * 2 / 3)]])
            )[0].astype(float)

            tracking_predictions.append(
                {
                    "frame_number": frame_number,
                    "class_name": label,
                    "class_id": int(class_id),
                    "track_id": int(track_id),
                    "bbox": {
                        "x_left": float(xmin),
                        "y_top": float(ymin),
                        "width": float(xmax - xmin),
                        "height": float(ymax - ymin),
                    },
                    "map_position": {
                        "x": float(x_t),
                        "y": float(y_t),
                    },
                }
            )
        frame_number += 1
    return frame_number, tracking_predictions, track_history
