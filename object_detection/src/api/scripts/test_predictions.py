import json
import time
from datetime import datetime
from typing import List

import cv2
import matplotlib.path as mplPath
import numpy as np
import pykalman
import requests
import tqdm
from numpy.linalg import lstsq
from test_config import get_calibration_data, get_homography_data


def calibrate_camera(pixel_positions: np.ndarray, target_values: np.ndarray) -> np.ndarray:
    """Computes a calibration between observed image and real metric heights.

    The calibration finds the best fitting plane ax + by + cz + d = target_values in
    a least square sense.
    It is assumed that z = 0 for all input points, i.e. they lay on a zero level plain.

    The other functions assume that the target values (real metric) is a ratio between the height
    in pixels and the height in lenght metric (such as centimeters).

    Args:
        pixel_positions: Array (N, 2), where N is the number of calibration points in
        the form (x, y).
        target_values: Array (N, 1), as the ratio between pixels and the desired height metric.

    Returns:
        Array (3, ) as the solution to ax + by + d = target_value
    """
    X = pixel_positions
    y = target_values

    X = np.hstack((np.ones((len(X), 1)), X))
    A = lstsq(X, y, rcond=None)[0]

    return A


def pixels2metric(positions: np.ndarray, A: np.ndarray, height_px: np.ndarray) -> np.ndarray:
    """Computes the real metric height for an object height_px high at the desired position.

    Args:
        positions: Array (N, 2). The positions (x, y) at which to compute the real metric height.
        A: Array (3, ). The calibration matrix.
        height_px: Array (N, 1). Heights in pixels to be recomputed to real metric values.

    Returns:
        Array (N, 1). The height in real metric.
    """
    positions = np.hstack((np.ones((len(positions), 1)), positions))
    heights_px_cm_ratio = positions @ A
    heights_cm = 1 / heights_px_cm_ratio * height_px
    return heights_cm


def metric2pixels(positions: np.ndarray, A: np.ndarray, height_cm: np.ndarray) -> np.ndarray:
    """Computes the pixel height for an object height_cm high at the desired position.

    Args:
        positions: Array (N, 2). The positions (x, y) at which to compute the pixel height.
        A: Array (3, ). The calibration matrix.
        height_cm: Array (N, 1). Heights in real metric (such as cm) to be recomputed to
        pixel values.

    Returns:
        Array (N, 1). The height in pixel values.
    """
    positions = np.hstack((np.ones((len(positions), 1)), positions))
    heights_px_cm_ratio = positions @ A
    heights_px = heights_px_cm_ratio * height_cm
    return heights_px


def apply_homography(M, points):
    """Applies homography to camera view points.

    Args:
        M: The homography matrix.
        points: [[(xmin + xmax)/2, (ymin + (ymax - ymin)*2/3)]].

    Returns:
        Points transformed to the map space.
    """
    assert len(points.shape) == 2
    assert points.shape[1] == 2
    points = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    converted_points = (M @ points.T).T
    converted_points = converted_points[:, :2] / converted_points[:, [2]]
    return converted_points


class KalmanFilter:
    def __init__(self, **kwargs):
        # the model assumes state vector [x, y, v_x, v_y]
        # the transition matrix F predicts next state of the system/objects
        transition_matrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        # measurement matrix H considers measurements [x, y]
        observation_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )
        self.kf = pykalman.KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            **kwargs,
        )

    def predict(self, measurements: np.ndarray) -> np.ndarray:
        """Predicts next [x, y] state."""
        assert len(measurements.shape) == 2
        assert measurements.shape[1] == 2
        state_means, state_covariances = self.kf.filter(measurements)
        next_state_mean, next_state_covariance = self.kf.filter_update(
            state_means[-1], state_covariances[-1]
        )
        return next_state_mean[:2]  # use only [x, y]

    def smooth(self, measurements: np.ndarray) -> np.ndarray:
        """Smooths state."""
        assert len(measurements.shape) == 2
        assert measurements.shape[1] == 2
        state_means, _ = self.kf.smooth(measurements)
        # state_means, _ = self.kf.smooth(state_means[:, :2])
        return state_means[-1, :2]  # use only [x, y]


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

    def __init__(self, image_size: tuple, calibration_matrix: np.ndarray):
        self.image_size = image_size  # width, height
        self.calibration_matrix = calibration_matrix
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
            self.mean_real_height[track_id] = np.median([x.real_height for x in track])
            self.mean_video_width[track_id] = np.median(
                [x.video_bbox[2] - x.video_bbox[0] for x in track[-30:]]
            )

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
        track_objects = self.get_track(track_id, top_n=20)
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


def cv2_draw_text(
    img: np.ndarray,
    x: int,
    y: int,
    label: str,
    color: tuple,
    font_scale: float = 0.8,
):
    """Draw a text to img."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(img, (x - 5, y + 5), (x + w + 5, y - h - 5), color, -1)
    cv2.putText(img, label, (x, y), font, font_scale, (0, 0, 0), thickness)


class PredVideo:
    def __init__(
        self,
        batch_frames: int = 50,
    ):
        self.batch_frames = batch_frames

        self.frame_number = 0
        self.kf, self.track_history = None, None
        self.fps, self.end_frame, self.image_size = None, None, None
        self.A = None

        self.preds_out = []

    def process_video(
        self,
        video_path: str,
        predictions: dict,
        data_calibration: List[np.ndarray],
        data_homography: List[np.ndarray],
        map_image_file_path: str,
        output_video_file_path: str,
    ):
        """Loads video and used supplied tracking predictions to plot them to a new video."""
        # define polygon points on map
        polygon = [
            [1100, 1100],
            [1084, 1468],
            [2300, 1520],
            [2310, 1140],
        ]
        polygon = np.array(polygon)

        # load map image
        map_image = cv2.imread(map_image_file_path)

        map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)
        # map_image = cv2.resize(map_image, (1920, 1080))
        polygon = (polygon / 2).astype(int)

        capture = cv2.VideoCapture(video_path)
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        self.end_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.image_size = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        self.calibrate(data_calibration, data_homography)

        # create kalman filter for estimating next object's position for drawing an arrow
        self.kf = KalmanFilter()
        # process frames
        self.track_history = TrackHistory(image_size=self.image_size, calibration_matrix=self.A)

        output_image_size = (self.image_size[0], self.image_size[1] * 2)
        # output video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_video_file_path, fourcc, self.fps, output_image_size)

        self.frame_number = 0
        for batch_index in tqdm.tqdm(range(self.end_frame // self.batch_frames + 1)):
            frames = self._get_batch_frames(capture, self.batch_frames)
            if not frames:
                break

            self._process_batch(
                frames,
                predictions,
                map_image,
                polygon,
                video_writer,
            )

        capture.release()
        video_writer.release()

    @staticmethod
    def _get_batch_frames(capture: cv2.VideoCapture, frame_count: int) -> List[np.ndarray]:
        frames = []
        for frame_id in range(0, frame_count):
            # get image from video
            ret, img = capture.read()
            if img is None:
                continue

            # process i-th frame
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return frames

    def calibrate(self, data_calibration, data_homography):
        """Create calibration and homography matrices."""
        # [x_im, y_im] * [a1, a2]^T = y
        self.A = calibrate_camera(*data_calibration)

        # estimate homography
        # self.M, _ = cv2.findHomography(*data_homography)

    def _process_batch(
        self,
        frames: List[np.ndarray],
        predictions: dict,
        map_image,
        polygon,
        video_writer,
    ):
        result_preds = []
        polygon_path = mplPath.Path(polygon)

        for i, image in enumerate(frames):
            image = image.copy()
            _map_image = map_image.copy()

            framed_predictions = predictions.get(self.frame_number, None)
            self.frame_number += 1
            if framed_predictions is None:
                continue
            # add bounding boxes to image
            any_object_in_polygon = False
            for single_prediction in framed_predictions:
                label = single_prediction["class_name"]
                # class_id = single_prediction["class_id"]
                bbox = single_prediction["image_rect"]
                map_xy = single_prediction["position"]
                track_id = single_prediction["track_id"]

                xmin, ymin, xmax, ymax = (
                    bbox["x_left"],
                    bbox["y_top"],
                    bbox["x_left"] + bbox["width"],
                    bbox["y_top"] + bbox["height"],
                )
                x_t, y_t = int(map_xy["x"]), int(map_xy["y"])

                # update track history
                # self.track_history.add_object(track_id, (xmin, ymin, xmax, ymax))
                #
                # # get current track object with fixed bounding box
                # detected_object = self.track_history.get_current_object(track_id)
                # # xmin, ymin, xmax, ymax = detected_object.estimated_video_bbox
                #
                # # estimate position on the map
                # # x_t, y_t = apply_homography(
                # #     self.M, np.array([[(xmin + xmax) / 2, (ymin + (ymax - ymin) * 2 / 3)]])
                # # )[0].astype(int)
                # detected_object.map_xy = (x_t, y_t)

                # test if the object is inside the polygon
                any_object_in_polygon = any_object_in_polygon or polygon_path.contains_point(
                    (x_t, y_t)
                )

                # get trajectory history
                track_objects = self.track_history.get_track(track_id, top_n=20)
                if len(track_objects) > 5:
                    going_up = track_objects[0].map_xy[1] - track_objects[-1].map_xy[1] > 3
                else:
                    going_up = False

                # visualization setings
                color = (0, 255, 255) if label == "person" else (255, 255, 0)
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

                # draw detection
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                if label == "person":
                    video_width = xmax - xmin
                    cv2.ellipse(
                        image,
                        center=(int(xmin + (xmax - xmin) / 2), ymax),
                        axes=(video_width, int(0.35 * video_width)),
                        angle=0.0,
                        startAngle=-45,
                        endAngle=235,
                        color=color,
                        thickness=2,
                        lineType=cv2.LINE_4,
                    )
                cv2_draw_text(
                    image,
                    x=xmin,
                    y=ymin,
                    label=f"{track_id:.0f}",
                    color=color,
                    font_scale=1.8,
                )
                cv2_draw_text(
                    _map_image,
                    x=x_t + 20 * int(going_up),  # include margin
                    y=y_t + 10 * int(going_up),  # include margin
                    label=f"{track_id:.0f}-",
                    color=color,
                    font_scale=1.8,
                )

                # # draw trajectory
                thickness = 2
                for start_object, end_object in zip(track_objects[:-1], track_objects[1:]):
                    cv2.line(
                        image,
                        start_object.estimated_video_xy,
                        end_object.estimated_video_xy,
                        color,
                        thickness,
                    )
                    cv2.line(_map_image, start_object.map_xy, end_object.map_xy, color, thickness)
                cv2.circle(_map_image, (x_t, y_t), 14, color, -1)
                # draw trajectory arrow
                min_distance = 5
                norm_distance = 40
                if len(track_objects) > 1:
                    next_map_xy = self.kf.predict(np.array([x.map_xy for x in track_objects]))
                    distance = np.linalg.norm(next_map_xy - np.array((x_t, y_t)))
                    if distance > min_distance:
                        # normalize distance
                        x1, y1 = (x_t, y_t)
                        x2, y2 = next_map_xy
                        norm_next_map_xy = (
                            int(x1 + (norm_distance / distance) * (x2 - x1)),
                            int(y1 + (norm_distance / distance) * (y2 - y1)),
                        )
                        cv2.arrowedLine(
                            _map_image,
                            (x_t, y_t),
                            norm_next_map_xy,
                            color,
                            thickness=4,
                            tipLength=0.5,
                        )
            # draw polygon area
            transparent_copy = np.zeros_like(_map_image, np.uint8)
            if any_object_in_polygon:
                alpha = 0.2
                color = (255, 0, 0)
            else:
                alpha = 0.5
                color = (255, 255, 128)
            cv2.drawContours(transparent_copy, [polygon[:, None, ...]], 0, color, -1)
            mask = transparent_copy.astype(bool)
            _map_image[mask] = cv2.addWeighted(_map_image, alpha, transparent_copy, 1 - alpha, 0)[
                mask
            ]

            # merge two images
            concat_image = np.concatenate([image, _map_image], axis=0)

            # write image to the output video
            concat_image = cv2.cvtColor(concat_image, cv2.COLOR_RGB2BGR)
            video_writer.write(concat_image)

        return result_preds


# Requests
def post_video(
    url: str,
    port: str,
    video_path: str,
    video_datetime: str,
    homography_type: str,
    data_calibration: list,
    data_homography: list,
) -> dict:
    """Post video request."""
    response = requests.post(
        url=f"http://{url}:{port}",
        files={
            "video": open(video_path, "rb"),
        },
        data={
            "video_parameters": json.dumps(
                {
                    "video_datetime": video_datetime,
                    "homography_type": homography_type,
                    "data_calibration": {
                        "X": data_calibration[0].tolist(),
                        "y": data_calibration[1].tolist(),
                    },
                    "data_homography": {
                        "video_points": data_homography[0].tolist(),
                        "map_points": data_homography[1].tolist(),
                    },
                }
            )
        },
    )
    video_data = response.json()
    print(video_data)
    return video_data


def is_status_finished(url: str, port: str, video_id: str) -> bool:
    """Tests if video status is finished."""
    response = requests.get(
        f"http://{url}:{port}/video/status", params={"video_id": video_id}
    ).json()
    video_status = response.get("status", None)
    return video_status == "finished"


def get_predictions(
    url: str, port: str, video_id: str, start_datetime: datetime, end_datetime: datetime
) -> list:
    """Lists predictions of the video."""
    response = requests.get(
        f"http://{url}:{port}/video",
        params={
            "video_id": video_id,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
        },
    )
    content = response.json()
    return content["predictions"]


if __name__ == "__main__":
    # map_image_file_path = "../../../resources/googlemap_hd.png"
    map_image_file_path = "../../../resources/googlemap-1080p.png"
    # video_path = "../../../resources/C0001_10fps-cut.mp4"
    video_path = "../../../resources/00001-10fps.MTS"
    # video_path = "../../../resources/C0003_10fps.mp4"
    output_video_file_path = "../../out/test_from_api.mp4"

    data_calibration = get_calibration_data(video_path)
    data_homography = get_homography_data(video_path, wgs=True)

    url = "0.0.0.0"
    # url = "147.228.47.72"
    port = "8080"
    video_datetime = "2024-07-18 11:47:54.000000"
    homography_type = "wgs"

    video_data = post_video(
        url, port, video_path, video_datetime, homography_type, data_calibration, data_homography
    )
    video_id = video_data["video_id"]
    # video_id = "dd64b9a4-e3b0-4434-ba08-3ce816b7d116"

    while not is_status_finished(url, port, video_id):
        time.sleep(10)

    date_format = "%Y-%m-%d %H:%M:%S.%f"
    # start_datetime = datetime.strptime("2024-07-03 11:31:27.809041", date_format)
    # end_datetime = datetime.strptime("2024-07-03 11:41:27.809041", date_format)

    start_datetime = datetime.strptime("2024-07-18 11:47:54.000000", date_format)
    end_datetime = datetime.strptime("2024-07-18 11:52:54.000000", date_format)

    predictions = get_predictions(url, port, video_id, start_datetime, end_datetime)
    predictions = {p["frame_number"]: p["objects"] for p in predictions}
    print(len(predictions))

    processor = PredVideo(batch_frames=100)
    processor.process_video(
        video_path,
        predictions,
        data_calibration,
        data_homography,
        map_image_file_path,
        output_video_file_path,
    )
