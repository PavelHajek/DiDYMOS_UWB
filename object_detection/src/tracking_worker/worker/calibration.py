import numpy as np
import pykalman
from numpy.linalg import lstsq


def calibrate_camera(pixel_positions: np.ndarray, target_values: np.ndarray) -> np.ndarray:
    """Computes a calibration between observed image and real metric heights.

    The calibration finds the best fitting plane ax + by + cz + d = target_values in the
    least square sense.
    It is assumed that z = 0 for all input points, i.e. they lay on a zero level plain.

    The other functions assume that the target values (real metric) is a ratio between
    the height in pixels and the height in lenght metric (such as centimeters).

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
