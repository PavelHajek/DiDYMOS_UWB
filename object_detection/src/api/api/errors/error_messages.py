from enum import IntEnum

from ..database import HomographyType
from ..schemas import DataCalibration, DataHomography

IMAGE_EXTENSIONS = ["jpe", "jpeg", "jpg", "png"]
VIDEO_EXTENSIONS = ["mp4", "mts"]


class InputImageError(IntEnum):
    """Processing input errors returned to the client."""

    NotSupportedExtension = 0
    LowResolution = 1
    BluredImage = 2
    ImageLoadingError = 3

    def description(self, spec: str = None):
        """Return description of input error.

        :param spec: Specification string with additional information.
        :return: String description.
        """
        spec = f"'{spec}' " if spec else ""
        if self == InputImageError.NotSupportedExtension:
            out = (
                f"Image file extension {spec}is not supported. "
                f"Please use one of the following: {IMAGE_EXTENSIONS}"
            )
        elif self == InputImageError.ImageLoadingError:
            out = "Image is corrupted and cannot be loaded."
        elif self == InputImageError.LowResolution:
            out = "Image resolution is too low."
        elif self == InputImageError.BluredImage:
            out = "Image is blured."
        else:
            raise ValueError()
        return out

    def code(self):
        """Return error code of input error."""
        return int(self)


class InputVideoError(IntEnum):
    """Processing input errors returned to the client."""

    NotSupportedExtension = 0
    CameraCalibrationMissing = 1
    HomographyMissing = 2
    VideoDatetimeMissing = 3
    HomographyTypeMissing = 4
    NotSupportedHomographyType = 5

    def description(self, spec: str = None):
        """Return description of input error.

        :param spec: Specification string with additional information.
        :return: String description.
        """
        spec = f"'{spec}' " if spec else ""
        if self == InputVideoError.NotSupportedExtension:
            out = (
                f"Video file extension {spec}is not supported. "
                f"Please use one of the following: {VIDEO_EXTENSIONS}"
            )
        elif self == InputVideoError.CameraCalibrationMissing:
            out = f"Camera calibration data missing.\nRequired format: {DataCalibration.schema()}"
        elif self == InputVideoError.HomographyMissing:
            out = f"Homography data missing.\nRequired format: {DataHomography.schema()}"
        elif self == InputVideoError.VideoDatetimeMissing:
            out = "'video_datetime' missing."
        elif self == InputVideoError.HomographyTypeMissing:
            out = "'homography_type' missing."
        elif self == InputVideoError.NotSupportedHomographyType:
            out = (
                f"Homography type {spec}is not supported. "
                f"Please use one of the following: {HomographyType.values()}"
            )
        else:
            raise ValueError()
        return out

    def code(self):
        """Return error code of input error."""
        return int(self)
