import json
from datetime import datetime
from typing import List, Optional, Tuple

from pydantic import BaseModel, model_validator

from ..database import HomographyType, VideoStatus


class DataCalibration(BaseModel):
    X: List[Tuple[int, int]] = None
    y: List[float] = None


class DataHomography(BaseModel):
    video_points: List[Tuple[int, int]] = None
    map_points: List[Tuple[float, float]] = None


class VideoParametersInput(BaseModel):
    video_datetime: datetime
    homography_type: HomographyType
    data_calibration: DataCalibration
    data_homography: DataHomography

    @model_validator(mode="before")
    @classmethod
    def validate_to_json(cls, value):
        """Parse string as a json format value."""
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


class VideoResponse(BaseModel):
    video_id: str
    video_datetime: datetime
    homography_type: HomographyType
    status: VideoStatus


class DetailedVideoResponse(VideoResponse):
    frame_rate: Optional[float] = None
    end_frame: Optional[float] = None
    date_created: datetime
    date_updated: datetime


class DetailedVideoList(BaseModel):
    videos: List[DetailedVideoResponse]


class ImageRect(BaseModel):
    x_left: float
    y_top: float
    width: float
    height: float


class Position(BaseModel):
    x: float
    y: float


class Prediction(BaseModel):
    class_name: str
    class_id: int
    timestamp: datetime
    image_rect: ImageRect
    position: Position
    track_id: int


class FramePredictions(BaseModel):
    frame_number: int
    objects: List[Prediction]


class VideoPredictions(BaseModel):
    predictions: List[FramePredictions]
