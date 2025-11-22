from enum import Enum
from typing import Optional

from pydantic import BaseModel

"""
Input Models
"""


class TrackingWorkerRequest(BaseModel):
    request_id: str
    video_id: str
    video_path: str
    data_calibration: list
    data_homography: list


"""
Response Models
"""


class WorkerPredictionStatus(str, Enum):
    DONE = "DONE"
    ERROR = "ERROR"


class TrackingWorkerResponse(BaseModel):
    request_id: str
    video_id: str
    status: WorkerPredictionStatus
    frame_rate: Optional[float] = None
    end_frame: Optional[int] = None
    predictions: Optional[list] = None
    error: Optional[str] = None
