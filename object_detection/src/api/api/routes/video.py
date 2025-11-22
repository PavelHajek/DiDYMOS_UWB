from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from commonlib.log import RequestLogger

from ..database import VideoNotFoundException, get_db_connection
from ..dependencies import request_logger_dependency
from ..schemas import DetailedVideoList, DetailedVideoResponse, VideoPredictions, VideoResponse

router = APIRouter(
    prefix="/video",
    tags=["video"],
)


@router.get("/status", status_code=status.HTTP_200_OK, name="Get Video Status")
def get_video_status(
    video_id: str,
    # dependencies
    req_logger: RequestLogger = Depends(request_logger_dependency),
) -> VideoResponse:
    """Endpoint for Video record."""
    # get data from the database
    db_connection = get_db_connection()
    try:
        video = db_connection.video.get_video_data(str(video_id))
    except VideoNotFoundException:
        raise HTTPException(
            detail=f"Video with id: {video_id} not found.", status_code=status.HTTP_404_NOT_FOUND
        )

    req_logger.info(f"Retrieved Video record with id={video_id}; {video}")
    return VideoResponse(video_id=video_id, **video)


@router.get("", status_code=status.HTTP_200_OK, name="Get Video Predictions")
def get_video_predictions(
    # query
    video_id: str,
    start_datetime: datetime = datetime.min,
    end_datetime: datetime = datetime.max,
    # dependencies
    req_logger: RequestLogger = Depends(request_logger_dependency),
) -> VideoPredictions:
    """Endpoint for Prediction list."""
    # get data from the database
    db_connection = get_db_connection()
    try:
        predictions_list = db_connection.video.get_prediction_list(
            video_id=video_id, start_datetime=start_datetime, end_datetime=end_datetime
        )
    except VideoNotFoundException:
        raise HTTPException(
            detail=f"Video with id: {video_id} not found.", status_code=status.HTTP_404_NOT_FOUND
        )

    req_logger.info(
        f"Retrieved Video Predictions list with id={video_id}; with length={len(predictions_list)}"
    )

    framed_predictions = {}
    for pred in predictions_list:
        frame_number = pred["frame_number"]
        if frame_number not in framed_predictions:
            framed_predictions[frame_number] = []
        framed_predictions[frame_number].append(
            {
                "class_name": pred["class_name"],
                "class_id": pred["class_id"],
                "timestamp": pred["prediction_datetime"],
                "image_rect": pred["bbox"],
                "position": pred["map_position"],
                "track_id": pred["track_id"],
            }
        )

    predictions_list = [
        {"frame_number": frame, "objects": preds} for frame, preds in framed_predictions.items()
    ]

    return VideoPredictions(predictions=predictions_list)


@router.get("/all", status_code=status.HTTP_200_OK, name="Get all Videos")
def get_all_videos(
    # dependencies
    req_logger: RequestLogger = Depends(request_logger_dependency),
) -> DetailedVideoList:
    """List all Videos in the database."""
    db_connection = get_db_connection()
    videos = db_connection.video.get_all_videos()
    req_logger.info(f"Retrieved Videos list with length={len(videos)}")
    for video in videos:
        video["video_id"] = str(video["id"])
    response = [DetailedVideoResponse(**video) for video in videos]

    return DetailedVideoList(videos=response)
