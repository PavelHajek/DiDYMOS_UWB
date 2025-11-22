import logging
from datetime import datetime
from typing import List, Tuple
from uuid import UUID

from sqlalchemy import Select, and_, select
from sqlalchemy.exc import DataError, NoResultFound
from sqlalchemy.orm import Session

from ..enums import VideoStatus
from ..exceptions import VideoNotFoundException
from ..model import Video, VideoPrediction
from .base import BaseService

logger = logging.getLogger("database")


def _select_single_video(session: Session, video_id: str) -> Video:
    stmt = select(Video).where(Video.id == str(video_id))
    try:
        video = session.scalars(stmt).first()
        if video is None:
            raise NoResultFound
    except Exception:  # catch error in case of invalid UUID
        msg = f"Video record with id={video_id} not found."
        logger.warning(msg)
        raise VideoNotFoundException(msg)
    return video


def _fetch_all_records_data(session: Session, stmt: Select) -> List[dict]:
    try:
        resp = session.execute(stmt).fetchall()
        videos = [x[0].data() for x in resp]
    except DataError:  # catch error in case of invalid UUID
        videos = []
    return videos


class VideoService(BaseService):
    def create_video(
        self,
        video_datetime: datetime,
        homography_type: str,
        fps: float = None,
        end_frame: int = None,
    ) -> Tuple[str, str]:
        """Create Video record in the database."""
        with Session(self.engine) as session:
            video = Video(
                video_datetime=video_datetime,
                homography_type=homography_type,
                frame_rate=fps,
                end_frame=end_frame,
            )
            session.add(video)
            session.commit()
            session.refresh(video)
            logger.info(f"Created Video record: {video}.")
            return str(video.id), video.status

    def update_video(
        self, video_id: str, status: str = None, fps: float = None, end_frame: int = None
    ) -> UUID:
        """Update a Video record in the database."""
        with Session(self.engine) as session:
            video = _select_single_video(session, video_id)
            if status:
                video.status = status
            if fps:
                video.frame_rate = fps
            if end_frame:
                video.end_frame = end_frame
            session.commit()
            logger.info(f"Update Video record: {video}.")

            return video.id

    def get_video_data(self, video_id: str) -> dict:
        """Get Video from the database."""
        with Session(self.engine) as session:
            video = _select_single_video(session, video_id)
            logger.info(f"Retrieved Video: {video}.")
            return video.data()

    def get_all_videos(self) -> list:
        """Get a list of all Video records from the database."""
        stmt = select(Video)
        with Session(self.engine) as session:
            videos = _fetch_all_records_data(session, stmt)
            logger.info(f"Retrieved list of Video records with {len(videos)} records.")
            return videos

    def create_prediction(
        self,
        video_id: str,
        prediction_datetime: datetime,
        frame_number: int,
        class_name: str,
        class_id: int,
        track_id: int,
        bbox: dict,
        map_position: dict,
    ):
        """Create Prediction record in the database."""
        with Session(self.engine) as session:
            prediction = VideoPrediction(
                video_id=video_id,
                prediction_datetime=prediction_datetime,
                frame_number=frame_number,
                class_name=class_name,
                class_id=class_id,
                track_id=track_id,
                bbox=bbox,
                map_position=map_position,
            )
            session.add(prediction)
            session.commit()
            return str(prediction.id)

    def create_predictions(
        self,
        video_id: str,
        predictions: List[dict],
    ):
        """Create multiple Prediction records in the database.

        Optimized for high number of predictions.
        :param video_id: ID of a video in the database.
        :param predictions: List of predictions. Predictions must have all attributes
         of VideoPrediction.
        :return:
        """
        with Session(self.engine) as session:
            for prediction in predictions:
                prediction = VideoPrediction(video_id=video_id, **prediction)
                session.add(prediction)
            session.commit()
            logger.info(f"Created {len(predictions)} Prediction records.")

    def get_prediction_list(
        self, video_id: str, start_datetime: datetime, end_datetime: datetime
    ) -> list:
        """Get a list of Prediction records from the database."""
        video = self.get_video_data(video_id)
        if video["status"] != VideoStatus.FINISHED:
            return []

        stmt = (
            select(VideoPrediction)
            .where(VideoPrediction.video_id == video_id)
            .where(and_(start_datetime <= VideoPrediction.prediction_datetime))
        )
        if end_datetime >= start_datetime:
            stmt = stmt.where(and_(VideoPrediction.prediction_datetime < end_datetime))
        stmt = stmt.order_by(VideoPrediction.prediction_datetime.asc())

        # execute statement
        with Session(self.engine) as session:
            predictions = _fetch_all_records_data(session, stmt)
            logger.info(f"Retrieved list of Prediction records with {len(predictions)} records.")
            return predictions
