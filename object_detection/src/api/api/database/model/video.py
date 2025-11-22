import logging
import uuid

from sqlalchemy import JSON, Column, DateTime, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from ..enums import HomographyType, VideoStatus
from .base import Base

logger = logging.getLogger("database")


class Video(Base):
    """Video record."""

    __tablename__ = "video"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_datetime = Column(DateTime(timezone=False))
    homography_type = Column(Enum(HomographyType))
    status = Column(Enum(VideoStatus), default=VideoStatus.PROCESSING)
    frame_rate = Column(Float, nullable=True)
    end_frame = Column(Integer, nullable=True)

    predictions = relationship(
        "VideoPrediction", passive_deletes=True, order_by="asc(VideoPrediction.frame_number)"
    )


class VideoPrediction(Base):
    """Prediction record describing a single detected object in a single frame."""

    __tablename__ = "video_prediction"
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(UUID, ForeignKey("video.id"))
    prediction_datetime = Column(DateTime(timezone=False))
    frame_number = Column(Integer, nullable=False)
    class_name = Column(String, nullable=False)
    class_id = Column(Integer, nullable=False)
    track_id = Column(Integer, nullable=False)
    bbox = Column(JSON, nullable=False)
    map_position = Column(JSON, nullable=False)
