from enum import Enum
from typing import Any


class CoreEnum(Enum):
    @classmethod
    def values(cls) -> list:
        """Get list of Enum values."""
        return [x.value for x in cls]

    @classmethod
    def names(cls) -> list:
        """Get list of Enum names."""
        return [x.name for x in cls]

    @classmethod
    def get(cls, value: Any, default: Any = None) -> Enum:
        """Get Enum instance for the given value."""
        return cls(value) if value in cls.values() else default


# """
# Video related.
# """


class VideoStatus(str, CoreEnum):
    PROCESSING = "processing"
    FINISHED = "finished"
    CRASHED = "crashed"


class HomographyType(str, CoreEnum):
    PIXEL = "pixel"
    WGS = "wgs"
