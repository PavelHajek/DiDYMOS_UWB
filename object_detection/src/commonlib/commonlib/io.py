import io
import json
import logging
import os
import traceback
from typing import Any, Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger("app")


class ImageError(Exception):
    """Exception raised while loading a Pillow image."""


def verify_image(image_bytes: bytes):
    """Load a Pillow image from bytes and verify it is not corrupted.

    :param image_bytes: Image content in bytes.
    """
    try:
        image_pil = Image.open(io.BytesIO(image_bytes))
        image_pil.verify()
    except UnidentifiedImageError:
        error = traceback.format_exc()
        logger.error(f"Error while loading Pillow image:\n{error}")
        raise ImageError()


def load_image(file_path: str) -> Image.Image:
    """Load a Pillow image from the file system.

    :param file_path: Image path in the file system.
    :return: Pillow image.
    """
    try:
        image_pil = Image.open(file_path).copy()
        image_pil = correct_image_orientation(image_pil)
    except UnidentifiedImageError:
        error = traceback.format_exc()
        logger.error(f"Error while loading Pillow image:\n{error}")
        raise ImageError()
    return image_pil


def load_json(file_path: str) -> dict:
    """Load a JSON file from the file system.

    :param file_path: File path in the file system.
    :return: A dictionary.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def load_file(file_path: str) -> Any:
    """Load a file from the file system.

    :param file_path: File path in the file system.
    :return: A file content.
    """
    with open(file_path, "rb") as f:
        return f.read()


def save_image(image: Union[Image.Image, np.ndarray], file_path: str, format: str = None):
    """Save a Pillow image to the file system.

    :param image: A Pillow image to save.
    :param file_path: Image path in the local file system.
    :param format: Optional format override.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    with open(file_path, "wb") as f:
        image.save(f, format=format)


def save_json(dict_: dict, file_path: str):
    """Save a dictionary as JSON to the file system.

    :param dict_: A dictionary to save.
    :param file_path: File path in the local file system.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(dict_, f)


def save_file(file_content: Any, file_path: str):
    """Save a file to the file system.

    :param file_content: File content to save.
    :param file_path: File path in the local file system.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)


def correct_image_orientation(image: Image) -> Image:
    """Correct image orientation if EXIF data present.

    Load an image and correct its orientation using EXIF data if available
    else return copied image.
    :param image: PIL Image object
    :return: A PIL Image object with corrected or original orientation.
    """
    image = ImageOps.exif_transpose(image)
    return image
