from enum import IntEnum

IMAGE_EXTENSIONS = ["jpe", "jpeg", "jpg", "png"]


class InputError(IntEnum):
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
        if self == InputError.NotSupportedExtension:
            out = (
                f"Image file extension {spec}is not supported. "
                f"Please use one of the following: {IMAGE_EXTENSIONS}"
            )
        elif self == InputError.ImageLoadingError:
            out = "Image is corrupted and cannot be loaded."
        elif self == InputError.LowResolution:
            out = "Image resolution is too low."
        elif self == InputError.BluredImage:
            out = "Image is blured."
        else:
            raise ValueError()
        return out

    def code(self):
        """Return error code of input error."""
        return int(self)
