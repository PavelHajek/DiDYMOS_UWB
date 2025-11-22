import logging
import logging.config
from datetime import datetime
from typing import Any, Union

import yaml


def setup_logging(cfg_file: str):
    """Setup logging configuration from a file.

    :param cfg_file: Logging configuration yaml file.
    """
    assert cfg_file.lower().split(".")[-1] in ["yml", "yaml"]
    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)


class RequestLogger:
    """Custom Logger that wraps logging.Logger and adds request_id in log messages."""

    def __init__(self, request_id: str = None):
        self.logger = logging.getLogger("app")
        if request_id is None:
            request_id = f"request_id={datetime.now().timestamp()}"
        self.request_id = request_id

    def _include_request_id(self, msg: str) -> str:
        """Add request id to the message."""
        msg = f"({self.request_id}) {msg}"
        return msg

    def debug(self, msg: str, *args, **kwargs):
        """Log 'msg % args' with severity 'DEBUG'."""
        msg = self._include_request_id(msg)
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log 'msg % args' with severity 'INFO'."""
        msg = self._include_request_id(msg)
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log 'msg % args' with severity 'WARNING'."""
        msg = self._include_request_id(msg)
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log 'msg % args' with severity 'ERROR'."""
        msg = self._include_request_id(msg)
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Log 'msg % args' with severity 'CRITICAL'."""
        msg = self._include_request_id(msg)
        self.logger.critical(msg, *args, **kwargs)


def simplify_object(obj: Union[list, dict, Any], max_items: int = 10) -> Union[list, dict, Any]:
    """Make input objects like lists or dictionaries shorted to omit long logs.

    Objects and nested objects like lists or dictionaries are trimmed in length
    specified by max_items. Other data types are kept the same.

    :param obj: Any object to simplify.
    :param max_items: Maximum number of items in list or dictionary.
    :return: A simplified object of the same type.
    """
    if isinstance(obj, list):
        if len(obj) > max_items:
            obj = obj[:max_items] + ["..."]
        obj = [simplify_object(x) for x in obj]
    elif isinstance(obj, dict):
        if len(obj) > max_items:
            obj = dict(list(obj.items())[:max_items] + [("...", "...")])
        obj = {k: simplify_object(v) for k, v in obj.items()}
    elif isinstance(obj, float):
        obj = round(obj, 4)
    return obj
