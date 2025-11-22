import logging

from commonlib.log import RequestLogger

logger = logging.getLogger("app")


class BaseService:
    def __init__(self, req_logger: RequestLogger = None):
        self.req_logger = req_logger or logger
