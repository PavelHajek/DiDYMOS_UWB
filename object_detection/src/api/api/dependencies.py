import logging

from fastapi import Request

from commonlib.log import RequestLogger

logger = logging.getLogger("app")


def request_logger_dependency(request: Request) -> RequestLogger:
    """Create request logger with randomly generated request id and log request information."""
    # initialize logger with randomly generated request id
    req_logger = RequestLogger()
    req_logger.info(
        f"Received {request.method} request '{request.url}' with "
        f"path_params={request.path_params}; query_params={request.query_params or {} }."
    )
    request.state.req_logger = req_logger
    return req_logger
