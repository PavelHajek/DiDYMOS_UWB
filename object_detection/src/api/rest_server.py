import logging
import os
import tempfile
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Callable

from fastapi import BackgroundTasks, Body, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from api import config
from api.database import HomographyType, get_db_connection, init_db_connection
from api.errors import VIDEO_EXTENSIONS, InputVideoError
from api.routes import include_routers
from api.schemas import ErrorResponse, HealthCheckResponse, VideoParametersInput, VideoResponse
from api.services.video import VideoService
from api.services.workers import close_pika_connection, init_pika_connection
from commonlib.log import RequestLogger, setup_logging

# setup logging
setup_logging("logging.yaml")
logger = logging.getLogger("app")


@asynccontextmanager
async def livespan(app: FastAPI):
    """Start auxiliary services on startup and shutdown of the FastAPI app."""
    logger.info("Starting API.")

    # create db connection
    _ = init_db_connection(db_url=os.environ["POSTGRES_URL"])

    # create pika connection
    init_pika_connection(
        broker_url=config.RABBITMQ_BROKER_URL,
        queues=config.RABBITMQ_QUEUES.values(),
        max_queue_length=config.MAX_QUEUE_LENGTH,
    )
    yield
    # stop pika connection on shutdown of the FastAPI app.
    close_pika_connection()


# Create App
app = FastAPI(title="DIDYMOS API", version="0.1.0-dev", lifespan=livespan)
include_routers(app)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next: Callable):
    """Add process time to http response header."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    m, s = divmod(int(process_time), 60)
    logger.info(f"Processed a request in {m:02d}:{s:02d} (total seconds = {process_time}).")
    return response


@app.get("/")
def health_check() -> HealthCheckResponse:
    """Return a response with a current time to inform that the server is running.

    :return: A JSON Response with a current server time.
    """
    logger.debug("Received a health check GET request.")
    return HealthCheckResponse(now=datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))


def create_input_error_response(
    req_logger: RequestLogger,
    input_error: InputVideoError,
    description_spec: str = None,
) -> JSONResponse:
    """Create a JSON Response with 422 HTTP status code.

    :param req_logger: Request logger which includes request id in log messages.
    :param input_error: Input error enum instance.
    :param description_spec: Additional error specification.
    :return: A JSONResponse with error information.
    """
    out = ErrorResponse(
        error=input_error.description(description_spec),
        error_code=input_error.code(),
    )
    req_logger.warning(f"Returning input error output: {out.dict()}.")
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=out.dict())


@app.post("/", status_code=status.HTTP_201_CREATED)
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(),
    video_parameters: VideoParametersInput = Body(),
):
    """Process video. Detect objects, their tracks, and project to map.

    Request Example:
    ```python
    resp = requests.post(
        "http://127.0.0.1:11433/",
        files={
            "video": open(video_path, "rb"),
        },
        data={
            "video_datetime": "2024-07-03 11:31:27.809041",
            "video_parameters": {
                "data_calibration": {
                    "X": [[1314, 1422], [330, 1192]],
                    "y": [1.047457627118644, 1.042857142857143]
                },
                "data_homography": {
                    "video_points": [[1573, 274], [1887, 159]],
                    "map_points": [[229, 1087], [518, 962]]
                }
            }
        }
    )
    ```
    :param background_tasks: Collection of tasks called after the response to client.
    :param video: A video send as bytes to API.
    :param video_parameters: Parameters of camera calibration and map points.
    :return: A JSON Response with class and confidence or error message.
    """
    # initialized logger with request id
    request_id = str(uuid.uuid4())
    req_logger = RequestLogger(request_id)
    req_logger.info(f"Received POST request with a file. Assigning ID {request_id}.")

    video_dir = "/data/videos"
    os.makedirs(video_dir, exist_ok=True)

    try:
        # Check extension
        video_name = video.filename
        if video_name.lower().split(".")[-1] not in VIDEO_EXTENSIONS:
            req_logger.debug(f"File name: {video_name}, {video.content_type}")
            return create_input_error_response(
                req_logger,
                InputVideoError.NotSupportedExtension,
                description_spec=video_name.split(".")[-1],
            )

        # Video needs to be saved locally to be then used in cv2
        with tempfile.NamedTemporaryFile(dir=video_dir, delete=False) as temp_video:
            temp_video_path = temp_video.name
            temp_video.write(await video.read())

        video_datetime = video_parameters.video_datetime
        if video_datetime is None:
            return create_input_error_response(req_logger, InputVideoError.VideoDatetimeMissing)

        homography_type = video_parameters.homography_type
        if homography_type is None:
            return create_input_error_response(req_logger, InputVideoError.HomographyTypeMissing)
        if homography_type not in HomographyType.values():
            return create_input_error_response(
                req_logger, InputVideoError.NotSupportedHomographyType
            )

        # Validate calibration and homography parameters for tracking
        data_calibration, data_homography = (
            video_parameters.data_calibration,
            video_parameters.data_homography,
        )
        X, y = data_calibration.X, data_calibration.y
        if X is None or y is None:
            return create_input_error_response(req_logger, InputVideoError.CameraCalibrationMissing)
        data_calibration = [X, y]

        video_points, map_points = data_homography.video_points, data_homography.map_points
        if video_points is None or map_points is None:
            return create_input_error_response(req_logger, InputVideoError.HomographyMissing)
        data_homography = [video_points, map_points]

        video_service = get_db_connection().video
        video_id, video_status = video_service.create_video(
            video_datetime=video_datetime, homography_type=homography_type
        )

        worker_service = VideoService(req_logger=req_logger)
        background_tasks.add_task(
            worker_service.process_video,
            video_id,
            temp_video_path,
            data_calibration,
            data_homography,
        )

        # Retrieve result
        return VideoResponse(
            video_id=video_id,
            video_datetime=video_datetime,
            homography_type=homography_type,
            status=video_status,
        )

    except Exception:
        error = traceback.format_exc()
        req_logger.critical(f"Unexpected Error:\n{error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected Error.",
        )


def update_schema_name(function: Callable, name: str):
    """Updates the Pydantic schema name for a FastAPI function with UploadFile parameter.

    The function parameter can be either of `fastapi.UploadFile = File(...)` or `bytes = File(...)`.

    This is a known issue that was reported on FastAPI#1442 in which
    the schema for file upload routes were auto-generated with no
    customization options. This renames the auto-generated schema to
    something more useful and clear.

    Thanks to:
    https://stackoverflow.com/questions/60765317/...
    .../how-to-create-an-openapi-schema-for-an-uploadfile-in-fastapi

    :param function: The function object to modify.
    :param name: The new name of the schema.
    """
    for route in app.routes:
        if route.endpoint is function:
            route.body_field.type_.__name__ = name
            break


# update schema name for OpenAPI documentation in Swagger UI
update_schema_name(process_video, "Video")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ["API_HOST"], port=int(os.environ["API_PORT"]))
