from fastapi import FastAPI

from . import video


def include_routers(app: FastAPI):
    """Include all routers defined in this module to the FastAPI application instance."""
    app.include_router(video.router)
