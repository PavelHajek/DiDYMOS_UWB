from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    now: str


class ErrorResponse(BaseModel):
    error: str
    error_code: int
