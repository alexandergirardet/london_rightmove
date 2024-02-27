from pydantic import BaseModel, ValidationError, validator


class Coordinates(BaseModel):
    longitude: float
    latitude: float
