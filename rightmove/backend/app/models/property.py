from pydantic import BaseModel, ValidationError, validator


class Property(BaseModel):
    bedrooms: float
    bathrooms: float
    longitude: float
    latitude: float
    walk_score: float
