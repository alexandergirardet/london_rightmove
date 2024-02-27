from pydantic import BaseModel, ValidationError, validator


class PricingCategory(BaseModel):
    category: str

    # Optional: Validator to provide a more specific error message
    @validator("category")
    def check_category(cls, v):
        if v not in ["Cheap", "Average", "Expensive"]:
            raise ValidationError('Pricing must be "Cheap", "Average", or "Expensive"')
        return v
