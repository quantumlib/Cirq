from pydantic import BaseModel, Field

class ImageUploadSuccessResponse(BaseModel):
    message: str = Field(default="Image uploaded successfully")
    image_id: str = Field(..., description="The unique ID (filename) of the uploaded image.")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Image uploaded successfully",
                "image_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef.png"
            }
        }

# For error responses, FastAPI's default HTTPException detail is often sufficient.
# However, if a more structured error response is needed across the API:
# class ErrorDetail(BaseModel):
#     message: str
#     loc: list[str] | None = None # Optional location of the error
#     type: str | None = None      # Optional error type

# class ErrorResponse(BaseModel):
#     detail: ErrorDetail | str
