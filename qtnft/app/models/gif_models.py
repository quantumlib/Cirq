from pydantic import BaseModel, Field

class BasicGifRequest(BaseModel):
    image_id: str = Field(
        ..., 
        description="The ID (filename) of the previously uploaded image to be used for GIF generation.",
        example="a1b2c3d4-e5f6-7890-1234-567890abcdef.png"
    )

class BasicGifResponse(BaseModel):
    message: str = Field(default="Basic GIF generated successfully")
    gif_id: str = Field(..., description="The unique ID (filename) of the generated GIF.")
    gif_url: str = Field(..., description="The relative URL to access the generated GIF.", example="/gifs/a1b2c3d4-e5f6-7890-1234-567890abcdef.gif")


    class Config:
        json_schema_extra = {
            "example": {
                "message": "Basic GIF generated successfully",
                "gif_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef.gif",
                "gif_url": "/gifs/a1b2c3d4-e5f6-7890-1234-567890abcdef.gif"
            }
        }
