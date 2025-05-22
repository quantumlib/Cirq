from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.responses import FileResponse # For serving the GIF if needed directly
import logging

from ..services import gif_animation_service
from ..models.gif_models import BasicGifRequest, BasicGifResponse
from ..config import Settings, settings as app_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency to get settings
def get_settings() -> Settings:
    return app_settings

@router.post(
    "/generate_basic",
    response_model=BasicGifResponse,
    summary="Generate a basic animated GIF",
    description="Takes an image_id of a previously uploaded image and generates a simple animated GIF."
)
async def generate_basic_gif_route(
    request_data: BasicGifRequest,
    current_settings: Settings = Depends(get_settings) # Inject settings
):
    logger.info(f"Received request to generate basic GIF for image_id: {request_data.image_id}")

    try:
        gif_filename = await gif_animation_service.generate_basic_animation_gif(
            image_id=request_data.image_id
        )
        
        # Construct a relative URL for the GIF.
        # This assumes you'll have a static route set up to serve files from GENERATED_GIFS_DIR.
        # For example, if GENERATED_GIFS_DIR is 'qtnft/generated_gifs' and it's served at '/gifs',
        # then the URL would be '/gifs/your_gif_id.gif'.
        # If main.py serves GENERATED_GIFS_DIR at app.mount("/gifs", StaticFiles(directory=settings.GENERATED_GIFS_DIR)...)
        gif_url_path = f"/gifs/{gif_filename}" # Relative path

        logger.info(f"Basic GIF '{gif_filename}' generated successfully for image_id: {request_data.image_id}.")
        return BasicGifResponse(
            message="Basic GIF generated successfully",
            gif_id=gif_filename,
            gif_url=gif_url_path
        )
    except FileNotFoundError:
        logger.warning(f"Image not found for GIF generation: image_id '{request_data.image_id}'")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image with ID '{request_data.image_id}' not found."
        )
    except ValueError as e: # Catch specific ValueErrors from the service (e.g., processing issues)
        logger.error(f"ValueError during GIF generation for '{request_data.image_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # Or 400 if it's more like a bad input image
            detail=f"Failed to generate GIF: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during GIF generation for '{request_data.image_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while generating the GIF: {str(e)}"
        )

# Optional: Add a route to serve the generated GIFs if you don't want to rely solely on Nginx/CDN
# This requires StaticFiles to be mounted in main.py or this router to handle it.
# If GENERATED_GIFS_DIR is mounted as "/gifs" at the app level:
@router.get(
    "/{gif_id}",
    summary="Get a generated GIF",
    description="Serves a previously generated GIF file by its ID.",
    response_class=FileResponse # This will stream the file
)
async def get_generated_gif(
    gif_id: str,
    current_settings: Settings = Depends(get_settings)
):
    gif_filepath = current_settings.GENERATED_GIFS_DIR / gif_id
    
    # Basic security check: ensure gif_id is just a filename and not trying path traversal
    if not gif_id or "/" in gif_id or ".." in gif_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid gif_id format.")

    if not gif_filepath.is_file():
        logger.warning(f"Requested GIF not found: {gif_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="GIF not found.")
    
    # Determine media type for GIF
    media_type = "image/gif"
    
    logger.info(f"Serving GIF: {gif_id}")
    return FileResponse(path=gif_filepath, media_type=media_type, filename=gif_id)


# To make the /gifs/{gif_id} route work, you'd typically mount static files in main.py:
# from fastapi.staticfiles import StaticFiles
# app.mount("/static-gifs", StaticFiles(directory=settings.GENERATED_GIFS_DIR), name="generated_gifs")
# Then the BasicGifResponse.gif_url could be "/static-gifs/your_gif_id.gif"
# The route above provides a dynamic way if you prefer not to mount the whole directory,
# or want more control (e.g. auth checks before serving).
# For simplicity, the BasicGifResponse returns a /gifs/... path assuming a root mount.
# If using the dynamic route above, the prefix of this router matters.
# If this router is at /api/v1/gifs, then the url would be /api/v1/gifs/{gif_id}
# The BasicGifResponse.gif_url should reflect the actual accessible URL.
# Let's assume for now the gif_url in BasicGifResponse is a placeholder path
# that the frontend/client knows how to interpret, or a static mount is configured.
# The example GET route above is provided for completeness if dynamic serving is desired.
