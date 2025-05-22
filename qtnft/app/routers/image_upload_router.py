from fastapi import APIRouter, File, UploadFile, HTTPException, status, Depends
import logging

from ..services import file_handler
from ..models.image_models import ImageUploadSuccessResponse
from ..config import Settings, settings as app_settings # Renamed to avoid conflict

# Configure logging for this router
logger = logging.getLogger(__name__)
# Assuming basicConfig is set in main.py or file_handler.py, otherwise:
# logging.basicConfig(level=logging.INFO)


# Dependency to get settings - useful for testing or if settings are complex
# For simple cases like this, direct import of app_settings is also fine.
def get_settings() -> Settings:
    return app_settings

router = APIRouter()

@router.post(
    "/upload",
    response_model=ImageUploadSuccessResponse,
    summary="Upload an image",
    description="Upload an image file (PNG, JPG, GIF). Max file size: configurable (e.g., 10MB)."
)
async def upload_image_route(
    file: UploadFile = File(..., description="The image file to upload."),
    current_settings: Settings = Depends(get_settings) # Dependency injection for settings
):
    """
    Handles image uploads.

    - Validates file type and size.
    - Saves the file with a unique name to a configured directory.
    - Returns the unique ID (filename) of the saved image.
    """
    logger.info(f"Received file upload request for: {file.filename}, content type: {file.content_type}, size: {file.size}")

    # Perform validation using the service function
    try:
        await file_handler.validate_upload_file(
            file=file,
            max_size_bytes=current_settings.MAX_FILE_SIZE_BYTES,
            allowed_content_types=current_settings.ALLOWED_IMAGE_TYPES
        )
        logger.info(f"File '{file.filename}' passed validation.")
    except HTTPException as e:
        logger.warning(f"Validation failed for '{file.filename}': {e.detail}")
        raise  # Re-raise the HTTPException from validation
    except Exception as e: # Catch any other unexpected validation error
        logger.error(f"Unexpected error during validation for '{file.filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during file validation: {str(e)}"
        )
    
    # Save the file using the service function
    try:
        # Ensure the UPLOAD_DIR from settings is used.
        # The file_handler.save_upload_file can take destination_dir as an argument.
        # The config.py already ensures UPLOAD_DIR exists.
        unique_filename = await file_handler.save_upload_file(
            upload_file=file, # The UploadFile object itself
            destination_dir=current_settings.UPLOAD_DIR
        )
        logger.info(f"File '{file.filename}' successfully processed and saved as '{unique_filename}'.")
        
        return ImageUploadSuccessResponse(
            message="Image uploaded successfully.",
            image_id=unique_filename
        )
    except HTTPException as e:
        # Re-raise HTTPExceptions that might come from the save_upload_file service
        logger.error(f"HTTPException during file save for '{file.filename}': {e.detail}")
        raise e
    except Exception as e:
        # Catch any other unexpected errors from the saving process
        logger.error(f"Unexpected error during file save for '{file.filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while saving the image: {str(e)}"
        )

# --- Example of how to include this router in main.py (already done in main.py) ---
# from fastapi import FastAPI
# from .routers import image_upload_router
# 
# app = FastAPI()
# app.include_router(image_upload_router.router, prefix="/api/v1/images", tags=["Image Uploads"])
