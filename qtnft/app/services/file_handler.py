import uuid
import pathlib
import shutil
from fastapi import UploadFile, HTTPException, status
import aiofiles # For async file operations
import logging

from ..config import settings

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_file_extension(filename: str) -> str:
    """Safely extracts the file extension."""
    return pathlib.Path(filename).suffix.lower().lstrip('.')

async def validate_upload_file(
    file: UploadFile,
    max_size_bytes: int,
    allowed_content_types: list[str]
):
    """
    Validates the uploaded file based on its content type and size.
    Raises HTTPException if validation fails.
    """
    # Validate content type
    if file.content_type not in allowed_content_types:
        logger.warning(f"Upload failed: Unsupported file type '{file.content_type}'. Allowed: {allowed_content_types}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Allowed types are: {', '.join(allowed_content_types)}"
        )

    # Validate file size
    # Read the file in chunks to check size without loading entirely into memory if too large.
    # UploadFile.file is a SpooledTemporaryFile.
    # Checking its actual size before saving is crucial.
    # One way is to seek to the end, get the position, then seek back.
    
    # For UploadFile, a common way is to read it to determine its size.
    # If file.size is available and reliable from Starlette, it can be used.
    # Let's assume we need to be careful and check by reading.
    # However, for simplicity in this conceptual phase, we'll trust file.size if available,
    # or read it. FastAPI/Starlette's UploadFile.size should be populated.
    
    # If file.size is not available or to be absolutely sure:
    # current_pos = file.file.tell()
    # file.file.seek(0, 2) # Seek to end
    # file_size = file.file.tell()
    # file.file.seek(current_pos) # Reset to original position

    # Simpler approach for now, assuming file.size is populated by Starlette after upload
    # or that reading the file to check size is acceptable for the given max_size.
    # For very large files, streaming and checking size chunk by chunk before writing
    # to disk is better.
    
    # Let's try to get the size efficiently.
    # The `UploadFile` object has a `size` attribute that should be populated by Starlette.
    if file.size is None: # Fallback if size is not directly available
        contents = await file.read()
        file_size = len(contents)
        await file.seek(0) # Reset file pointer after reading
    else:
        file_size = file.size
        
    if file_size > max_size_bytes:
        logger.warning(f"Upload failed: File size {file_size} bytes exceeds limit of {max_size_bytes} bytes.")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {file_size / (1024*1024):.2f}MB exceeds the limit of {max_size_bytes / (1024*1024):.2f}MB."
        )


async def save_upload_file(
    upload_file: UploadFile,
    destination_dir: pathlib.Path
) -> str:
    """
    Saves the uploaded file to the specified directory with a unique name.
    Returns the unique filename.
    Raises HTTPException if saving fails.
    """
    if not upload_file.filename:
        # This case should ideally be caught by FastAPI if file is mandatory,
        # but good to have a check.
        logger.error("Upload failed: No filename provided with the upload.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided.")

    # Ensure destination directory exists (should be done at app startup by config, but check again)
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    original_extension = get_file_extension(upload_file.filename)
    if not original_extension or original_extension not in [ct.split('/')[-1] for ct in settings.ALLOWED_IMAGE_TYPES if '/' in ct]:
        # Fallback or stricter check if extension is not in allowed types (e.g. jpeg vs jpg)
        # For simplicity, derive from content type if possible or use a default
        content_type_ext_map = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif"}
        original_extension = content_type_ext_map.get(upload_file.content_type, "dat") # .dat as a generic fallback

    unique_filename = f"{uuid.uuid4()}.{original_extension}"
    destination_path = destination_dir / unique_filename

    try:
        # Save the file asynchronously
        # Important: UploadFile.read() reads the whole file into memory.
        # For large files, UploadFile.file.read() in chunks or shutil.copyfileobj is better.
        # aiofiles helps perform the write operation asynchronously.
        async with aiofiles.open(destination_path, 'wb') as out_file:
            # Read chunk by chunk to avoid loading large files into memory at once
            while content := await upload_file.read(1024 * 1024):  # Read 1MB chunks
                await out_file.write(content)
        
        logger.info(f"File '{upload_file.filename}' successfully saved as '{unique_filename}' to '{destination_path}'")
        return unique_filename
    except IOError as e:
        logger.error(f"IOError during file save: {e}. Failed to save '{unique_filename}'.")
        # Attempt to remove partially written file if error occurs
        if destination_path.exists():
            try:
                pathlib.Path.unlink(destination_path)
            except Exception as unlink_e:
                logger.error(f"Could not remove partially written file {destination_path}: {unlink_e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save file due to an internal error."
        )
    except Exception as e:
        logger.error(f"Unexpected error during file save: {e}. Failed to save '{unique_filename}'.")
        if destination_path.exists(): # Check again in case of other errors
             try:
                pathlib.Path.unlink(destination_path)
             except Exception as unlink_e:
                logger.error(f"Could not remove partially written file {destination_path}: {unlink_e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    finally:
        await upload_file.close()


# --- Notes for Future Enhancement ---
# 1. Object Storage Integration:
#    - Replace local saving logic with client calls to AWS S3, Google Cloud Storage, etc.
#    - `unique_filename` might then become the object key or a URL.
#
# 2. Database Integration:
#    - After successful upload (local or object store), save metadata to a DB:
#      (image_id, original_filename, storage_path, content_type, size, upload_timestamp, user_id)
#    - The `image_id` returned to the client would be the DB record's primary key.
#
# 3. Advanced File Type Verification:
#    - Use libraries like `python-magic` to verify file content against its magic numbers,
#      as `Content-Type` header can be spoofed.
#      Example:
#      import magic
#      mime_type = magic.from_buffer(await upload_file.read(2048), mime=True) # Read first 2KB
#      await upload_file.seek(0) # Reset pointer
#      if mime_type not in settings.ALLOWED_IMAGE_TYPES:
#          raise HTTPException(...)
#
# 4. More Granular Size Check (Streaming):
#    If `UploadFile.size` is not trusted or for very large files and strict memory control,
#    read from `upload_file.file` (the SpooledTemporaryFile) in chunks, summing chunk lengths,
#    and raising HTTP_413_REQUEST_ENTITY_TOO_LARGE if limit exceeded *before* writing to final destination.
#    This is partly implemented in save_upload_file by reading in chunks. The validation function
#    currently relies on UploadFile.size or a full read if size is None.
#
# 5. Security - Filename Extension:
#    The current `get_file_extension` is basic. For production, ensure the derived extension
#    is robustly checked against a strict allow-list and doesn't contain malicious parts.
#    Using the content_type to determine extension as a fallback is a good safety measure.
#    Example: allowed_extensions = {"png", "jpg", "jpeg", "gif"}
#    if original_extension not in allowed_extensions: handle error or use default.
#    The current code uses a map from content_type for this as a safer default.

"""
Example of how this service might be used in a router:

from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from ..services import file_handler
from ..models.image_models import ImageUploadSuccessResponse
from ..config import settings
import pathlib

router = APIRouter()

@router.post("/upload", response_model=ImageUploadSuccessResponse)
async def upload_image_route(file: UploadFile = File(...)):
    await file_handler.validate_upload_file(
        file=file,
        max_size_bytes=settings.MAX_FILE_SIZE_BYTES,
        allowed_content_types=settings.ALLOWED_IMAGE_TYPES
    )
    
    try:
        unique_filename = await file_handler.save_upload_file(
            upload_file=file,
            destination_dir=settings.UPLOAD_DIR
        )
        return ImageUploadSuccessResponse(image_id=unique_filename)
    except HTTPException as e:
        # Re-raise HTTPExceptions from the service layer
        raise e
    except Exception as e:
        # Catch any other unexpected errors from saving
        logger.error(f"Unexpected error in upload_image_route: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the image."
        )
"""
