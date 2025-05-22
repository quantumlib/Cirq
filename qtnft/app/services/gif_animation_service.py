import asyncio
import uuid
import pathlib
from PIL import Image, ImageOps, ImageDraw # Pillow for image manipulation
import imageio # For GIF generation
import numpy as np # For converting PIL Images to format imageio likes
import logging

from ..config import settings

logger = logging.getLogger(__name__)
# Ensure basicConfig is set, e.g. in main.py or here if testing standalone
# logging.basicConfig(level=logging.INFO)

def _create_placeholder_animation_frames(
    base_image: Image.Image,
    num_frames: int,
    canvas_width: int,
    canvas_height: int
) -> list[Image.Image]:
    """
    Generates a list of PIL Image objects for a simple animation.
    Example: Slight continuous zoom in and out (ping-pong).
    """
    frames = []
    
    # Resize base_image to fit canvas while maintaining aspect ratio, then paste onto canvas bg
    # This ensures all frames start from a consistently sized and placed image.
    img_copy = base_image.copy()
    img_copy.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
    
    # Create a background canvas (e.g., transparent or a solid color)
    # Using RGBA for potential transparency in frames.
    background = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 0)) # Transparent white
    
    # Calculate position to center the image
    paste_x = (canvas_width - img_copy.width) // 2
    paste_y = (canvas_height - img_copy.height) // 2
    
    prepared_image = background.copy()
    prepared_image.paste(img_copy, (paste_x, paste_y), img_copy if img_copy.mode == 'RGBA' else None)

    for i in range(num_frames):
        # Create a new frame based on the prepared image
        frame_image = prepared_image.copy()
        
        # --- Simple Animation Example: Pulsating Zoom (Zoom in then out) ---
        # Calculate zoom factor: 1.0 -> 1.1 -> 1.0 (ping-pong)
        # Normalize i to a value between 0 and 1 for the first half, then 1 to 0 for the second
        # This creates a zoom-in then zoom-out effect
        half_frames = num_frames / 2.0
        if i < half_frames:
            progress = i / half_frames # 0 to 1
        else:
            progress = (num_frames - i -1) / half_frames # 1 to 0 (almost)

        min_zoom = 1.0
        max_zoom = 1.05 # Slight zoom to 105%
        current_zoom = min_zoom + (max_zoom - min_zoom) * progress
        
        zoomed_width = int(frame_image.width * current_zoom)
        zoomed_height = int(frame_image.height * current_zoom)
        
        # Resize the content (not the whole frame_image, but what's on it)
        # For simplicity, we'll zoom the prepared_image then crop/paste
        content_to_zoom = prepared_image.copy() # The image already centered on transparent bg
        zoomed_content = content_to_zoom.resize((zoomed_width, zoomed_height), Image.Resampling.LANCZOS)
        
        # Calculate new position to keep it centered after zoom
        new_x = (canvas_width - zoomed_content.width) // 2
        new_y = (canvas_height - zoomed_content.height) // 2
        
        # Create final frame: new background, paste zoomed content
        final_frame = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 0))
        final_frame.paste(zoomed_content, (new_x, new_y), zoomed_content if zoomed_content.mode == 'RGBA' else None)
        
        # Optional: Add a frame number for debugging
        # draw = ImageDraw.Draw(final_frame)
        # draw.text((10, 10), f"Frame {i+1}/{num_frames}", fill="black")

        frames.append(final_frame.convert("RGB")) # Convert to RGB for GIF if no transparency needed, or ensure palette handles RGBA
        
    logger.info(f"Generated {len(frames)} frames for animation.")
    return frames

async def generate_basic_animation_gif(image_id: str) -> str:
    """
    Loads an image, generates a simple animation, saves it as a GIF,
    and returns the GIF's filename.
    All blocking operations (file I/O, image processing) are run in threads.
    """
    image_path = settings.UPLOAD_DIR / image_id

    # --- 1. Validate image_id and Load Image (Blocking) ---
    def _load_and_validate_image():
        # Basic sanitization/check for image_id format (though UUIDs are generally safe)
        # For example, ensure it doesn't contain path traversal characters.
        # As image_id is from our upload service, it should be relatively safe.
        if not image_path.is_file():
            logger.error(f"Image file not found at path: {image_path}")
            raise FileNotFoundError(f"Image with ID '{image_id}' not found.")
        
        try:
            img = Image.open(image_path)
            img.load() # Ensure image data is loaded
            logger.info(f"Image '{image_id}' loaded successfully from '{image_path}'.")
            return img
        except Exception as e:
            logger.error(f"Failed to load image '{image_id}': {e}")
            raise ValueError(f"Could not load or process image '{image_id}'.")

    try:
        base_image = await asyncio.to_thread(_load_and_validate_image)
    except FileNotFoundError as e:
        raise # Re-raise to be caught by router as 404
    except ValueError as e:
        raise # Re-raise to be caught by router as 400/500

    # --- 2. Animation Parameters & Frame Generation (Blocking) ---
    num_frames = settings.GIF_DURATION_SECONDS * settings.GIF_FPS
    
    try:
        pil_frames = await asyncio.to_thread(
            _create_placeholder_animation_frames,
            base_image,
            num_frames,
            settings.CANVAS_WIDTH,
            settings.CANVAS_HEIGHT
        )
    except Exception as e: # Catch errors from frame generation
        logger.error(f"Error during frame generation for '{image_id}': {e}")
        raise ValueError(f"Failed to generate animation frames: {str(e)}")

    if not pil_frames:
        logger.error(f"No frames generated for image '{image_id}'. Cannot create GIF.")
        raise ValueError("Animation frame generation resulted in zero frames.")

    # --- 3. Convert PIL Frames to NumPy arrays (Blocking) ---
    def _convert_frames_to_numpy(frames_pil: list[Image.Image]) -> list[np.ndarray]:
        logger.info(f"Converting {len(frames_pil)} PIL frames to NumPy arrays.")
        # Imageio prefers NumPy arrays. Convert RGB PIL images.
        return [np.array(frame.convert("RGB")) for frame in frames_pil] 

    try:
        np_frames = await asyncio.to_thread(_convert_frames_to_numpy, pil_frames)
    except Exception as e:
        logger.error(f"Error converting frames to NumPy arrays for '{image_id}': {e}")
        raise ValueError(f"Failed to convert frames for GIF processing: {str(e)}")


    # --- 4. GIF Generation using Imageio (Blocking & CPU/Memory Intensive) ---
    gif_filename = f"{uuid.uuid4()}.gif"
    gif_filepath = settings.GENERATED_GIFS_DIR / gif_filename

    def _save_gif():
        try:
            imageio.mimsave(
                gif_filepath,
                np_frames,
                fps=settings.GIF_FPS,
                loop=settings.GIF_LOOP, # 0 for infinite loop
                subrectangles=True, # Optimization
                palettesize=256 # Standard for GIFs, can affect quality/size
            )
            logger.info(f"GIF for image '{image_id}' saved as '{gif_filename}' to '{gif_filepath}'.")
            return gif_filename
        except Exception as e:
            logger.error(f"Failed to save GIF '{gif_filename}': {e}")
            # Attempt to remove partially written file if error occurs
            if gif_filepath.exists():
                try:
                    gif_filepath.unlink()
                except Exception as unlink_e:
                    logger.error(f"Could not remove partially written GIF {gif_filepath}: {unlink_e}")
            raise ValueError(f"Failed to generate and save GIF: {str(e)}")

    try:
        saved_gif_name = await asyncio.to_thread(_save_gif)
        return saved_gif_name
    except ValueError as e:
        raise # Re-raise to be caught by router as 500


# --- Notes for Future/Production ---
# 1. Celery/Background Tasks: For a production environment, `generate_basic_animation_gif`
#    (especially the frame generation and imageio.mimsave parts) should be offloaded
#    to a background worker (e.g., Celery) to prevent blocking API server resources
#    and to handle potentially long processing times. The API would return a task ID.
#
# 2. More Advanced Animations: The `_create_placeholder_animation_frames` is where
#    more complex animation logic would go (e.g., integrating the "quantum effect").
#
# 3. Resource Limits: Implement checks for image dimensions or complexity if needed
#    to prevent excessive memory/CPU usage, especially if not using background workers.
#
# 4. Input Validation: More robust validation of `image_id` if it's not guaranteed
#    to be a system-generated UUID (e.g., checking for directory traversal, allowed characters).

# --- Example for direct testing (if needed) ---
# async def main_test_gif_service():
#     # Prerequisite: An image must exist in settings.UPLOAD_DIR
#     # For example, create a dummy image there named "test_image.png"
#     # Ensure settings.UPLOAD_DIR and settings.GENERATED_GIFS_DIR are correctly pointing
#     # to existing or creatable directories.
#     logging.basicConfig(level=logging.INFO)
#     test_image_name = "YOUR_TEST_IMAGE_IN_UPLOAD_DIR.png" # Replace this
    
#     # Create a dummy file for testing if it doesn't exist
#     dummy_image_path = settings.UPLOAD_DIR / test_image_name
#     if not dummy_image_path.exists():
#         try:
#             img = Image.new('RGB', (100, 100), color = 'red')
#             img.save(dummy_image_path)
#             logger.info(f"Created dummy test image: {dummy_image_path}")
#         except Exception as e:
#             logger.error(f"Could not create dummy test image: {e}")
#             return

#     try:
#         logger.info(f"Attempting to generate GIF for image_id: {test_image_name}")
#         gif_id = await generate_basic_animation_gif(test_image_name)
#         logger.info(f"Successfully generated GIF: {gif_id}. Stored in: {settings.GENERATED_GIFS_DIR / gif_id}")
#     except FileNotFoundError:
#         logger.error(f"Test image '{test_image_name}' not found. Please place it in '{settings.UPLOAD_DIR}'.")
#     except ValueError as e:
#         logger.error(f"ValueError during GIF generation test: {e}")
#     except Exception as e:
#         logger.error(f"An unexpected error occurred during GIF generation test: {e}", exc_info=True)

# if __name__ == "__main__":
#     # This setup is needed to run the async main_test_gif_service
#     # Ensure your config.py and directory structure are compatible.
#     # from qtnft.app.config import settings # Adjust import if necessary
    
#     # Manually ensure directories for test if not handled by config loading outside FastAPI app
#     # settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
#     # settings.GENERATED_GIFS_DIR.mkdir(parents=True, exist_ok=True)
    
#     asyncio.run(main_test_gif_service())
