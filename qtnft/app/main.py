from fastapi import FastAPI, APIRouter
from fastapi.staticfiles import StaticFiles # For serving static files
import logging
from contextlib import asynccontextmanager # Required for lifespan

from .routers import image_upload_router, gif_generation_router, nft_minting_router # Import new router
from .config import settings
from .services import price_fetcher_service # For lifespan events of httpx client

# --- Logging Configuration ---
# Configure logging early, before other modules might try to use it.
# Adjust level and format as needed.
# logging.basicConfig( # Commented out to allow Uvicorn to control logging if preferred
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
# Use this instead for basic config if not using Uvicorn's log config
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

logger = logging.getLogger(__name__)


# --- Lifespan Events for HTTPX client in Price Fetcher ---
# These ensure the httpx.AsyncClient is initialized at startup and closed at shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI): # app parameter is conventional for lifespan managers
    logger.info("Application startup: Initializing HTTPX client for price fetcher.")
    await price_fetcher_service.get_http_client() # Initializes the client
    try:
        yield
    finally:
        logger.info("Application shutdown: Closing HTTPX client for price fetcher.")
        await price_fetcher_service.close_http_client()


app = FastAPI(
    lifespan=lifespan, # Add lifespan context manager
    title=settings.APP_NAME,
    version="0.1.0", # Consider making this configurable
    description="API for QNFT project: Image Upload, GIF Generation, Price Fetching, and NFT Minting.",
    debug=settings.DEBUG
)

# --- Static Files Mounting ---
# Serve generated GIFs. This allows URLs like 'http://localhost:8000/static/gifs/your_gif_id.gif'
# The `gif_url` field in BasicGifResponse should align with this path.
# Note: For production, serving static files directly via FastAPI is not recommended for performance.
# Use a reverse proxy (Nginx) or CDN instead.
static_gifs_path = "/static/gifs" # Define path to avoid magic strings
app.mount(static_gifs_path, StaticFiles(directory=settings.GENERATED_GIFS_DIR), name="generated_gifs")
logger.info(f"Serving generated GIFs statically from directory: {settings.GENERATED_GIFS_DIR} at {static_gifs_path}")

# Potentially serve uploaded images if needed for direct access (consider security)
# static_uploads_path = "/static/uploads"
# app.mount(static_uploads_path, StaticFiles(directory=settings.UPLOAD_DIR), name="uploaded_images")
# logger.info(f"Serving uploaded images statically from directory: {settings.UPLOAD_DIR} at {static_uploads_path}")


# --- Routers ---
api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(image_upload_router.router, prefix="/images", tags=["Image Management"])
api_v1_router.include_router(gif_generation_router.router, prefix="/gifs", tags=["GIF Generation"])
api_v1_router.include_router(nft_minting_router.router, prefix="/nfts", tags=["NFT Minting"]) # Add NFT Minting router

app.include_router(api_v1_router)

@app.get("/", tags=["Root"], summary="API Root Information")
async def read_root():
    """Provides a welcome message and a link to the API documentation."""
    return {"message": "Welcome to the QNFT API. Visit /docs for interactive API documentation."}

# --- Further Setup (Examples for a real app) ---
# if settings.USE_CORS:
#     from fastapi.middleware.cors import CORSMiddleware
#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=["*"], # Or specific origins
#         allow_credentials=True,
#         allow_methods=["*"],
#         allow_headers=["*"],
#     )

# if settings.DATABASE_URL:
#     # Initialize database connections here
#     pass

logger.info(f"Application '{settings.APP_NAME}' initialized. Debug mode: {settings.DEBUG}")
