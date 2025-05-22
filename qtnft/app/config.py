import pathlib
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define the base directory of the application
# For example, if config.py is in qtnft/app/config.py,
# BASE_DIR would be the 'qtnft' directory.
# Adjust as necessary depending on your project structure.
# Assuming this file is in qtnft/app/
APP_DIR = pathlib.Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent # This would be 'qtnft' directory

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "QNFT Service API"
    DEBUG: bool = True # Set to False in production

    # Image Upload Settings
    UPLOAD_DIR: pathlib.Path = BASE_DIR / "uploads" / "images"
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_IMAGE_TYPES: list[str] = ["image/png", "image/jpeg", "image/gif"]

    # Derived settings
    @property
    def MAX_FILE_SIZE_BYTES(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    # Pydantic settings configuration
    model_config = SettingsConfigDict(
        env_file=".env", # Load .env file if it exists
        env_file_encoding='utf-8',
        extra='ignore' # Ignore extra fields from .env
    )

    # GIF Generation Settings
    GENERATED_GIFS_DIR: pathlib.Path = BASE_DIR / "generated_gifs"
    GIF_DURATION_SECONDS: int = 5
    GIF_FPS: int = 10
    CANVAS_WIDTH: int = 500
    CANVAS_HEIGHT: int = 500
    GIF_LOOP: int = 0 # 0 for infinite loop

    # Solana Configuration
    SOLANA_RPC_URL: str = Field(default="http://127.0.0.1:8899", env="SOLANA_RPC_URL")
    SOLANA_COMMITMENT: str = Field(default="confirmed", env="SOLANA_COMMITMENT") # e.g., processed, confirmed, finalized
    SERVICE_PAYER_KEYPAIR_PATH: str = Field(
        default=str(BASE_DIR / "wallets" / "service_payer_wallet.json"),
        env="SERVICE_PAYER_KEYPAIR_PATH"
    )

    # Key Metaplex Program Addresses (Defaults are for Mainnet, Devnet, Localnet)
    METAPLEX_TOKEN_METADATA_PROGRAM_ID: str = Field(default="metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s", env="METAPLEX_TOKEN_METADATA_PROGRAM_ID")
    SPL_ASSOCIATED_TOKEN_ACCOUNT_PROGRAM_ID: str = Field(default="ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL", env="SPL_ASSOCIATED_TOKEN_ACCOUNT_PROGRAM_ID")
    SPL_TOKEN_PROGRAM_ID: str = Field(default="TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA", env="SPL_TOKEN_PROGRAM_ID")

    # NFT Minting Defaults
    NFT_ON_CHAIN_NAME_DEFAULT: str = Field(default="QNFT", env="NFT_ON_CHAIN_NAME_DEFAULT", max_length=32) 
    NFT_ON_CHAIN_SYMBOL_DEFAULT: str = Field(default="QNFT", env="NFT_ON_CHAIN_SYMBOL_DEFAULT", max_length=10)
    NFT_SELLER_FEES_BASIS_POINTS: int = Field(default=500, env="NFT_SELLER_FEES_BASIS_POINTS") # 500 = 5%
    NFT_IS_MUTABLE: bool = Field(default=True, env="NFT_IS_MUTABLE")

    # Admin Fee Configuration for NFT Minting
    ADMIN_WALLET_ADDRESS: Optional[str] = Field(
        default=None, 
        env="ADMIN_WALLET_ADDRESS",
        description="Solana public key of the admin wallet to receive minting fees. If None or empty, fee is disabled."
    )
    ADMIN_MINT_FEE_SOL: float = Field(
        default=0.01, 
        env="ADMIN_MINT_FEE_SOL",
        description="Admin fee in SOL for each NFT mint. Set to 0 to disable if address is present but no fee desired."
    )

    # Derived property for admin fee in lamports
    @property
    def ADMIN_MINT_FEE_LAMPORTS(self) -> int:
        return int(self.ADMIN_MINT_FEE_SOL * 1_000_000_000)

    # Permanent Storage Settings
    STORAGE_PROVIDER: str = Field(default="arweave", env="STORAGE_PROVIDER", pattern="^(arweave|ipfs_pinning_service)$")

    # Arweave Specific Settings
    ARWEAVE_WALLET_PATH: Optional[str] = Field(
        default=str(BASE_DIR / "wallets" / "arweave_wallet.json"),
        env="ARWEAVE_WALLET_PATH"
    )
    ARWEAVE_GATEWAY_URL: str = Field(default="https://arweave.net", env="ARWEAVE_GATEWAY_URL")
    
    # IPFS Pinning Service Specific Settings (Example for Pinata)
    IPFS_PINNING_SERVICE_NAME: Optional[str] = Field(default=None, env="IPFS_PINNING_SERVICE_NAME")
    IPFS_PINNING_API_ENDPOINT: Optional[str] = Field(default=None, env="IPFS_PINNING_API_ENDPOINT")
    IPFS_PINNING_API_KEY: Optional[str] = Field(default=None, env="IPFS_PINNING_API_KEY")
    IPFS_PINNING_API_SECRET: Optional[str] = Field(default=None, env="IPFS_PINNING_API_SECRET")
    IPFS_GATEWAY_URL: str = Field(default="https://ipfs.io", env="IPFS_GATEWAY_URL")

    # Preview Mode GIF Generation Settings
    PREVIEW_GIF_DURATION_SECONDS: int = 2
    PREVIEW_GIF_FPS: int = 5 
    PREVIEW_TRANSFORMATION_DURATION_FRAMES: int = 5 
    PREVIEW_FIB_SEGMENTS: int = 5 
    PREVIEW_GIF_PALETTESIZE: int = 128

    # Transformation settings
    TRANSFORMATION_DURATION_FRAMES: int = Field(default=10, description="Number of frames for placeholder transformation effect")


settings = Settings()

# Ensure relevant directories exist when settings are loaded
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.GENERATED_GIFS_DIR.mkdir(parents=True, exist_ok=True)

# Auto-create wallets directory if default paths are used and directory doesn't exist
# This is for local development convenience. In production, paths should be explicitly managed.
default_sol_wallet_path = str(BASE_DIR / "wallets" / "service_payer_wallet.json")
if settings.SERVICE_PAYER_KEYPAIR_PATH == default_sol_wallet_path:
    pathlib.Path(default_sol_wallet_path).parent.mkdir(parents=True, exist_ok=True)

if settings.ARWEAVE_WALLET_PATH: 
    default_ar_wallet_path = str(BASE_DIR / "wallets" / "arweave_wallet.json")
    if settings.ARWEAVE_WALLET_PATH == default_ar_wallet_path:
        pathlib.Path(default_ar_wallet_path).parent.mkdir(parents=True, exist_ok=True)
    # Placeholder for type hint if not already imported
    from typing import Optional 
    # Need to ensure Field is imported if not already
    from pydantic import Field
