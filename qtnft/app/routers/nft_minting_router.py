from fastapi import APIRouter, HTTPException, status, Depends
import logging

from ..services import solana_nft_service
from ..models.nft_models import BasicNftMintRequest, BasicNftMintResponse
from ..config import Settings, settings as app_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency to get settings
def get_settings() -> Settings:
    return app_settings

@router.post(
    "/mint_basic",
    response_model=BasicNftMintResponse,
    summary="Mint a basic QNFT (Metaplex NFT)",
    description=(
        "Creates a new Metaplex Non-Fungible Token (NFT) on the Solana blockchain. "
        "This involves creating a new mint, an associated token account for the recipient, "
        "minting one token, and creating Metaplex metadata and master edition accounts. "
        "The 'metadata_uri' is a placeholder for this basic version."
    )
)
async def mint_basic_nft_route(
    request_data: BasicNftMintRequest,
    current_settings: Settings = Depends(get_settings) # Inject settings if needed by router/service
):
    logger.info(
        f"Received request to mint basic NFT for gif_id: {request_data.gif_id}, "
        f"metadata_uri: {request_data.metadata_uri}, recipient: {request_data.recipient_address}"
    )

    try:
        # Ensure settings are loaded if solana_nft_service relies on them directly
        # (which it does via from ..config import settings)
        # No need to pass current_settings explicitly if service imports it directly.
        
        nft_mint_address, tx_signature = await solana_nft_service.mint_basic_nft(
            gif_id=request_data.gif_id,
            metadata_uri=request_data.metadata_uri,
            recipient_address_str=request_data.recipient_address
        )
        
        logger.info(f"Basic NFT minted successfully. Mint Address: {nft_mint_address}, Tx Signature: {tx_signature}")
        return BasicNftMintResponse(
            message="Basic NFT minted successfully",
            nft_mint_address=nft_mint_address,
            transaction_signature=tx_signature
        )
    except ValueError as e: # Catch specific errors like invalid recipient address
        logger.warning(f"ValueError during NFT minting request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RuntimeError as e: # Catch errors from the threaded Solana execution
        logger.error(f"RuntimeError during NFT minting (Solana interaction failed): {e}", exc_info=True)
        # These are likely 500 errors as they indicate issues with the minting process itself
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NFT minting failed due to a server-side or blockchain error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during basic NFT minting: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during NFT minting: {str(e)}"
        )
