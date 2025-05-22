import logging
import asyncio
from typing import Dict, Any, List, Optional # Optional added
from solders.pubkey import Pubkey
from solana.rpc.api import Client as SolanaClient # Sync client
# from solana.rpc.async_api import AsyncClient as AsyncSolanaClient # Alternative
from solana.rpc.commitment import Confirmed # For commitment level

from ..config import settings

logger = logging.getLogger(__name__)

# Helper function to get a Solana client instance
def _get_solana_client() -> SolanaClient:
    # In a real app, client management might be more sophisticated (e.g., shared instance via lifespan)
    # For now, creating a new client per high-level service call that needs it.
    return SolanaClient(settings.SOLANA_RPC_URL, commitment=Confirmed) # Use Confirmed or other from settings

async def _get_wallet_sol_balance(wallet_address: str, solana_client: SolanaClient) -> float:
    """Fetches SOL balance for a given wallet address."""
    logger.debug(f"Fetching SOL balance for wallet: {wallet_address}")
    try:
        wallet_pubkey = Pubkey.from_string(wallet_address)
        
        def sync_get_balance_call(): # Wrapper for to_thread
            return solana_client.get_balance(wallet_pubkey).value
            
        balance_lamports = await asyncio.to_thread(sync_get_balance_call)
        balance_sol = balance_lamports / 1_000_000_000.0 # Ensure float division
        logger.debug(f"Wallet {wallet_address} SOL balance: {balance_sol:.6f}")
        return balance_sol
    except ValueError: # Invalid pubkey string format
        logger.warning(f"Invalid wallet_address format for SOL balance check: {wallet_address}")
        return 0.0 # Or raise specific error
    except Exception as e:
        logger.error(f"Error fetching SOL balance for {wallet_address}: {e}", exc_info=True)
        return 0.0 # Default to 0 on error, or re-raise

async def _get_wallet_qnft_holdings(
    wallet_address: str, 
    solana_client: SolanaClient # Passed for consistency, though placeholder doesn't use it much
) -> Dict[str, Any]:
    """
    Placeholder/Conceptual: Simulates fetching QNFT holdings and checking for VIP Pass NFTs.
    WARNING: Direct on-chain querying for all tokens held by a wallet and then filtering
    is EXTREMELY INEFFICIENT and NOT SUITABLE FOR PRODUCTION without an indexer.
    This placeholder returns simulated data.
    """
    logger.debug(f"Fetching QNFT/VIP Pass holdings for wallet: {wallet_address} (Placeholder Logic)")
    
    # --- THIS IS WHERE INDEXER INTEGRATION IS CRITICAL FOR PRODUCTION ---
    # Real implementation would query an indexer (e.g., Helius, Moralis, Alchemy, etc.)
    # For example:
    # indexer_client = YourIndexerSDKClient(api_key=settings.INDEXER_API_KEY)
    # user_nfts = await indexer_client.get_nfts_for_owner(wallet_address)
    # qnft_count = 0
    # holds_vip_pass = False
    # if settings.QNFT_COLLECTION_IDENTIFIER:
    #     qnft_count = sum(1 for nft in user_nfts if nft.collection_mint == settings.QNFT_COLLECTION_IDENTIFIER)
    # for vip_mint_str in settings.VIP_PASS_NFT_MINTS:
    #     if any(nft.mint == vip_mint_str for nft in user_nfts):
    #         holds_vip_pass = True
    #         break
    # return {"qnft_count": qnft_count, "holds_vip_pass": holds_vip_pass}

    # --- Simple Placeholder Logic (NOT FOR PRODUCTION) ---
    await asyncio.sleep(0.05) # Simulate slight delay of an API call

    qnft_count = 0
    holds_vip_pass = False
    
    # Simulate some wallets having VIP passes or QNFTs for testing
    # This is very basic and just for demonstration.
    if "vip_test_wallet" in wallet_address.lower(): # Example: "MyVipTestWalletADDRESS..."
        holds_vip_pass = True
        qnft_count = settings.TIER_PRO_MIN_QNFT_COUNT + 2 # Ensure Pro if VIP, plus some more
    elif "pro_test_wallet" in wallet_address.lower(): # Example: "MyProTestWalletADDRESS..."
        qnft_count = settings.TIER_PRO_MIN_QNFT_COUNT
    elif "basic_test_wallet_with_few_qnft" in wallet_address.lower():
        qnft_count = settings.TIER_PRO_MIN_QNFT_COUNT -1 if settings.TIER_PRO_MIN_QNFT_COUNT > 0 else 0
        
    logger.info(f"Placeholder holdings for {wallet_address}: QNFTs={qnft_count}, HoldsVIPPass={holds_vip_pass}")
    return {"qnft_count": qnft_count, "holds_vip_pass": holds_vip_pass}

async def determine_user_tier(wallet_address: str) -> Dict[str, Any]:
    """
    Determines the user's tier based on configured criteria (SOL balance, QNFT count, VIP Pass).
    """
    logger.info(f"Determining user tier for wallet: {wallet_address}")
    # It's important to manage the lifecycle of the SolanaClient appropriately.
    # Creating it here means it's per-call. For higher load, a shared client (e.g., via lifespan) is better.
    solana_client = _get_solana_client() 

    # Fetch data concurrently (SOL balance and holdings)
    # Note: _get_wallet_qnft_holdings is a placeholder and currently fast.
    # If it were a real network call, concurrent execution would be more impactful.
    sol_balance_task = _get_wallet_sol_balance(wallet_address, solana_client)
    holdings_task = _get_wallet_qnft_holdings(wallet_address, solana_client)
    
    sol_balance, holdings = await asyncio.gather(sol_balance_task, holdings_task)
    
    qnft_count = holdings.get("qnft_count", 0)
    holds_vip_pass = holdings.get("holds_vip_pass", False)

    current_tier_name = settings.TIER_BASIC_NAME
    tier_reason = "Default Basic Tier"

    # Tier Logic: VIP takes precedence, then Pro, then Basic.
    # Check for VIP Pass first (primary way to get VIP)
    if holds_vip_pass and settings.VIP_PASS_NFT_MINTS: # Check if VIP passes are configured and user holds one
        current_tier_name = settings.TIER_VIP_NAME
        tier_reason = "Holds a VIP Pass NFT."
    # (Optional: Add secondary VIP path here if defined, e.g., high SOL and QNFT count)
    # elif sol_balance >= settings.TIER_VIP_MIN_SOL_BALANCE and qnft_count >= settings.TIER_VIP_MIN_QNFT_COUNT:
    #    current_tier_name = settings.TIER_VIP_NAME
    #    tier_reason = "Meets high SOL balance and QNFT count for VIP."
    
    # If not VIP, check for Pro
    elif current_tier_name == settings.TIER_BASIC_NAME: # Only check for Pro if not already VIP
        is_pro_by_sol = sol_balance >= settings.TIER_PRO_MIN_SOL_BALANCE
        # Pro by QNFT count only if collection ID is configured (meaning we can identify QNFTs)
        is_pro_by_qnft = settings.QNFT_COLLECTION_IDENTIFIER is not None and \
                         qnft_count >= settings.TIER_PRO_MIN_QNFT_COUNT

        if is_pro_by_sol or is_pro_by_qnft:
            current_tier_name = settings.TIER_PRO_NAME
            if is_pro_by_sol and is_pro_by_qnft:
                tier_reason = "Meets SOL balance and QNFT count for Pro Tier."
            elif is_pro_by_sol:
                tier_reason = "Meets SOL balance for Pro Tier."
            else: # is_pro_by_qnft
                tier_reason = "Meets QNFT count for Pro Tier."
    
    tier_details = {
        "sol_balance_sol": round(sol_balance, 4),
        "qnft_count_checked": qnft_count if settings.QNFT_COLLECTION_IDENTIFIER else "N/A (No Collection ID)",
        "holds_vip_pass_checked": holds_vip_pass,
        "reason": tier_reason
    }
            
    logger.info(f"Wallet {wallet_address} assigned tier: {current_tier_name}. Details: {tier_details}")
    
    # Structure for the API response
    return {
        "wallet_address": wallet_address,
        "tier": current_tier_name,
        "details": tier_details,
        "criteria_checked": { # Information about what criteria were applied
            "pro_tier_min_sol_balance": settings.TIER_PRO_MIN_SOL_BALANCE,
            "pro_tier_min_qnft_count": settings.TIER_PRO_MIN_QNFT_COUNT,
            "vip_pass_nft_mints_configured": len(settings.VIP_PASS_NFT_MINTS),
            "qnft_collection_identifier_configured": settings.QNFT_COLLECTION_IDENTIFIER is not None
        }
    }

# Example test function (can be run with `python -m qtnft.app.services.user_tier_service`)
# async def main_test_tier_service():
#     logging.basicConfig(level=logging.DEBUG) # Use DEBUG to see more logs
#     test_wallets = [
#         "some_basic_address_for_testing12345", # Should be Basic
#         "pro_test_wallet_meets_sol_criteriaABC", # Placeholder should give Pro by SOL (if SOL balance mocked or criteria low)
#         "pro_test_wallet_meets_qnft_criteriaXYZ", # Placeholder should give Pro by QNFT
#         "vip_test_wallet_has_vip_passDEF" # Placeholder should give VIP
#     ]

#     # Mocking settings for test if needed, or ensure .env is set up
#     # settings.TIER_PRO_MIN_SOL_BALANCE = 0.0001 # Lower for testing SOL balance part
#     # settings.QNFT_COLLECTION_IDENTIFIER = "TestCollectionMintAddress" # Enable QNFT counting
#     # settings.VIP_PASS_NFT_MINTS = ["AVIPPassMintAddress111111111111111111111111"] # Enable VIP pass checking

#     for wallet in test_wallets:
#         logger.info(f"\n--- Determining tier for: {wallet} ---")
#         tier_info = await determine_user_tier(wallet)
#         logger.info(f"Tier Info for {wallet}: {json.dumps(tier_info, indent=2)}")
#         # Add assertions here if writing actual tests

# if __name__ == "__main__":
#     # This requires qtnft.app.config to be importable.
#     # Example: python -m qtnft.app.services.user_tier_service (from project root)
#     import json # For pretty printing in test
#     asyncio.run(main_test_tier_service())
