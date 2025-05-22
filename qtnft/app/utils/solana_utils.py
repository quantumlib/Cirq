import json
import logging
from typing import List, Optional # Optional for type hinting cache
from solders.keypair import Keypair
from ..config import settings # Assuming this path is correct relative to execution

logger = logging.getLogger(__name__)

# In-memory cache for the loaded keypair to avoid repeated file reads
_service_payer_keypair_cache: Optional[Keypair] = None

def get_service_payer_keypair() -> Keypair:
    """
    Loads the service payer keypair from the path specified in settings.
    Caches the loaded keypair in memory to avoid repeated file reads.

    The keypair file is expected to be a JSON array of 64 numbers (uint8 values),
    which is the typical output of `solana-keygen new --outfile key.json`.

    Raises:
        FileNotFoundError: If the keypair file is not found.
        ValueError: If the keypair file format is incorrect, path is not configured,
                    or data length is unexpected.

    Returns:
        solders.keypair.Keypair: The loaded service payer keypair.
    """
    global _service_payer_keypair_cache
    if _service_payer_keypair_cache:
        return _service_payer_keypair_cache

    keypair_path_str = settings.SERVICE_PAYER_KEYPAIR_PATH
    if not keypair_path_str:
        logger.error("SERVICE_PAYER_KEYPAIR_PATH is not configured in settings.")
        raise ValueError("Service payer keypair path is not configured.")

    try:
        logger.debug(f"Attempting to load service payer keypair from: {keypair_path_str}")
        with open(keypair_path_str, 'r') as f:
            # solana-keygen stores the keypair as a JSON array of 64 numbers (bytes).
            key_data_json: List[int] = json.load(f)
        
        if not isinstance(key_data_json, list) or not all(isinstance(x, int) for x in key_data_json):
            raise ValueError("Keypair JSON file content is not a list of integers.")

        key_data_bytes = bytes(key_data_json)

        # The standard output from `solana-keygen new --outfile ...` is a 64-byte array.
        # `Keypair.from_bytes()` expects this 64-byte array (secret key + public key).
        if len(key_data_bytes) == 64:
            keypair = Keypair.from_bytes(key_data_bytes)
        # Some older or alternative formats might store only the 32-byte seed (private key part).
        # `Keypair.from_seed()` is used for a 32-byte seed.
        elif len(key_data_bytes) == 32: # Less common for solana-keygen JSON files directly
            logger.warning("Loading keypair from a 32-byte file; assuming it's a seed (private key).")
            keypair = Keypair.from_seed(key_data_bytes)
        else:
            raise ValueError(
                f"Keypair file at '{keypair_path_str}' contains data of unexpected length: {len(key_data_bytes)} bytes. "
                "Expected 64 bytes (standard solana-keygen output) or 32 bytes (seed/private key only)."
            )
        
        logger.info(f"Successfully loaded service payer keypair. Public Key: {keypair.pubkey()}")
        _service_payer_keypair_cache = keypair
        return _service_payer_keypair_cache
    
    except FileNotFoundError:
        logger.error(f"Service payer keypair file not found at the configured path: {keypair_path_str}")
        raise FileNotFoundError(f"Keypair file not found at {keypair_path_str}. Ensure SERVICE_PAYER_KEYPAIR_PATH is set correctly and the file exists.")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in keypair file '{keypair_path_str}': {e}")
        raise ValueError(f"Could not parse JSON from keypair file '{keypair_path_str}'. Ensure it's a valid JSON array of numbers.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the service payer keypair from '{keypair_path_str}': {e}", exc_info=True)
        raise ValueError(f"An unexpected error occurred while loading the keypair: {str(e)}")

# --- Example usage for standalone testing of this utility ---
# Note: This requires `qtnft.app.config.settings` to be importable and configured,
# and a dummy wallet file to exist at the configured path for a successful test.
async def main_test_keypair_loading():
    logging.basicConfig(level=logging.DEBUG) # Use DEBUG to see more logs from this util
    
    # For testing, you might need to ensure a dummy wallet file exists.
    # This part is tricky to run standalone without the FastAPI app context or proper project setup.
    # It's better to test this as part of integration tests or by running the app.
    
    # Example: Create a dummy wallet if one doesn't exist at the default path for local testing
    from ..config import BASE_DIR # Ensure this import works based on your execution context
    import os
    
    default_wallet_path_str = str(BASE_DIR / "wallets" / "service_payer_wallet.json")
    
    if settings.SERVICE_PAYER_KEYPAIR_PATH == default_wallet_path_str:
        dummy_wallet_dir = os.path.dirname(default_wallet_path_str)
        if not os.path.exists(dummy_wallet_dir):
            os.makedirs(dummy_wallet_dir)
            logger.info(f"Created dummy wallet directory for testing: {dummy_wallet_dir}")

        if not os.path.exists(default_wallet_path_str):
            try:
                kp_new = Keypair() # Generate a new one for testing
                # Save its full 64-byte representation as a JSON list of numbers
                key_bytes_list = list(bytes(kp_new)) 
                with open(default_wallet_path_str, "w") as f:
                   json.dump(key_bytes_list, f)
                logger.info(f"Created dummy wallet file at {default_wallet_path_str} for testing.")
            except Exception as e:
                logger.error(f"Could not create dummy wallet for testing: {e}")
                # Do not proceed if dummy wallet creation fails and it's the default path
                return
        else:
            logger.info(f"Using existing wallet file at {default_wallet_path_str} for testing.")

    try:
        loaded_kp = get_service_payer_keypair()
        logger.info(f"TEST SUCCESS: Loaded Keypair. PubKey: {loaded_kp.pubkey()}")
        
        # Test caching
        loaded_kp_cached = get_service_payer_keypair()
        assert loaded_kp is loaded_kp_cached, "Cache test failed: Keypair objects are different."
        logger.info("TEST SUCCESS: Keypair caching works.")

    except Exception as e:
        logger.error(f"TEST FAILED: Error in keypair loading test: {e}", exc_info=True)

if __name__ == "__main__":
    # This allows running `python -m qtnft.app.utils.solana_utils` from the project root
    # It requires the environment to be set up such that `qtnft.app.config` can be imported.
    # You might need to adjust PYTHONPATH or run as part of a larger test suite.
    # asyncio.run(main_test_keypair_loading()) # No async needed for get_service_payer_keypair
    
    # Simplified direct call for the synchronous function for basic testing structure
    # However, `main_test_keypair_loading` creates a dummy wallet, which is useful.
    # Let's make it runnable if needed.
    pass # `main_test_keypair_loading` can be called from elsewhere if needed.
    # To run the test:
    # 1. Ensure your PYTHONPATH is set up if running from a different directory.
    #    (e.g., `export PYTHONPATH=$PYTHONPATH:/path/to/your/qtnft_project`)
    # 2. Execute: `python -c "from qtnft.app.utils.solana_utils import main_test_keypair_loading; import asyncio; asyncio.run(main_test_keypair_loading())"`
    #    (Adjust import path if your structure is different)
    # This direct execution is complex due to relative imports and config dependency.
    # It's better tested via integration tests that load the FastAPI app or specific services.
