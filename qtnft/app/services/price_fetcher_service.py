import httpx
import asyncio
from datetime import datetime, timedelta, timezone
import logging
from typing import Dict, List, Optional, Any

from ..config import settings # Assuming settings will have relevant configs

# --- Configuration (can be moved to or augmented by settings from config.py) ---
COINGECKO_API_BASE_URL = getattr(settings, "COINGECKO_API_URL", "https://api.coingecko.com/api/v3")
# For CoinGecko, 'usd' is the common representation for USDC-like stablecoins
VS_CURRENCY = getattr(settings, "PRICE_VS_CURRENCY", "usd")
CACHE_EXPIRY_SECONDS = getattr(settings, "PRICE_CACHE_EXPIRY_SECONDS", 60) # 1 minute

# Cryptocurrencies to fetch (CoinGecko IDs)
CRYPTO_IDS_TO_FETCH = getattr(settings, "PRICE_CRYPTO_IDS", ["bitcoin", "solana"])
# Standardized pair names for service output (e.g., BTC_USD, SOL_USD)
# This map helps to translate from coingecko IDs to a common naming convention.
CRYPTO_ID_TO_PAIR_NAME_MAP = {
    "bitcoin": "BTC_USD",
    "solana": "SOL_USD",
}

# --- Logging Setup ---
logger = logging.getLogger(__name__)
# Assuming basicConfig is set elsewhere (e.g., main.py or via Uvicorn config)
# If not, logging.basicConfig(level=logging.INFO) here

# --- In-Memory Cache ---
# Structure: {"BTC_USD": {'price': float, 'timestamp_utc': datetime, 'source_id': 'bitcoin_usd'}}
price_cache: Dict[str, Dict[str, Any]] = {}
cache_lock = asyncio.Lock()

# --- HTTPX Client ---
# It's good practice to use a single client instance for connection pooling
http_client: Optional[httpx.AsyncClient] = None

async def get_http_client() -> httpx.AsyncClient:
    """Manages the lifecycle of the httpx.AsyncClient."""
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=10) # 10-second default timeout
    return http_client

async def close_http_client():
    """Closes the httpx.AsyncClient."""
    global http_client
    if http_client:
        await http_client.aclose()
        http_client = None

# --- Helper Functions ---
def _is_cache_valid(pair_name: str) -> bool:
    """Checks if the cache for a given pair is still valid."""
    cached_item = price_cache.get(pair_name)
    if not cached_item:
        return False
    age = datetime.now(timezone.utc) - cached_item['timestamp_utc']
    return age < timedelta(seconds=CACHE_EXPIRY_SECONDS)

async def _fetch_prices_from_api(
    client: httpx.AsyncClient,
    crypto_ids: List[str],
    vs_currencies: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Fetches prices from the CoinGecko API.
    Returns the API response JSON or None if an error occurs.
    """
    if not crypto_ids or not vs_currencies:
        logger.warning("Crypto IDs or VS currencies list is empty for API fetch.")
        return None

    ids_param = ",".join(crypto_ids)
    vs_param = ",".join(vs_currencies)
    url = f"{COINGECKO_API_BASE_URL}/simple/price?ids={ids_param}&vs_currencies={vs_param}&include_last_updated_at=true"

    try:
        response = await client.get(url)
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
        api_data = response.json()
        logger.info(f"Successfully fetched prices from CoinGecko for IDs: {ids_param}")
        return api_data
    except httpx.TimeoutException:
        logger.error(f"API request timed out: {url}")
    except httpx.HTTPStatusError as e:
        logger.error(f"API request failed with HTTPError: {e.response.status_code} - {e.response.text} for URL: {url}")
    except httpx.RequestError as e:
        logger.error(f"API request failed with RequestError: {e} for URL: {url}")
    except ValueError as e: # For JSON decoding errors
        logger.error(f"Failed to decode JSON response from API: {e} for URL: {url}")
    return None

# --- Public Service Interface ---
async def get_current_prices() -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Gets the latest prices for configured cryptocurrencies (e.g., BTC/USD, SOL/USD).
    Uses an in-memory cache to avoid excessive API calls.

    Returns:
        A dictionary mapping standardized pair names (e.g., "BTC_USD") to their price data.
        Price data includes: {'price': float, 'last_updated_utc': 'ISO_FORMAT_DATETIME', 'source_id': 'coingecko_id_vs_currency'}
        Returns None for a pair if its price cannot be fetched or an error occurs.
    """
    output_prices: Dict[str, Optional[Dict[str, Any]]] = {}
    needs_api_fetch_ids: List[str] = []
    client = await get_http_client()

    async with cache_lock:
        for coingecko_id in CRYPTO_IDS_TO_FETCH:
            pair_name = CRYPTO_ID_TO_PAIR_NAME_MAP.get(coingecko_id)
            if not pair_name:
                logger.warning(f"No pair name mapping for CoinGecko ID: {coingecko_id}")
                continue

            if _is_cache_valid(pair_name):
                output_prices[pair_name] = price_cache[pair_name]
                logger.info(f"Cache hit for {pair_name}")
            else:
                logger.info(f"Cache miss or expired for {pair_name}")
                needs_api_fetch_ids.append(coingecko_id)
                output_prices[pair_name] = price_cache.get(pair_name) # Keep stale data for now if available

    if needs_api_fetch_ids:
        logger.info(f"Fetching from API for: {needs_api_fetch_ids}")
        api_data = await _fetch_prices_from_api(client, needs_api_fetch_ids, [VS_CURRENCY])

        async with cache_lock:
            if api_data:
                for coingecko_id in needs_api_fetch_ids:
                    pair_name = CRYPTO_ID_TO_PAIR_NAME_MAP.get(coingecko_id)
                    if not pair_name: continue

                    if coingecko_id in api_data and VS_CURRENCY in api_data[coingecko_id]:
                        price = api_data[coingecko_id][VS_CURRENCY]
                        # CoinGecko provides last_updated_at as Unix timestamp
                        last_updated_unix = api_data[coingecko_id].get(f"{VS_CURRENCY}_last_updated_at", time.time())
                        last_updated_dt = datetime.fromtimestamp(last_updated_unix, tz=timezone.utc)
                        
                        current_data = {
                            "price": price,
                            "last_updated_utc": last_updated_dt.isoformat(),
                            "source_id": f"{coingecko_id}_{VS_CURRENCY}", # e.g. bitcoin_usd
                            "source_api": "CoinGecko"
                        }
                        price_cache[pair_name] = current_data
                        output_prices[pair_name] = current_data
                        logger.info(f"Updated cache for {pair_name} with new API data.")
                    else:
                        logger.warning(f"Could not find price for {coingecko_id}_{VS_CURRENCY} in API response. API response: {api_data.get(coingecko_id)}")
                        # If it was a cache miss and API failed, it will remain None or stale.
                        # If it was stale, it remains stale. If it was None, it remains None.
                        # If we want to explicitly mark it as failed:
                        # output_prices[pair_name] = {"error": "Failed to fetch from API", "price": price_cache.get(pair_name, {}).get('price')}

            else: # API fetch failed for all needed IDs
                logger.error(f"API fetch failed for IDs: {needs_api_fetch_ids}. Output prices will contain stale or None values.")
                # output_prices will contain stale data if it was there, or None.

    # Ensure all requested pairs are in the output, even if fetching failed and no stale data
    for cg_id in CRYPTO_IDS_TO_FETCH:
        pn = CRYPTO_ID_TO_PAIR_NAME_MAP.get(cg_id)
        if pn and pn not in output_prices:
            output_prices[pn] = None # Explicitly set to None if no data at all

    return output_prices

# --- Lifespan Integration (for FastAPI app) ---
# To be used with FastAPI's lifespan events to initialize/close the client
# @app.on_event("startup")
# async def startup_event():
#     await get_http_client() # Initialize client
#
# @app.on_event("shutdown")
# async def shutdown_event():
#     await close_http_client() # Close client

# --- Example Usage (for direct testing of this module) ---
async def main_test():
    import time
    print("--- Testing Cryptocurrency Price Fetcher Service ---")
    
    # Initialize client for testing
    await get_http_client()

    print("\n--- First fetch (should be API call) ---")
    prices1 = await get_current_prices()
    for pair, data in prices1.items():
        if data:
            print(f"{pair}: Price=${data['price']}, Last Updated (UTC): {data['last_updated_utc']}")
        else:
            print(f"{pair}: Could not fetch price.")

    print("\n--- Second fetch (should be cache hit for valid items) ---")
    await asyncio.sleep(2) # Ensure time difference for next potential fetch is minimal
    prices2 = await get_current_prices()
    for pair, data in prices2.items():
        if data:
            print(f"{pair}: Price=${data['price']}, Last Updated (UTC): {data['last_updated_utc']} (Cache Test)")
            # Assert that timestamp is same as first fetch for cached items
            if prices1.get(pair) and prices1[pair]['last_updated_utc'] != data['last_updated_utc']:
                 # This might happen if coingecko updates its data in between, even within seconds.
                 # The key is that an API call wasn't made if cache was valid.
                 logger.warning(f"Timestamp mismatch for {pair} on cache test, could be due to CoinGecko update or logic error.")
        else:
            print(f"{pair}: Could not fetch price. (Cache Test)")

    print(f"\n--- Testing Cache Expiry (will take ~{CACHE_EXPIRY_SECONDS}s) ---")
    # For actual test, you'd mock time or wait. We'll just simulate passage of time in mind.
    print(f"Simulating wait for {CACHE_EXPIRY_SECONDS + 5} seconds...")
    # In a real test, you might `await asyncio.sleep(CACHE_EXPIRY_SECONDS + 5)`
    # For now, we'll manually clear cache to simulate expiry for one item if possible for testing
    # This is a bit hacky for a direct test, normally you'd control time via mocks.
    if "BTC_USD" in price_cache:
        price_cache["BTC_USD"]['timestamp_utc'] -= timedelta(seconds=CACHE_EXPIRY_SECONDS + 10)
        logger.info("Manually expired BTC_USD cache for testing.")
    
    print("\n--- Third fetch (BTC_USD should fetch from API, SOL_USD might be cached or API if also expired) ---")
    prices3 = await get_current_prices()
    for pair, data in prices3.items():
        if data:
            print(f"{pair}: Price=${data['price']}, Last Updated (UTC): {data['last_updated_utc']} (Expiry Test)")
        else:
            print(f"{pair}: Could not fetch price. (Expiry Test)")

    # Clean up client
    await close_http_client()
    print("\n--- Testing Complete ---")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    # Add a simple stream handler for console output during direct testing
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s'))
    logger.addHandler(console_handler)
    
    asyncio.run(main_test())
