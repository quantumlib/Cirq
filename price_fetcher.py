import requests
import time
from datetime import datetime, timedelta, timezone
import logging

# --- Configuration ---
COINGECKO_API_BASE_URL = "https://api.coingecko.com/api/v3"
# Using 'usd' as it's the common representation for USDC on many platforms like CoinGecko
VS_CURRENCY = "usd" 
CACHE_EXPIRY_SECONDS = 60  # 1 minute cache

# Mapping for common symbols to CoinGecko API IDs
CRYPTO_MAP = {
    "BTC": "bitcoin",
    "SOL": "solana",
}
REVERSE_CRYPTO_MAP = {v: k for k, v in CRYPTO_MAP.items()} # For display purposes

# --- In-Memory Cache ---
# Structure: {(crypto_id, vs_currency): {'price': float, 'timestamp': datetime_object_utc}}
price_cache = {}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _fetch_prices_from_api(crypto_ids: list[str], vs_currencies: list[str]) -> dict | None:
    """
    Fetches prices from the CoinGecko API.
    crypto_ids: List of CoinGecko specific IDs (e.g., ['bitcoin', 'solana'])
    vs_currencies: List of currencies to price against (e.g., ['usd'])
    Returns a dictionary from the API or None if an error occurs.
    """
    if not crypto_ids or not vs_currencies:
        logging.warning("Crypto IDs or VS currencies list is empty.")
        return None

    ids_param = ",".join(crypto_ids)
    vs_param = ",".join(vs_currencies)
    url = f"{COINGECKO_API_BASE_URL}/simple/price?ids={ids_param}&vs_currencies={vs_param}"

    try:
        response = requests.get(url, timeout=10) # 10-second timeout
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
        return response.json()
    except requests.exceptions.Timeout:
        logging.error(f"API request timed out: {url}")
    except requests.exceptions.HTTPError as e:
        logging.error(f"API request failed with HTTPError: {e.status_code} - {e.response.text} for URL: {url}")
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed with RequestException: {e} for URL: {url}")
    except ValueError as e: # For JSON decoding errors
        logging.error(f"Failed to decode JSON response from API: {e} for URL: {url}")
    return None

def get_latest_price(crypto_symbol: str) -> dict | None:
    """
    Gets the latest price for a given crypto symbol (e.g., 'BTC', 'SOL'),
    denominated in VS_CURRENCY (e.g., 'USD').
    Uses an in-memory cache to avoid excessive API calls.

    Returns:
        A dictionary like {'symbol': 'BTC', 'price': 50000.0, 'vs_currency': 'USD', 'last_fetched_utc': 'ISO_FORMAT_DATETIME'}
        or None if the price cannot be fetched or the symbol is invalid.
    """
    if crypto_symbol not in CRYPTO_MAP:
        logging.warning(f"Invalid or unsupported crypto symbol: {crypto_symbol}")
        return None

    coingecko_id = CRYPTO_MAP[crypto_symbol]
    cache_key = (coingecko_id, VS_CURRENCY)
    
    # Check cache first
    cached_data = price_cache.get(cache_key)
    if cached_data:
        if datetime.now(timezone.utc) - cached_data['timestamp'] < timedelta(seconds=CACHE_EXPIRY_SECONDS):
            logging.info(f"Cache hit for {crypto_symbol}/{VS_CURRENCY.upper()}")
            return {
                "symbol": crypto_symbol,
                "price": cached_data['price'],
                "vs_currency": VS_CURRENCY.upper(),
                "last_fetched_utc": cached_data['timestamp'].isoformat()
            }
        else:
            logging.info(f"Cache expired for {crypto_symbol}/{VS_CURRENCY.upper()}")

    # If not in cache or expired, fetch from API
    logging.info(f"Cache miss or expired. Fetching {crypto_symbol}/{VS_CURRENCY.upper()} from API.")
    api_data = _fetch_prices_from_api([coingecko_id], [VS_CURRENCY])

    if api_data and coingecko_id in api_data and VS_CURRENCY in api_data[coingecko_id]:
        price = api_data[coingecko_id][VS_CURRENCY]
        current_time_utc = datetime.now(timezone.utc)
        
        # Update cache
        price_cache[cache_key] = {'price': price, 'timestamp': current_time_utc}
        logging.info(f"Successfully fetched and cached price for {crypto_symbol}: {price} {VS_CURRENCY.upper()}")
        
        return {
            "symbol": crypto_symbol,
            "price": price,
            "vs_currency": VS_CURRENCY.upper(),
            "last_fetched_utc": current_time_utc.isoformat()
        }
    else:
        logging.error(f"Could not retrieve valid price data for {crypto_symbol} from API response: {api_data}")
        # Potentially return stale cache data if preferred, but for now, we return None
        return None

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    print("--- Testing Cryptocurrency Price Fetcher ---")
    
    btc_price_info = get_latest_price("BTC")
    if btc_price_info:
        print(f"Bitcoin Price: ${btc_price_info['price']} {btc_price_info['vs_currency']} (Fetched at: {btc_price_info['last_fetched_utc']})")
    else:
        print("Could not fetch Bitcoin price.")

    sol_price_info = get_latest_price("SOL")
    if sol_price_info:
        print(f"Solana Price: ${sol_price_info['price']} {sol_price_info['vs_currency']} (Fetched at: {sol_price_info['last_fetched_utc']})")
    else:
        print("Could not fetch Solana price.")

    # Test caching - second call should be faster and use cache
    print("\n--- Testing Cache ---")
    time.sleep(2) # Ensure timestamp is different if fetched again
    
    btc_price_info_cached = get_latest_price("BTC")
    if btc_price_info_cached:
        print(f"Bitcoin Price (cached): ${btc_price_info_cached['price']} {btc_price_info_cached['vs_currency']} (Fetched at: {btc_price_info_cached['last_fetched_utc']})")
        assert btc_price_info['last_fetched_utc'] == btc_price_info_cached['last_fetched_utc'] # Timestamp should be the same due to cache
    else:
        print("Could not fetch Bitcoin price (cached).")

    print("\n--- Testing Cache Expiry (will take >60s) ---")
    # To test expiry, you'd ideally mock `time.sleep` or `datetime.now` 
    # or actually wait for CACHE_EXPIRY_SECONDS.
    # For this example, we'll just note it.
    print(f"Waiting for {CACHE_EXPIRY_SECONDS + 5} seconds to test cache expiry...")
    # time.sleep(CACHE_EXPIRY_SECONDS + 5)
    # btc_price_info_expired = get_latest_price("BTC")
    # if btc_price_info_expired:
    #     print(f"Bitcoin Price (after cache expiry): ${btc_price_info_expired['price']} (Fetched at: {btc_price_info_expired['last_fetched_utc']})")
    #     # This new timestamp should be different from the first one if API call was made
    #     if btc_price_info:
    #         assert btc_price_info_expired['last_fetched_utc'] != btc_price_info['last_fetched_utc']
    # else:
    #     print("Could not fetch Bitcoin price (after cache expiry).")

    print("\n--- Testing Invalid Symbol ---")
    invalid_price_info = get_latest_price("XYZ")
    if not invalid_price_info:
        print("Correctly handled invalid symbol XYZ.")

    # Example of how other services might use it:
    # price_data = price_fetcher.get_latest_price("BTC")
    # if price_data:
    #     print(f"The current price of BTC is {price_data['price']} USD, last updated at {price_data['last_fetched_utc']}.")
    # else:
    #     print("Could not retrieve BTC price.")
    print("\n--- Testing Complete ---")
