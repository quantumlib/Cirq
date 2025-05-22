from datetime import datetime, timezone

# --- Default Configuration ---
# These can be overridden by passing a config dict to the preparation function
DEFAULT_CONFIG = {
    "symbol": "QNFT",
    "seller_fee_basis_points": 500, # 5%
    "external_url": "https://example.com/qnft-project", # Project's external URL
    "creator_address": "YOUR_SERVICE_WALLET_ADDRESS_HERE", # Replace with actual service wallet
    "creator_share": 100,
    "category": "image", # Metaplex category
    "base_description": "A unique Quantum NFT, capturing live cryptocurrency prices at the moment of its creation, transformed by a quantum-resistant inspired algorithm."
}

def prepare_qnft_metadata(
    gif_url: str,
    gif_content_type: str = "image/gif",
    btc_price_info: dict = None, # {'price': float, 'timestamp_utc': 'ISO_STRING'}
    sol_price_info: dict = None, # {'price': float, 'timestamp_utc': 'ISO_STRING'}
    qr_algorithm_details: dict = None, # {'name': str, 'version': str}
    nft_mint_timestamp_utc: str = None,
    nft_name_base: str = "Quantum NFT",
    nft_number: int = None,
    override_config: dict = None
) -> dict:
    """
    Prepares the NFT metadata dictionary for a QNFT, conforming to Metaplex standards.

    Args:
        gif_url (str): Permanent URL/CID of the generated GIF.
        gif_content_type (str): MIME type of the GIF (default: "image/gif").
        btc_price_info (dict): BTC price data {'price': float, 'timestamp_utc': 'ISO_STRING'}.
        sol_price_info (dict): SOL price data {'price': float, 'timestamp_utc': 'ISO_STRING'}.
        qr_algorithm_details (dict): QR algorithm details {'name': str, 'version': str}.
        nft_mint_timestamp_utc (str): ISO 8601 UTC timestamp of NFT mint.
        nft_name_base (str): Base name for the NFT.
        nft_number (int, optional): Unique number for this NFT if part of a series.
        override_config (dict, optional): Dictionary to override default config values.

    Returns:
        dict: The NFT metadata as a Python dictionary.
    
    Raises:
        ValueError: If required inputs like gif_url are missing or price_info is malformed.
    """
    if not gif_url:
        raise ValueError("gif_url is required.")

    config = DEFAULT_CONFIG.copy()
    if override_config:
        config.update(override_config)

    # --- Dynamic Naming ---
    if nft_number is not None:
        name = f"{nft_name_base} #{nft_number:03d}" # e.g., Quantum NFT #001
    else:
        name = nft_name_base

    # --- Mint Timestamp ---
    if not nft_mint_timestamp_utc:
        nft_mint_timestamp_utc = datetime.now(timezone.utc).isoformat()
    
    # --- Description ---
    description = config["base_description"]
    if qr_algorithm_details and qr_algorithm_details.get('name'):
        description += f" Algorithm: {qr_algorithm_details['name']} v{qr_algorithm_details.get('version', 'N/A')}."
    description += f" Created on: {nft_mint_timestamp_utc}."

    # --- Attributes ---
    attributes = []
    if btc_price_info:
        if not isinstance(btc_price_info, dict) or 'price' not in btc_price_info or 'timestamp_utc' not in btc_price_info:
            raise ValueError("Invalid btc_price_info format. Expected {'price': float, 'timestamp_utc': 'ISO_STRING'}")
        attributes.append({"trait_type": "BTC Price (USD)", "value": str(btc_price_info['price'])})
        attributes.append({"trait_type": "BTC Price Timestamp", "value": btc_price_info['timestamp_utc']})
    
    if sol_price_info:
        if not isinstance(sol_price_info, dict) or 'price' not in sol_price_info or 'timestamp_utc' not in sol_price_info:
            raise ValueError("Invalid sol_price_info format. Expected {'price': float, 'timestamp_utc': 'ISO_STRING'}")
        attributes.append({"trait_type": "SOL Price (USD)", "value": str(sol_price_info['price'])})
        attributes.append({"trait_type": "SOL Price Timestamp", "value": sol_price_info['timestamp_utc']})

    if qr_algorithm_details:
        if not isinstance(qr_algorithm_details, dict) or 'name' not in qr_algorithm_details or 'version' not in qr_algorithm_details:
            raise ValueError("Invalid qr_algorithm_details format. Expected {'name': str, 'version': str}")
        attributes.append({"trait_type": "QR Algorithm", "value": qr_algorithm_details['name']})
        attributes.append({"trait_type": "QR Algorithm Version", "value": qr_algorithm_details['version']})
    
    attributes.append({"trait_type": "Mint Timestamp (UTC)", "value": nft_mint_timestamp_utc})

    # --- Metadata Dictionary ---
    metadata = {
        "name": name,
        "symbol": config["symbol"],
        "description": description,
        "seller_fee_basis_points": config["seller_fee_basis_points"],
        "image": gif_url,
        "animation_url": gif_url, # Important for animated content like GIFs
        "external_url": config["external_url"],
        "attributes": attributes,
        "properties": {
            "files": [
                {"uri": gif_url, "type": gif_content_type}
            ],
            "category": config["category"],
            "creators": [
                {
                    "address": config["creator_address"],
                    # Verified is typically set to False in metadata JSON.
                    # The on-chain Metaplex program marks it true if the creator address signs the transaction.
                    "verified": False, 
                    "share": config["creator_share"]
                }
            ]
        }
    }
    return metadata

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    print("--- Testing NFT Metadata Preparation Service ---")

    # Example Inputs
    test_gif_url = "ar://example_gif_cid_or_url"
    test_btc_data = {"price": 60000.75, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    test_sol_data = {"price": 150.25, "timestamp_utc": datetime(2023, 10, 27, 12, 0, 0, tzinfo=timezone.utc).isoformat()}
    test_qr_data = {"name": "QRA_Placeholder", "version": "1.1"}
    mint_time = datetime(2023, 10, 28, 0, 0, 0, tzinfo=timezone.utc).isoformat()

    # Basic metadata
    metadata1 = prepare_qnft_metadata(
        gif_url=test_gif_url,
        btc_price_info=test_btc_data,
        sol_price_info=test_sol_data,
        qr_algorithm_details=test_qr_data,
        nft_mint_timestamp_utc=mint_time,
        nft_name_base="My Quantum Art",
        nft_number=1
    )
    import json
    print("\nMetadata for NFT #1:")
    print(json.dumps(metadata1, indent=2))

    # Metadata with custom config
    custom_cfg = {
        "symbol": "MQNFT",
        "seller_fee_basis_points": 1000, # 10%
        "creator_address": "ANOTHER_WALLET_ADDRESS_HERE",
        "external_url": "https://myart.io/collection"
    }
    metadata2 = prepare_qnft_metadata(
        gif_url="ipfs://another_gif_cid",
        btc_price_info=test_btc_data,
        sol_price_info=test_sol_data,
        qr_algorithm_details=test_qr_data,
        nft_mint_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        nft_name_base="Special Edition QNFT",
        override_config=custom_cfg
    )
    print("\nMetadata for Special Edition NFT (no number, custom config):")
    print(json.dumps(metadata2, indent=2))

    # Test missing required input
    try:
        prepare_qnft_metadata(gif_url=None)
    except ValueError as e:
        print(f"\nSuccessfully caught error for missing gif_url: {e}")

    # Test malformed price input
    try:
        prepare_qnft_metadata(gif_url=test_gif_url, btc_price_info={"price": 123}) # Missing timestamp
    except ValueError as e:
        print(f"\nSuccessfully caught error for malformed btc_price_info: {e}")

    print("\n--- Testing Complete ---")
