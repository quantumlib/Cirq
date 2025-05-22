import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path # For sequence file path management

from ..config import settings # For defaults like symbol, seller_fee, creator address, project_base_url
from ..utils.solana_utils import get_service_payer_keypair # To get the service wallet's public key

logger = logging.getLogger(__name__)

# --- Sequence ID Management (Placeholder for local development) ---
# In a production environment, this should be replaced by a robust database sequence (e.g., PostgreSQL SERIAL or Redis INCR).
_sequence_file_path = settings.BASE_DIR / "run_data" / "qnft_sequence_id.txt"

def _get_next_sequence_id_placeholder() -> int:
    """
    Placeholder for generating a unique sequence ID.
    Reads from a local file, increments, and writes back.
    WARNING: Not safe for concurrent access (race conditions possible).
    """
    _sequence_file_path.parent.mkdir(parents=True, exist_ok=True)
    current_id = 0
    try:
        with _sequence_file_path.open("r+") as f:
            content = f.read().strip()
            if content:
                current_id = int(content)
            next_id = current_id + 1
            f.seek(0)
            f.write(str(next_id))
            f.truncate()
            logger.info(f"Generated QNFT Sequence ID (placeholder): {next_id}")
            return next_id
    except FileNotFoundError:
        # File doesn't exist, create it with 1
        with _sequence_file_path.open("w") as f:
            f.write("1")
            logger.info("Initialized QNFT Sequence ID file. First ID: 1")
            return 1
    except Exception as e:
        logger.error(f"Critical error managing sequence ID file '{_sequence_file_path}': {e}. "
                       "NFT generation will likely fail or have duplicate IDs.", exc_info=True)
        # Return a potentially problematic ID or raise an exception to halt.
        # For robustness in a critical path, raising might be better.
        raise RuntimeError(f"Failed to get or update sequence ID from file: {e}")


# --- Helper to generate descriptive title ---
def _generate_descriptive_title(
    price_info: Dict[str, Dict[str, Any]], 
    price_movement_info: Optional[Dict[str, Any]], 
    applied_effects_summary: Dict[str, Any]
) -> str:
    """
    Generates a short descriptive highlight for the NFT name based on market data or effects.
    """
    # Priority to significant price movement
    if price_movement_info and price_movement_info.get("symbol"):
        symbol_display = price_movement_info["symbol"].replace("_USD", "")
        direction = price_movement_info.get("direction")
        percent_change = price_movement_info.get("percent_change", 0.0)

        # Define thresholds for what's "significant"
        if direction == "up" and percent_change >= settings.SIGNIFICANT_PRICE_CHANGE_THRESHOLD_UPPER: # e.g. 1.0 (%)
            return f"{symbol_display} Surge"
        elif direction == "down" and percent_change <= settings.SIGNIFICANT_PRICE_CHANGE_THRESHOLD_LOWER: # e.g. -1.0 (%)
            return f"{symbol_display} Dip"
        # Add more nuanced checks if desired, e.g., for stability or smaller notable changes

    # Fallback to Quantum Art Style if prominent
    qi_style_info = applied_effects_summary.get("quantum_inspired_art", {})
    qi_style_name = qi_style_info.get("style_name")
    if qi_style_name:
        # Convert snake_case to Title Case
        return qi_style_name.replace("_", " ").title() 
            
    # Generic fallback
    return "Market Snapshot"


async def prepare_final_nft_metadata(
    permanent_gif_url: str,
    price_info: Dict[str, Dict[str, Any]],
    applied_effects_summary: Dict[str, Any],
    pqc_processed_data: Dict[str, Any], # From PqcCryptographyService
    generation_timestamp_utc: str,
    qnft_sequence_id: int, # Assume sequence_id is now passed in by orchestrator
    price_movement_info: Optional[Dict[str, Any]] = None,
    original_image_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Constructs the final NFT JSON metadata dictionary according to Metaplex standards.
    """
    logger.info(f"Preparing final NFT metadata for QNFT ID #{qnft_sequence_id}, GIF URL: {permanent_gif_url}")

    descriptive_title = _generate_descriptive_title(price_info, price_movement_info, applied_effects_summary)
    # Ensure name length is within Metaplex on-chain limits if this name were to be used there directly.
    # However, for off-chain metadata, there's more flexibility.
    # The on-chain name is typically shorter and set from settings.NFT_ON_CHAIN_NAME_DEFAULT.
    nft_name_off_chain = f"QTNFT #{qnft_sequence_id:05d} - {descriptive_title}"

    # --- Build Description ---
    desc_parts = [
        "A unique Quantum-Time NFT (QTNFT) capturing a snapshot of market dynamics and transforming it into algorithmic art.",
        f"This QTNFT reflects Bitcoin at ${price_info.get('BTC_USD', {}).get('price', 'N/A')} "
        f"and Solana at ${price_info.get('SOL_USD', {}).get('price', 'N/A')} "
        f"as recorded on {generation_timestamp_utc}.",
    ]
    effect_descs = []
    if applied_effects_summary.get("fibonacci_animation"):
        fib_type = applied_effects_summary['fibonacci_animation'].get('type', 'Fibonacci')
        effect_descs.append(f"a '{fib_type.replace('_', ' ').title()}' animation")
    if applied_effects_summary.get("quantum_inspired_art"):
        qi_name = applied_effects_summary['quantum_inspired_art'].get('style_name', 'Quantum Art')
        effect_descs.append(f"the '{qi_name.replace('_', ' ').title()}' style")
    
    if effect_descs:
        desc_parts.append(f"It is visualized with {', and '.join(effect_descs)}.")
    
    adv_transform_info = applied_effects_summary.get("advanced_transformation_placeholder", {})
    if adv_transform_info.get("transformed_regions_count", 0) > 0:
        target_placeholder = adv_transform_info.get("region_1", {}).get("target_type_placeholder", "an object")
        desc_parts.append(f"It features a symbolic transformation of a detected element into '{target_placeholder}'.")
    
    nft_description = " ".join(desc_parts)

    # --- Build Attributes ---
    attributes: List[Dict[str, Any]] = [ # Ensure value is string for most marketplaces
        {"trait_type": "QNFT Sequence ID", "value": str(qnft_sequence_id)},
        {"trait_type": "Generation Timestamp", "value": generation_timestamp_utc, "display_type": "date"}, # Example display_type
    ]
    if original_image_id:
        attributes.append({"trait_type": "Original Image Ref", "value": str(original_image_id)})

    for coin, data in price_info.items():
        coin_name = coin.replace('_USD', '')
        attributes.append({"trait_type": f"{coin_name} Price (USD)", "value": str(data.get("price", "N/A"))})
        # Timestamps can be verbose for attributes; consider if needed or use display_type: "date"
        # attributes.append({"trait_type": f"{coin_name} Price Timestamp", "value": str(data.get("last_updated_utc", "N/A"))})

    if price_movement_info:
        symbol_display = price_movement_info.get("symbol", "Crypto").replace("_USD", "")
        attributes.append({"trait_type": f"{symbol_display} Price Direction", "value": str(price_movement_info.get("direction", "N/A"))})
        attributes.append({"trait_type": f"{symbol_display} Price Change %", "value": f"{price_movement_info.get('percent_change', 0.0):.3f}%" })

    # Applied Effects as Attributes
    fib_anim = applied_effects_summary.get("fibonacci_animation", {})
    if fib_anim:
        attributes.append({"trait_type": "Fibonacci Animation", "value": str(fib_anim.get('type', 'N/A')).replace('_',' ').title()})
        if 'segments' in fib_anim: attributes.append({"trait_type": "Fibonacci Segments", "value": str(fib_anim['segments'])})

    qi_art = applied_effects_summary.get("quantum_inspired_art", {})
    if qi_art:
        attributes.append({"trait_type": "Quantum Art Style", "value": str(qi_art.get('style_name', 'N/A')).replace('_',' ').title()})
        if 'intensity' in qi_art: attributes.append({"trait_type": "Quantum Art Intensity", "value": f"{qi_art['intensity']:.2f}"})
        if 'complexity' in qi_art: attributes.append({"trait_type": "Quantum Art Complexity", "value": f"{qi_art['complexity']:.2f}"})
        if 'color_hue_shift' in qi_art: attributes.append({"trait_type": "Quantum Art Hue Shift", "value": f"{qi_art['color_hue_shift']}°"})
    
    adv_trans = applied_effects_summary.get("advanced_transformation_placeholder", {})
    if adv_trans.get("transformed_regions_count", 0) > 0:
        attributes.append({"trait_type": "Symbolic Transformation", "value": "Applied"})
        attributes.append({"trait_type": "Transformed Element", "value": str(adv_trans.get("region_1", {}).get("original_type", "N/A")).title()})
        attributes.append({"trait_type": "Symbolic Target", "value": str(adv_trans.get("region_1", {}).get("target_type_placeholder", "N/A")).title()})
    else:
        attributes.append({"trait_type": "Symbolic Transformation", "value": "Not Applied"})
        
    # --- Construct Final JSON Structure ---
    # Use short on-chain name/symbol from settings for the `symbol` field.
    # The `name` field in off-chain metadata can be longer and more descriptive.
    final_metadata = {
        "name": nft_name_off_chain,
        "symbol": settings.NFT_ON_CHAIN_SYMBOL_DEFAULT, 
        "description": nft_description,
        "seller_fee_basis_points": settings.NFT_SELLER_FEES_BASIS_POINTS,
        "image": permanent_gif_url,
        "animation_url": permanent_gif_url, 
        "external_url": f"{settings.PROJECT_BASE_URL.rstrip('/')}/qnft/{qnft_sequence_id}" if settings.PROJECT_BASE_URL else "",
        "attributes": attributes,
        "properties": {
            "files": [{"uri": permanent_gif_url, "type": "image/gif"}],
            "category": "image", # Metaplex standard
            "creators": [
                {
                    "address": str(get_service_payer_keypair().pubkey()), # Actual pubkey string
                    "verified": True, # Service wallet signs the mint
                    "share": 100
                }
            ],
            "pqc_demonstration": pqc_processed_data # Output from PqcCryptographyService
        }
    }

    logger.info(f"Final NFT metadata dictionary prepared for QNFT ID #{qnft_sequence_id}.")
    return final_metadata

# Example of how settings might be augmented for this service (in config.py)
# class Settings(BaseSettings):
#     # ... existing ...
#     PROJECT_BASE_URL: str = "https://yourproject.com" # For external_url
#     SIGNIFICANT_PRICE_CHANGE_THRESHOLD_UPPER: float = 1.0 # % for "Surge" title
#     SIGNIFICANT_PRICE_CHANGE_THRESHOLD_LOWER: float = -1.0 # % for "Dip" title
#     # NFT_ON_CHAIN_SYMBOL_DEFAULT: str = "QNFT" (already added in prev step)
#     # NFT_SELLER_FEES_BASIS_POINTS: int = 500 (already added in prev step)

# Example of how this service would be called by an orchestrator:
# async def orchestrate_qnft_creation(image_id_from_upload: str, ...):
#     # ... (steps: fetch prices, generate gif, upload gif to get permanent_gif_url, apply pqc) ...
#     qnft_sequence_id = _get_next_sequence_id_placeholder() # Orchestrator might manage this
#     final_json_metadata = await nft_metadata_service.prepare_final_nft_metadata(
#         permanent_gif_url=...,
#         price_info=...,
#         applied_effects_summary=..., # This needs to be carefully constructed by the orchestrator
#         pqc_processed_data=...,
#         generation_timestamp_utc=datetime.now(timezone.utc).isoformat(),
#         qnft_sequence_id=qnft_sequence_id, # Pass the generated ID
#         price_movement_info=..., # Optional
#         original_image_id=image_id_from_upload
#     )
#     # ... (then upload final_json_metadata to permanent storage, then mint NFT) ...
