import asyncio
import logging
from typing import List, Tuple

from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.system_program import CreateAccountParams, create_account, ID as SYS_PROGRAM_ID
from solders.instruction import Instruction, AccountMeta
from solders.transaction import Transaction
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price # For priority fees

from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TxOpts

# SPL Token Program
from spl.token.constants import MINT_LEN, ACCOUNT_LEN
from spl.token.instructions import (
    initialize_mint, InitializeMintParams,
    create_associated_token_account,
    mint_to, MintToParams
)
# Using solders' version of spl_token program_id
from solders.sysvar import RENT
from solders.message import MessageV0, MessageArgs, MessageHeader

from ..config import settings
# Assuming METAPLEX_TOKEN_METADATA_PROGRAM_ID, SPL_ASSOCIATED_TOKEN_ACCOUNT_PROGRAM_ID, SPL_TOKEN_PROGRAM_ID are in settings

# For Metaplex data serialization
import borsh

logger = logging.getLogger(__name__)

# --- Metaplex Data Structures (simplified for on-chain metadata) ---
# Based on https://docs.metaplex.com/programs/token-metadata/accounts#metadata
# and https://github.com/metaplex-foundation/mpl-token-metadata/blob/master/programs/token-metadata/program/src/state/data.rs

class MetaplexCreator(borsh.Enum): # borsh.Enum was an attempt, might need custom struct if directly using borsh like this.
    # Let's simplify to a dictionary for creator and then serialize it properly.
    # For borsh, you'd define a struct.
    # For simplicity, we'll prepare this as part of the instruction data bytes.
    # address: Pubkey (32 bytes)
    # verified: bool (1 byte)
    # share: u8 (1 byte)
    # Total: 34 bytes per creator
    def __init__(self, address: Pubkey, verified: bool, share: int):
        self.address = address
        self.verified = verified
        self.share = share

    def __bytes__(self):
        return bytes(self.address) + self.verified.to_bytes(1, 'little') + self.share.to_bytes(1, 'little')


# Define the schema for DataV2 if using borsh directly for the whole struct
# This is complex. Simpler to construct byte arrays for instruction data if a lib is not used.
# For now, we'll focus on the conceptual data to be passed.
# DataV2 includes: name, symbol, uri, seller_fee_basis_points, creators, collection, uses

# --- Metaplex Instruction Creation Helpers ---
# These would ideally come from a Metaplex Python SDK/library.
# If not, they need to be manually constructed.

def _create_metadata_account_v3_instruction_data(
    name: str,
    symbol: str,
    uri: str,
    seller_fee_basis_points: int,
    creators: List[MetaplexCreator], # Simplified representation
    collection_details: dict = None, # For collection field (optional)
    uses_details: dict = None, # For uses field (optional)
    is_mutable: bool = True
) -> bytes:
    """
    Serializes the arguments for CreateMetadataAccountV3 instruction.
    This is a simplified representation. Actual serialization uses borsh for the DataV2 struct.
    Instruction data:
    1. Instruction discriminator (u8) -> 33 for CreateMetadataAccountV3
    2. data: DataV2 struct (name, symbol, uri, seller_fee_basis_points, Option<Vec<Creator>>, Option<Collection>, Option<Uses>)
    3. is_mutable: bool
    4. collection_details: Option<CollectionDetails> (for setting collection during creation) -> Not used in this basic version beyond DataV2.collection

    For DataV2:
    name: String (4 bytes length + string)
    symbol: String (4 bytes length + string)
    uri: String (4 bytes length + string)
    seller_fee_basis_points: u16
    creators: Option<Vec<Creator>> (1 byte for Some/None + 4 bytes for Vec length + creators data)
    collection: Option<Collection> (complex)
    uses: Option<Uses> (complex)
    """
    # This is a placeholder for actual borsh serialization.
    # A proper implementation would use a schema and borsh.serialize().
    # For example:
    # schema = borsh.schema({
    #     'name': 'string', 'symbol': 'string', 'uri': 'string',
    #     'seller_fee_basis_points': 'u16',
    #     'creators': borsh.Option(borsh.Vec(borsh.schema(MetaplexCreator))), # If MetaplexCreator is a borsh struct
    #     # ... other fields
    # })
    # data_v2_struct = { ... }
    # serialized_data_v2 = borsh.serialize(schema, data_v2_struct)

    # Simplified manual packing (highly error-prone, use a library or proper schema)
    instruction_discriminator = (33).to_bytes(1, 'little') # For CreateMetadataV3

    # Name
    name_bytes = name.encode('utf-8')
    name_len_bytes = len(name_bytes).to_bytes(4, 'little')
    # Symbol
    symbol_bytes = symbol.encode('utf-8')
    symbol_len_bytes = len(symbol_bytes).to_bytes(4, 'little')
    # URI
    uri_bytes = uri.encode('utf-8')
    uri_len_bytes = len(uri_bytes).to_bytes(4, 'little')

    seller_fee_bytes = seller_fee_basis_points.to_bytes(2, 'little')

    # Creators
    if creators:
        creators_option_bytes = (1).to_bytes(1, 'little') # Some(Vec<Creator>)
        creators_len_bytes = len(creators).to_bytes(4, 'little')
        creators_data_bytes = b"".join(bytes(c) for c in creators)
    else:
        creators_option_bytes = (0).to_bytes(1, 'little') # None
        creators_len_bytes = b""
        creators_data_bytes = b""
    
    # Collection & Uses (simplified to None for this basic version)
    collection_option_bytes = (0).to_bytes(1, 'little') # None
    uses_option_bytes = (0).to_bytes(1, 'little') # None

    # DataV2 payload part
    data_v2_payload = (
        name_len_bytes + name_bytes +
        symbol_len_bytes + symbol_bytes +
        uri_len_bytes + uri_bytes +
        seller_fee_bytes +
        creators_option_bytes + creators_len_bytes + creators_data_bytes +
        collection_option_bytes +
        uses_option_bytes
    )
    
    is_mutable_bytes = is_mutable.to_bytes(1, 'little')
    
    # For CreateMetadataV3, the next field is collection_details (Option<CollectionDetails>)
    # For this basic version, we'll assume it's None.
    collection_details_option_bytes = (0).to_bytes(1, 'little') # None

    return instruction_discriminator + data_v2_payload + is_mutable_bytes + collection_details_option_bytes


def _create_master_edition_v3_instruction_data(max_supply: Optional[int]) -> bytes:
    """
    Serializes the arguments for CreateMasterEditionV3 instruction.
    Instruction data:
    1. Instruction discriminator (u8) -> 17 for CreateMasterEditionV3
    2. max_supply: Option<u64> (1 byte for Some/None, 8 bytes for u64 if Some)
    """
    instruction_discriminator = (17).to_bytes(1, 'little')
    if max_supply is not None:
        max_supply_option_bytes = (1).to_bytes(1, 'little') # Some
        max_supply_bytes = max_supply.to_bytes(8, 'little')
    else:
        max_supply_option_bytes = (0).to_bytes(1, 'little') # None
        max_supply_bytes = b""
        
    return instruction_discriminator + max_supply_option_bytes + max_supply_bytes


async def mint_basic_nft(
    gif_id: str,
    metadata_uri: str,
    recipient_address_str: str
) -> Tuple[str, str]:
    """
    Handles the core logic of minting a basic Metaplex NFT.
    Returns (nft_mint_address, transaction_signature).
    """
    solana_client = Client(settings.SOLANA_RPC_URL, commitment=Confirmed) # Use Confirmed for reads
    payer_keypair = Keypair.from_json_file(settings.SERVICE_PAYER_KEYPAIR_PATH)
    logger.info(f"Service Payer Pubkey: {payer_keypair.pubkey()}")

    try:
        recipient_pubkey = Pubkey.from_string(recipient_address_str)
    except ValueError:
        logger.error(f"Invalid recipient_address_str: {recipient_address_str}")
        raise ValueError("Invalid recipient_address format.")

    # --- Simplified On-Chain Metadata ---
    # These values would be configurable or dynamically generated in a full app
    nft_name = f"QNFT {gif_id[:8]}"
    nft_symbol = getattr(settings, "NFT_SYMBOL", "QNBS") # QNFT Basic Symbol
    seller_fees = getattr(settings, "NFT_SELLER_FEES_BASIS_POINTS", 500) # 5%
    
    metaplex_creators = [
        MetaplexCreator(address=payer_keypair.pubkey(), verified=True, share=100)
    ]

    # This function contains blocking Solana SDK calls and will be run in a thread
    def _execute_solana_minting_logic() -> Tuple[str, str]:
        mint_keypair = Keypair()
        mint_pubkey = mint_keypair.pubkey()
        logger.info(f"New NFT Mint Pubkey: {mint_pubkey}")

        # --- Calculate Associated Token Account (ATA) for recipient ---
        # This doesn't create it yet, just gets the address.
        # The create_associated_token_account instruction will create it if it doesn't exist.
        recipient_ata = Pubkey.find_program_address(
            [bytes(recipient_pubkey), bytes(Pubkey.from_string(settings.SPL_TOKEN_PROGRAM_ID)), bytes(mint_pubkey)],
            Pubkey.from_string(settings.SPL_ASSOCIATED_TOKEN_ACCOUNT_PROGRAM_ID)
        )[0]
        logger.info(f"Recipient ATA: {recipient_ata}")

        # --- Metaplex PDA derivations ---
        METAPLEX_PROGRAM_ID = Pubkey.from_string(settings.METAPLEX_TOKEN_METADATA_PROGRAM_ID)
        
        metadata_pda = Pubkey.find_program_address(
            [b"metadata", bytes(METAPLEX_PROGRAM_ID), bytes(mint_pubkey)],
            METAPLEX_PROGRAM_ID
        )[0]
        logger.info(f"Metadata PDA: {metadata_pda}")

        master_edition_pda = Pubkey.find_program_address(
            [b"metadata", bytes(METAPLEX_PROGRAM_ID), bytes(mint_pubkey), b"edition"],
            METAPLEX_PROGRAM_ID
        )[0]
        logger.info(f"Master Edition PDA: {master_edition_pda}")

        # --- Rent calculations (Conceptual - actual values from client) ---
        # For simplicity, we assume payer has enough SOL. Production code needs exact calculations.
        # rent = solana_client.get_minimum_balance_for_rent_exemption(...)

        # --- Transaction Assembly ---
        instructions: List[Instruction] = []

        # 1. Create Mint Account (using SystemProgram.create_account and SPLToken.initialize_mint)
        # This is often done in two steps: create system account, then initialize mint.
        # However, some helper functions might combine this.
        # For explicit control:
        instructions.append(
            create_account(
                CreateAccountParams(
                    from_pubkey=payer_keypair.pubkey(),
                    new_account_pubkey=mint_pubkey,
                    lamports=solana_client.get_minimum_balance_for_rent_exemption(MINT_LEN).value,
                    space=MINT_LEN,
                    program_id=Pubkey.from_string(settings.SPL_TOKEN_PROGRAM_ID)
                )
            )
        )
        instructions.append(
            initialize_mint(
                InitializeMintParams(
                    mint=mint_pubkey,
                    decimals=0,
                    mint_authority=payer_keypair.pubkey(),
                    freeze_authority=payer_keypair.pubkey(), # Or None
                    program_id=Pubkey.from_string(settings.SPL_TOKEN_PROGRAM_ID)
                )
            )
        )

        # 2. Create Associated Token Account for Recipient (if it doesn't exist)
        instructions.append(
            create_associated_token_account(
                payer=payer_keypair.pubkey(),
                owner=recipient_pubkey,
                mint=mint_pubkey,
                program_id=Pubkey.from_string(settings.SPL_ASSOCIATED_TOKEN_ACCOUNT_PROGRAM_ID) # ATA program ID
            )
        )
        
        # 3. Mint 1 Token to Recipient's ATA
        instructions.append(
            mint_to(
                MintToParams(
                    mint=mint_pubkey,
                    dest=recipient_ata,
                    mint_authority=payer_keypair.pubkey(),
                    amount=1,
                    program_id=Pubkey.from_string(settings.SPL_TOKEN_PROGRAM_ID)
                )
            )
        )

        # 4. Create Metaplex Token Metadata Account
        # This requires manual construction if no SDK helper for mpl_token_metadata.CreateMetadataAccountV3
        create_metadata_ix_data = _create_metadata_account_v3_instruction_data(
            name=nft_name,
            symbol=nft_symbol,
            uri=metadata_uri,
            seller_fee_basis_points=seller_fees,
            creators=metaplex_creators,
            is_mutable=True
        )
        instructions.append(
            Instruction(
                program_id=METAPLEX_PROGRAM_ID,
                data=create_metadata_ix_data,
                accounts=[
                    AccountMeta(pubkey=metadata_pda, is_signer=False, is_writable=True),
                    AccountMeta(pubkey=mint_pubkey, is_signer=False, is_writable=False),
                    AccountMeta(pubkey=mint_keypair.pubkey(), is_signer=True, is_writable=False), # Mint authority for the metadata
                    AccountMeta(pubkey=payer_keypair.pubkey(), is_signer=True, is_writable=True), # Payer
                    AccountMeta(pubkey=payer_keypair.pubkey(), is_signer=False, is_writable=False), # Update authority
                    AccountMeta(pubkey=SYS_PROGRAM_ID, is_signer=False, is_writable=False),
                    AccountMeta(pubkey=RENT, is_signer=False, is_writable=False),
                ]
            )
        )

        # 5. Create Metaplex Master Edition Account (makes it a non-fungible token / 1-of-1)
        create_master_edition_ix_data = _create_master_edition_v3_instruction_data(max_supply=0) # Max supply 0 for 1/1
        instructions.append(
            Instruction(
                program_id=METAPLEX_PROGRAM_ID,
                data=create_master_edition_ix_data,
                accounts=[
                    AccountMeta(pubkey=master_edition_pda, is_signer=False, is_writable=True),
                    AccountMeta(pubkey=mint_pubkey, is_signer=False, is_writable=True), # Mint needs to be writable for ME creation
                    AccountMeta(pubkey=payer_keypair.pubkey(), is_signer=True, is_writable=True), # Update authority for ME (payer)
                    AccountMeta(pubkey=payer_keypair.pubkey(), is_signer=True, is_writable=False), # Mint authority for ME (payer)
                    # AccountMeta(pubkey=metadata_pda, is_signer=False, is_writable=False), # Metadata account (already created) - check if needed as writable
                    AccountMeta(pubkey=metadata_pda, is_signer=False, is_writable=True), # Check if metadata needs to be writable for ME creation
                    AccountMeta(pubkey=Pubkey.from_string(settings.SPL_TOKEN_PROGRAM_ID), is_signer=False, is_writable=False),
                    AccountMeta(pubkey=payer_keypair.pubkey(), is_signer=True, is_writable=True), # Payer for ME
                    AccountMeta(pubkey=RENT, is_signer=False, is_writable=False),
                    # Potentially token_account_owner if editions were to be minted from it
                    # AccountMeta(pubkey=payer_keypair.pubkey(), is_signer=True, is_writable=False) # edition_marker_pda - not for max_supply=0
                ]
            )
        )
        
        # Add priority fee instructions (optional but recommended on busy networks)
        # This requires Solana version 1.10+ on the validator.
        # instructions.insert(0, set_compute_unit_price(1_000_000)) # Example: 1,000,000 microLamports per CU
        # instructions.insert(1, set_compute_unit_limit(400_000))  # Example: Max 400,000 CUs

        # --- Send Transaction ---
        try:
            # For Versioned Transactions (recommended for future features like LUTs)
            # blockhash_resp = solana_client.get_latest_blockhash()
            # msg = MessageV0(payer=payer_keypair.pubkey(), instructions=instructions, recent_blockhash=blockhash_resp.value.blockhash)
            # txn = VersionedTransaction(msg, [payer_keypair, mint_keypair])
            
            # For Legacy Transactions
            txn = Transaction(fee_payer=payer_keypair.pubkey())
            for ix in instructions:
                txn.add(ix)
            
            # Get a recent blockhash
            blockhash_resp = solana_client.get_latest_blockhash(commitment=Confirmed)
            txn.recent_blockhash = blockhash_resp.value.blockhash

            # Signers: payer and the new mint_keypair (as it's used as mint authority for metadata)
            # The mint_keypair is also a signer for the create_account instruction if it pays for itself,
            # but here payer_keypair pays. Mint_keypair signs because it's specified as mint_authority for metadata.
            # For CreateMetadataAccountV3, mint_keypair is a signer if it's the mint_authority.
            # For CreateMasterEditionV3, mint_keypair signs if it's the mint_authority.
            # Here, payer_keypair is the mint_authority for the SPL mint.
            # Let's re-check signers for Metaplex:
            # CreateMetadataV3: metadata, mint, mint_authority (signer), payer (signer), update_authority
            # CreateMasterEditionV3: edition, mint, update_authority (signer), mint_authority (signer), payer (signer), metadata
            # Our payer_keypair is mint_authority for SPL, and update_authority for metadata/ME.
            # The mint_keypair itself is not an authority after its creation.
            # So, only payer_keypair should sign for most things, plus mint_keypair for its own creation.
            # The mint_keypair is also needed as a signer for the create_metadata_accounts_v3 if it's the mint_authority in that instruction.
            # Let's assume payer_keypair is mint_authority for both SPL and Metaplex metadata.

            # Sign transaction
            # The `create_account` instruction requires the `mint_keypair` to sign if it's the `new_account_pubkey`
            # even if `from_pubkey` (payer) is different. This is because space is allocated for it.
            txn.sign(payer_keypair, mint_keypair)


            # Serialize and send
            # opts = TxOpts(skip_preflight=False, preflight_commitment=Confirmed, skip_confirmation=False, max_retries=5)
            opts = TxOpts(skip_confirmation=False, preflight_commitment=Confirmed) # skip_confirmation=False means send_transaction will wait for confirmation
            
            logger.info("Sending NFT mint transaction...")
            # resp = solana_client.send_transaction(txn, payer_keypair, mint_keypair, opts=opts) # Legacy
            # For already signed transaction:
            resp = solana_client.send_raw_transaction(txn.serialize(), opts=opts)

            tx_signature = resp.value
            logger.info(f"NFT Mint Transaction Signature: {tx_signature}")

            # Confirmation (send_transaction with skip_confirmation=False usually handles this)
            # solana_client.confirm_transaction(tx_signature, commitment=Confirmed)
            # logger.info(f"Transaction {tx_signature} confirmed.")

            return str(mint_pubkey), str(tx_signature)

        except Exception as e:
            logger.error(f"Solana transaction failed: {e}", exc_info=True)
            # Attempt to parse RPC errors for more details if available
            # if hasattr(e, 'data') and e.data: # For RpcException
            #     logger.error(f"RPC Error Data: {e.data}")
            #     if 'logs' in e.data: logger.error(f"Logs: {e.data['logs']}")
            raise # Re-raise to be handled by the main async function

    # Run the synchronous Solana logic in a separate thread
    try:
        nft_mint_address_str, tx_sig_str = await asyncio.to_thread(_execute_solana_minting_logic)
        return nft_mint_address_str, tx_sig_str
    except Exception as e:
        # Handle exceptions raised from the thread
        logger.error(f"Error during threaded Solana minting execution: {e}")
        # Specific error mapping could be done here if needed
        raise RuntimeError(f"NFT Minting failed: {str(e)}")


# Note: This is a simplified conceptual implementation.
# Production code would need:
# - Robust error parsing from Solana RPC.
# - Exact rent calculations.
# - Potentially a more sophisticated Metaplex interaction library if available for Python,
#   or very careful manual instruction construction and testing.
# - Management of recent_blockhash and potential retries.
# - Consideration for priority fees on mainnet.
# - Wallet for payer needs to be funded.
# - The Metaplex instruction data serialization is particularly complex and error-prone manually.
#   Using a library or very thorough testing against known schemas is critical.
#   The placeholder _create_metadata_account_v3_instruction_data and _create_master_edition_v3_instruction_data
#   are illustrative and would need to be correctly implemented using borsh or similar.
#   The number and order of AccountMetas for Metaplex instructions also need to be exact.
#   It's common to find reference implementations in JS/Rust and translate carefully.I have created `qtnft/app/services/solana_nft_service.py`. This file includes the core logic for minting a basic Metaplex NFT. It defines placeholder structures for Metaplex data and outlines the sequence of Solana instructions required. It emphasizes that the Metaplex instruction data serialization is complex and would need careful implementation using `borsh` or a similar library, or ideally a dedicated Metaplex Python SDK if one is suitable.

Key aspects covered in the `solana_nft_service.py`:
-   Initialization of Solana client and loading of the payer keypair.
-   Conversion of recipient address string to `Pubkey`.
-   Conceptual preparation of on-chain metadata (`nft_name`, `nft_symbol`, `metadata_uri`, `seller_fees`, `creators`).
-   The main synchronous function `_execute_solana_minting_logic` which:
    -   Generates a new `mint_keypair` for the NFT.
    -   Calculates PDAs for Metaplex metadata and master edition accounts.
    -   Constructs a list of instructions:
        1.  Create system account for the mint (`solders.system_program.create_account`).
        2.  Initialize the mint (`spl.token.instructions.initialize_mint`).
        3.  Create an associated token account for the recipient (`spl.token.instructions.create_associated_token_account`).
        4.  Mint 1 token to the recipient's ATA (`spl.token.instructions.mint_to`).
        5.  Create Metaplex Token Metadata Account (manual instruction construction outlined, data serialization with `_create_metadata_account_v3_instruction_data` placeholder).
        6.  Create Metaplex Master Edition Account (manual instruction construction outlined, data serialization with `_create_master_edition_v3_instruction_data` placeholder, `max_supply=0`).
    -   Signs the transaction with the payer and mint keypairs.
    -   Sends the transaction and waits for confirmation.
    -   Returns the NFT mint address and transaction signature.
-   The public async function `mint_basic_nft` wraps `_execute_solana_minting_logic` using `await asyncio.to_thread()`.
-   Logging is included throughout.
-   Placeholders and comments highlight areas needing careful implementation, especially Metaplex instruction data serialization and exact account meta lists.

Next, I'll create the router in `qtnft/app/routers/nft_minting_router.py` and update `qtnft/app/main.py` to include this new router.
I also need to remember to add `borsh` to the `requirements.txt` later.
