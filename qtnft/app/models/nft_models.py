from pydantic import BaseModel, Field

class BasicNftMintRequest(BaseModel):
    gif_id: str = Field(
        ..., 
        description="Identifier of the generated GIF that this NFT represents.", 
        example="a1b2c3d4-e5f6.gif"
    )
    metadata_uri: str = Field(
        ...,
        description="Placeholder URI for the off-chain JSON metadata. Actual upload and URI generation is a later feature.",
        example="https_arweave_placeholder_uri_for_metadata_json"
    )
    recipient_address: str = Field(
        ...,
        description="Solana public key string of the wallet that will receive the NFT.",
        example="RecipientSo1anaPublicKeyString1111111111111"
    )

class BasicNftMintResponse(BaseModel):
    message: str = Field(default="Basic NFT minted successfully")
    nft_mint_address: str = Field(..., description="The public key of the newly minted NFT's Mint Account.")
    transaction_signature: str = Field(..., description="The signature of the successful minting transaction.")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Basic NFT minted successfully",
                "nft_mint_address": "MintPublicKeyGeneratedOnSolana1111111111",
                "transaction_signature": "TransactionSignatureGeneratedOnSolana2222222222222"
            }
        }
