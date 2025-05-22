export interface NftAttribute {
  trait_type: string;
  value: string | number; // Value can be string or number
  display_type?: string; // Optional: e.g., "date", "number", "boost_percentage"
}

export interface QNftItem {
  nft_id: string; // Can be mint_address if that's the primary ID used by API
  mint_address: string;
  name: string;
  gif_url: string;
  metadata_url: string;
  mint_timestamp_utc: string; // ISO 8601
  sequence_id: number;
  key_attributes?: NftAttribute[]; // Optional array of select attributes
  thumbnail_url?: string; // Optional smaller preview
}

export interface PaginatedNftResponse {
  data: QNftItem[];
  pagination: {
    currentPage: number;
    itemsPerPage: number;
    totalItems: number;
    totalPages: number;
  };
}
