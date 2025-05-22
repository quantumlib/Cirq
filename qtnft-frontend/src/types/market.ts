export interface CryptoPriceData {
  price: number;
  last_updated_utc: string; // ISO 8601 string
  source_api?: string;
}

export interface AllCryptoPrices {
  BTC_USD?: CryptoPriceData;
  SOL_USD?: CryptoPriceData;
  // Add other cryptos if supported by backend
}

export interface PriceApiResponse {
  data: AllCryptoPrices;
  timestamp: string; // ISO 8601 string of when the API response was generated
}
