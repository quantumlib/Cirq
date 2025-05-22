import apiClient from './api'; // Assuming apiClient is your configured Axios instance
import { PriceApiResponse, AllCryptoPrices } from '../types/market'; // Adjust path as needed

const API_ENDPOINT = '/prices/current'; // Relative to apiClient.defaults.baseURL

export const fetchCurrentCryptoPrices = async (): Promise<AllCryptoPrices> => {
  try {
    const response = await apiClient.get<PriceApiResponse>(API_ENDPOINT);
    
    if (response.data && response.data.data) {
      // Basic validation for expected structure might be useful here
      if (typeof response.data.data !== 'object') {
        throw new Error('Invalid price data format: "data.data" is not an object.');
      }
      // Further checks, e.g., if BTC_USD or SOL_USD exist and have a price property
      // if (response.data.data.BTC_USD && typeof response.data.data.BTC_USD.price !== 'number') {
      //   console.warn('BTC_USD price is not a number:', response.data.data.BTC_USD);
      // }
      return response.data.data;
    }
    // Log the problematic response if it doesn't match expectations but isn't an HTTP error
    console.error('Invalid price data format received from API. Response data:', response.data);
    throw new Error('Invalid price data format received from API.');
  } catch (error: any) {
    // Log the error with more details if available from Axios error object
    const errorDetails = error.response?.data?.detail || error.response?.data || error.message || 'An unknown error occurred';
    console.error('Failed to fetch current crypto prices:', errorDetails);
    
    // Re-throw a more specific error or the original error's message
    // If error.response.data.detail exists (FastAPI default for HTTPExceptions), use it.
    if (error.response?.data?.detail) {
        throw new Error(String(error.response.data.detail));
    }
    throw new Error(String(errorDetails));
  }
};
