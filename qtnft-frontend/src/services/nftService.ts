import apiClient from './api'; // Assuming apiClient is your configured Axios instance
import { PaginatedNftResponse, QNftItem, GetNftsParams } from '../types/nft'; // Adjust path as needed, added GetNftsParams

const API_ENDPOINT_BASE = '/nfts'; // Base path for NFT related data

export const fetchPlatformNfts = async (
  params: GetNftsParams = {} // Default to empty object if no params provided
): Promise<PaginatedNftResponse> => {
  try {
    const response = await apiClient.get<PaginatedNftResponse>(`${API_ENDPOINT_BASE}/list`, { params });
    
    if (response.data && Array.isArray(response.data.data) && response.data.pagination) {
      // Optional: Further validation or transformation of data if needed
      // For example, ensuring dates are in a consistent format or numbers are correctly parsed
      // (though Axios typically handles JSON parsing well).
      return response.data;
    }
    
    // Log the problematic response if it doesn't match expectations but isn't an HTTP error
    console.error('Invalid NFT list data format received from API. Response data:', response.data);
    throw new Error('Invalid NFT list data format received from API.');

  } catch (error: any) {
    // Log the error with more details if available from Axios error object
    const errorDetails = error.response?.data?.detail || error.response?.data || error.message || 'An unknown error occurred';
    console.error('Failed to fetch platform NFTs:', errorDetails);
    
    // Re-throw a more specific error or the original error's message
    if (error.response?.data?.detail) {
        throw new Error(String(error.response.data.detail));
    }
    throw new Error(String(errorDetails));
  }
};
