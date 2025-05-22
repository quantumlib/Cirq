import apiClient from './api'; // Assuming apiClient is your configured Axios instance
import { ChartApiResponse, OhlcvDataPoint, Timeframe } from '../types/chart'; // Adjust path as needed

const API_ENDPOINT_BASE = '/charts'; // Base path for chart data, e.g., backend serves /api/v1/charts/sol_btc

export const fetchOhlcvData = async (
  pair: string, // e.g., "sol_btc" - will be part of the URL path
  timeframe: Timeframe,
  limit?: number // Optional: number of data points to fetch
): Promise<OhlcvDataPoint[]> => {
  try {
    const params: { timeframe: Timeframe; limit?: number } = { timeframe };
    if (limit) {
      params.limit = limit;
    }

    const response = await apiClient.get<ChartApiResponse>(`${API_ENDPOINT_BASE}/${pair.toLowerCase()}`, { params });
    
    if (response.data && Array.isArray(response.data.data)) {
      // Ensure data points have the 'time' field required by lightweight-charts
      // and that it's a number (Unix timestamp in seconds).
      // Also ensure OHLC are numbers.
      return response.data.data
        .filter(d => // Basic validation of critical fields
            typeof d.time === 'number' &&
            typeof d.open === 'number' &&
            typeof d.high === 'number' &&
            typeof d.low === 'number' &&
            typeof d.close === 'number'
        )
        .map(d => ({
            time: d.time, // Already a number from backend, or cast if necessary
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close,
            volume: typeof d.volume === 'number' ? d.volume : undefined, // Ensure volume is number or undefined
        }));
    }
    
    // Log the problematic response if it doesn't match expectations but isn't an HTTP error
    console.error('Invalid chart data format received from API. Response data:', response.data);
    throw new Error('Invalid chart data format received from API.');

  } catch (error: any) {
    // Log the error with more details if available from Axios error object
    const errorDetails = error.response?.data?.detail || error.response?.data || error.message || 'An unknown error occurred';
    console.error(`Failed to fetch OHLCV data for ${pair} (timeframe ${timeframe}):`, errorDetails);
    
    // Re-throw a more specific error or the original error's message
    if (error.response?.data?.detail) {
        throw new Error(String(error.response.data.detail));
    }
    throw new Error(String(errorDetails));
  }
};
