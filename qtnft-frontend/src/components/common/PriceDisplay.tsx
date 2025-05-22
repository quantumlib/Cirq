import React, { useState, useEffect, useCallback } from 'react';
import { fetchCurrentCryptoPrices } from '../../services/marketDataService'; // Adjust path if needed
import { AllCryptoPrices } from '../../types/market'; // Adjust path if needed
import { formatDistanceToNowStrict, parseISO } from 'date-fns'; // For relative time

// Helper to format price
const formatPrice = (price: number | undefined): string => {
  if (price === undefined || price === null) return 'N/A'; // Handle null as well
  return price.toLocaleString('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 });
};

// Helper to format last updated time
const formatLastUpdated = (isoTimestamp: string | undefined): string => {
  if (!isoTimestamp) return 'N/A';
  try {
    // Ensure the date string is valid and parseISO can handle it
    const date = parseISO(isoTimestamp);
    return formatDistanceToNowStrict(date, { addSuffix: true });
  } catch (e) {
    console.warn("Failed to parse date for 'last updated':", isoTimestamp, e);
    return 'Invalid date';
  }
};

interface PriceDisplayProps {
  refreshIntervalMs?: number; // Interval in milliseconds for auto-refresh
  showLastUpdatedOverall?: boolean; // Whether to show the "Updated X ago" for the component itself
}

const PriceDisplay: React.FC<PriceDisplayProps> = ({ 
  refreshIntervalMs = 30000, // Default 30 seconds
  showLastUpdatedOverall = true 
}) => {
  const [prices, setPrices] = useState<AllCryptoPrices | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true); // isLoading specifically for the current fetch operation
  const [initialLoadComplete, setInitialLoadComplete] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [lastSuccessfulFetchTime, setLastSuccessfulFetchTime] = useState<Date | null>(null);

  const getPrices = useCallback(async (isInitialLoad: boolean) => {
    if (!isInitialLoad) { // For background refreshes, don't show main loading unless error occurred previously
      // Optionally, could have a more subtle refresh indicator
    } else {
      setIsLoading(true); // Only for the very first load or manual refresh triggered by error
    }
    setError(null); // Clear previous errors on new attempt

    try {
      const data = await fetchCurrentCryptoPrices();
      setPrices(data);
      setLastSuccessfulFetchTime(new Date()); // Record time of successful fetch
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to fetch prices.';
      setError(errorMessage);
      console.error("PriceDisplay fetch error:", err);
      // Keep stale data if available, but show error
    } finally {
      setIsLoading(false);
      if (!initialLoadComplete) {
        setInitialLoadComplete(true);
      }
    }
  }, [initialLoadComplete]); // Removed 'isLoading' from deps of getPrices to manage it more manually

  // Initial fetch and setup interval for refreshing
  useEffect(() => {
    getPrices(true); // Initial fetch, isInitialLoad = true

    if (refreshIntervalMs > 0) {
      const intervalId = setInterval(() => getPrices(false), refreshIntervalMs); // Subsequent fetches are not initial
      return () => clearInterval(intervalId); // Cleanup interval on component unmount
    }
  }, [getPrices, refreshIntervalMs]);


  // Display logic
  if (isLoading && !initialLoadComplete) {
    return <div className="text-sm text-gray-400 p-2 animate-pulse">Loading prices...</div>;
  }

  // If there's an error, and we have no prices at all (not even stale), show error.
  // If we have stale prices, we can show them *with* an error message.
  if (error && !prices) {
    return <div className="text-sm text-red-500 p-2">Error: {error}</div>;
  }
  
  // If no prices and no error (e.g. API returned empty successfully, though our service throws error for that)
  if (!prices && !error) {
     return <div className="text-sm text-gray-500 p-2">Price data currently unavailable.</div>;
  }

  const btcData = prices?.BTC_USD;
  const solData = prices?.SOL_USD;

  return (
    <div className="flex flex-col sm:flex-row items-center space-y-2 sm:space-y-0 sm:space-x-3 p-2 bg-gray-700 bg-opacity-60 rounded-lg text-xs text-gray-200 shadow">
      {/* BTC Price */}
      <div className="flex items-center space-x-1.5">
        <span className="font-semibold text-orange-400">BTC:</span>
        <span className="font-mono">{formatPrice(btcData?.price)}</span>
      </div>

      {/* SOL Price */}
      <div className="flex items-center space-x-1.5">
        <span className="font-semibold text-purple-400">SOL:</span>
        <span className="font-mono">{formatPrice(solData?.price)}</span>
      </div>
      
      {/* Error message display alongside stale data if applicable */}
      {error && prices && (
         <div className="text-red-400 text-xs sm:ml-2" title={error}> (Update failed)</div>
      )}

      {/* Overall Last Successful Fetch Time for the component */}
      {showLastUpdatedOverall && lastSuccessfulFetchTime && !error && ( // Only show if no current error
         <div className="text-gray-400 text-xs pt-0.5 sm:pt-0 hidden md:block" title={`Last successful update: ${lastSuccessfulFetchTime.toISOString()}`}>
           (Updated {formatLastUpdated(lastSuccessfulFetchTime.toISOString())})
         </div>
      )}
    </div>
  );
};

export default PriceDisplay;
