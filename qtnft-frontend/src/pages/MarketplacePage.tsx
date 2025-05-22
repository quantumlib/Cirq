import React, { useState, useEffect, useCallback } from 'react';
import { fetchPlatformNfts, GetNftsParams } from '../services/nftService'; // Adjust path as needed
import { QNftItem, PaginatedNftResponse } from '../types/nft'; // Adjust path as needed
import NftCard from '../components/nft/NftCard'; // Adjust path as needed
// import PaginationControls from '../components/common/PaginationControls'; // Future component for more advanced pagination

const ITEMS_PER_PAGE = 12; // Or fetch from settings/config if it becomes dynamic

const MarketplacePage: React.FC = () => {
  const [nfts, setNfts] = useState<QNftItem[]>([]);
  const [pagination, setPagination] = useState({
    currentPage: 1,
    totalPages: 1,
    totalItems: 0,
  });
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadNfts = useCallback(async (page: number) => {
    // For subsequent loads (not initial), we can set loading, but it might cause a flash.
    // Consider a more subtle loading indicator for page changes.
    if (page === 1 && nfts.length === 0) { // Only show full page loading on very first load
        setIsLoading(true);
    } else {
        // For page changes, you might want a different loading state, e.g. a small spinner near pagination
        // For now, we'll use the main isLoading for simplicity of this step
        setIsLoading(true); 
    }
    setError(null);
    try {
      const params: GetNftsParams = { page, limit: ITEMS_PER_PAGE, sortBy: 'mint_timestamp_utc_desc' }; // Default sort
      const response = await fetchPlatformNfts(params);
      setNfts(response.data);
      setPagination(response.pagination);
    } catch (err: any) {
      setError(err.message || 'Failed to load NFTs for the marketplace.');
      setNfts([]); // Clear NFTs on error to avoid showing stale data with an error
    } finally {
      setIsLoading(false);
    }
  }, [nfts.length]); // nfts.length dependency helps manage initial load state vs subsequent loads

  useEffect(() => {
    loadNfts(pagination.currentPage);
  }, [loadNfts, pagination.currentPage]); // Refetch when currentPage changes

  const handleNextPage = () => {
    if (pagination.currentPage < pagination.totalPages) {
      setPagination(prev => ({ ...prev, currentPage: prev.currentPage + 1 }));
    }
  };

  const handlePreviousPage = () => {
    if (pagination.currentPage > 1) {
      setPagination(prev => ({ ...prev, currentPage: prev.currentPage - 1 }));
    }
  };

  // Initial loading state (full page spinner)
  if (isLoading && nfts.length === 0 && pagination.currentPage === 1) { 
    return (
      <div className="text-center py-20">
        <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-purple-500 mx-auto"></div>
        <p className="mt-4 text-lg text-gray-300">Loading QNFTs...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-20 px-4">
        <p className="text-2xl text-red-500 mb-4">Failed to Load QNFTs</p>
        <p className="text-gray-400 mb-6">{error}</p>
        <button 
          onClick={() => loadNfts(pagination.currentPage)}
          className="px-6 py-2 bg-purple-600 text-white font-semibold rounded-lg hover:bg-purple-700 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (nfts.length === 0) {
    return (
      <div className="text-center py-20">
        <p className="text-2xl text-gray-400">No QNFTs Found</p>
        <p className="text-gray-500 mt-2">It seems there are no QNFTs minted on the platform yet, or matching your criteria.</p>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="text-3xl sm:text-4xl font-bold text-purple-400 mb-10 text-center tracking-tight">
        QNFT Platform Gallery
      </h1>
      
      {/* Placeholder for Filters/Sort Controls (Future Enhancement) */}
      {/* <div className="mb-8 p-4 bg-gray-800 rounded-lg shadow"> ... UI for filters/sorting ... </div> */}

      {/* Subtle loading indicator for page changes if main content is already visible */}
      {isLoading && nfts.length > 0 && (
        <div className="text-center text-sm text-purple-400 my-4">Updating gallery...</div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
        {nfts.map((nft) => (
          <NftCard key={nft.nft_id || nft.mint_address} nft={nft} />
        ))}
      </div>

      {/* Basic Pagination Controls */}
      {pagination.totalPages > 1 && (
        <div className="mt-12 flex flex-col sm:flex-row justify-center items-center space-y-3 sm:space-y-0 sm:space-x-4">
          <button
            onClick={handlePreviousPage}
            disabled={pagination.currentPage <= 1 || isLoading}
            className="px-5 py-2 bg-gray-700 text-gray-300 rounded-md hover:bg-gray-600 disabled:opacity-60 disabled:cursor-not-allowed transition-colors w-full sm:w-auto"
          >
            &larr; Previous
          </button>
          <span className="text-gray-400 text-sm">
            Page {pagination.currentPage} of {pagination.totalPages}
          </span>
          <button
            onClick={handleNextPage}
            disabled={pagination.currentPage >= pagination.totalPages || isLoading}
            className="px-5 py-2 bg-gray-700 text-gray-300 rounded-md hover:bg-gray-600 disabled:opacity-60 disabled:cursor-not-allowed transition-colors w-full sm:w-auto"
          >
            Next &rarr;
          </button>
        </div>
      )}
      {pagination.totalItems > 0 && (
        <p className="text-center text-xs text-gray-500 mt-3">
          Displaying {nfts.length} of {pagination.totalItems} QNFTs
        </p>
      )}
    </div>
  );
};

export default MarketplacePage;
