import React from 'react';
import { Link } from 'react-router-dom'; // For linking to detail page
import { QNftItem } from '../../types/nft'; // Adjust path as needed
import { formatDistanceToNowStrict, parseISO } from 'date-fns'; // For mint date

interface NftCardProps {
  nft: QNftItem;
}

const NftCard: React.FC<NftCardProps> = ({ nft }) => {
  const displayImageUrl = nft.thumbnail_url || nft.gif_url;
  // Future detail page URL structure, e.g., using mint_address as a unique identifier
  const detailPageUrl = `/nfts/${nft.mint_address || nft.nft_id}`; 

  let formattedMintDate = 'Unknown date';
  try {
    if (nft.mint_timestamp_utc) {
      formattedMintDate = formatDistanceToNowStrict(parseISO(nft.mint_timestamp_utc), { addSuffix: true });
    }
  } catch (error) {
    console.warn(`Failed to parse mint_timestamp_utc for NFT ${nft.nft_id}: ${nft.mint_timestamp_utc}`, error);
    // formattedMintDate remains 'Unknown date'
  }

  return (
    <Link to={detailPageUrl} className="block group focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-opacity-75 rounded-lg">
      <div className="bg-gray-800 rounded-lg shadow-xl overflow-hidden transition-all duration-300 ease-in-out group-hover:shadow-purple-500/40 group-hover:scale-105 h-full flex flex-col">
        <div className="aspect-square w-full overflow-hidden bg-gray-700"> {/* Fixed aspect ratio for image container */}
          {displayImageUrl ? (
            <img
              src={displayImageUrl}
              alt={nft.name || 'QNFT GIF'}
              className="w-full h-full object-cover transition-opacity duration-300 group-hover:opacity-90"
              loading="lazy" // Basic lazy loading for images
              onError={(e) => {
                // Optional: Handle image loading errors, e.g., show a placeholder
                (e.target as HTMLImageElement).src = 'https://via.placeholder.com/300/1A1E26/9CA3AF?text=QNFT'; // Placeholder image
                (e.target as HTMLImageElement).alt = 'Error loading QNFT image';
              }}
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-gray-700">
              <span className="text-gray-500">No Image</span>
            </div>
          )}
        </div>
        <div className="p-4 flex flex-col flex-grow">
          <h3 className="text-md font-semibold text-purple-400 truncate group-hover:text-purple-300" title={nft.name}>
            {nft.name || 'Unnamed QNFT'}
          </h3>
          <p className="text-xs text-gray-500 mt-0.5">
            ID: #{nft.sequence_id}
          </p>
          <p className="text-xs text-gray-400 mt-0.5 mb-2">
            Minted: {formattedMintDate}
          </p>
          
          {/* Optional: Display a few key attributes */}
          {nft.key_attributes && nft.key_attributes.length > 0 && (
            <div className="mt-auto border-t border-gray-700 pt-2 space-y-1">
              {nft.key_attributes.slice(0, 2).map(attr => ( // Show max 2 key attributes
                <div key={attr.trait_type} className="flex justify-between items-center text-xs">
                  <span className="text-gray-500 truncate pr-1" title={attr.trait_type}>{attr.trait_type}:</span>
                  <span className="text-gray-300 font-medium truncate" title={String(attr.value)}>{String(attr.value)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </Link>
  );
};

export default NftCard;
