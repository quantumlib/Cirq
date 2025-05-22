import React from 'react';
import { Outlet, Link } from 'react-router-dom'; // Outlet renders child routes
import { WalletMultiButton } from '@solana/wallet-adapter-react-ui'; // Import the button
import PriceDisplay from '../common/PriceDisplay'; // Import PriceDisplay

const AppLayout: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col bg-gray-900 text-gray-100 font-sans">
      {/* Header */}
      <header className="bg-gray-800 shadow-lg sticky top-0 z-50">
        <nav className="container mx-auto px-4 sm:px-6 py-3 flex justify-between items-center">
          <Link to="/" className="text-xl sm:text-2xl font-bold text-purple-400 hover:text-purple-300 transition-colors">
            QNFT Core
          </Link>
          <div className="flex items-center space-x-2 sm:space-x-4">
            <Link to="/" className="px-2 py-2 sm:px-3 text-xs sm:text-sm rounded hover:bg-gray-700 transition-colors">Home</Link>
            <Link to="/mint" className="px-2 py-2 sm:px-3 text-xs sm:text-sm rounded hover:bg-gray-700 transition-colors">Mint QNFT</Link>
            <Link to="/marketplace" className="px-2 py-2 sm:px-3 text-xs sm:text-sm rounded hover:bg-gray-700 transition-colors">Gallery</Link>
            {/* Wallet Connection Button - styled for better fit */}
            <WalletMultiButton className="!bg-purple-600 hover:!bg-purple-700 !text-xs sm:!text-sm !font-semibold !py-1.5 sm:!py-2 !px-2 sm:!px-3" />
          </div>
        </nav>
        {/* Optional: Price Display integrated into header or a sub-header bar */}
        <div className="bg-gray-750 py-1.5 px-4 sm:px-6 border-t border-b border-gray-700">
          <div className="container mx-auto">
             <PriceDisplay refreshIntervalMs={60000} showLastUpdatedOverall={true} />
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-grow container mx-auto px-4 sm:px-6 py-8">
        <Outlet /> {/* Child route components will be rendered here */}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-gray-400 text-center py-5 mt-auto border-t border-gray-700">
        <p className="text-sm">&copy; {new Date().getFullYear()} QNFT Project. Conceptual. All Rights Reserved (not really).</p>
      </footer>
    </div>
  );
};

export default AppLayout;
