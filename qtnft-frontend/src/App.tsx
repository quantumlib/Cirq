import React from 'react'; // Make sure React is imported
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import AppLayout from './components/layout/AppLayout';
import HomePage from './pages/HomePage';
import MintPage from './pages/MintPage';
import MarketplacePage from './pages/MarketplacePage'; // Import the new MarketplacePage
// import NftDetailPage from './pages/NftDetailPage'; // Future page for NFT details
// import NotFoundPage from './pages/NotFoundPage'; // Optional

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AppLayout />}> {/* AppLayout provides consistent structure */}
          <Route index element={<HomePage />} /> 
          <Route path="mint" element={<MintPage />} />
          <Route path="marketplace" element={<MarketplacePage />} /> {/* Add route for Marketplace */}
          {/* Example for a future NFT detail page */}
          {/* <Route path="nfts/:mintAddress" element={<NftDetailPage />} />  */}
          {/* Add other top-level routes here */}
          {/* <Route path="*" element={<NotFoundPage />} /> */}
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
