// For lightweight-charts, time needs to be UTC Timestamp in seconds.
// Open, High, Low, Close should be numbers.
// Volume is also a number.

export interface OhlcvDataPoint {
  time: number;       // Unix timestamp (seconds, UTC)
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;    // Optional, if volume data is available
}

export interface ChartApiResponse {
  data: OhlcvDataPoint[];
  pair: string;
  timeframe: string;
}

// Define supported timeframes. This can be extended.
// These string values should match what the backend API expects for the 'timeframe' query parameter.
export type Timeframe = "1m" | "5m" | "15m" | "1H" | "4H" | "1D" | "1W" | "1M";

// Example of timeframe configuration for UI and API requests
export interface TimeframeOption {
  label: string;        // Display label for UI (e.g., "1 Hour", "1 Day")
  value: Timeframe;     // Value to send to API (e.g., "1H", "1D")
  // Optional: seconds for this timeframe, useful for calculating start times or data density
  // seconds?: number; 
}

export const TIMEFRAME_OPTIONS: TimeframeOption[] = [
  { label: "1H", value: "1H" },
  { label: "4H", value: "4H" },
  { label: "1D", value: "1D" },
  { label: "1W", value: "1W" },
  { label: "1M", value: "1M" },
  // Finer-grained timeframes (if backend supports them)
  // { label: "1m", value: "1m" },
  // { label: "5m", value: "5m" },
  // { label: "15m", value: "15m" },
];

export const DEFAULT_TIMEFRAME_OPTION = TIMEFRAME_OPTIONS.find(tf => tf.value === "1D") || TIMEFRAME_OPTIONS[0];
