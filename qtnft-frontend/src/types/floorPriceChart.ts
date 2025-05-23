// Using existing LineChartDataPoint from types/chart.ts if it's generic enough,
// or define a specific one here if needed. Let's assume we can reuse or define it here.

export interface FloorPriceDataPoint {
  time: number;   // Unix timestamp (seconds, UTC), typically start of the period (e.g., day)
  value: number;  // The floor price value
}

export interface FloorPriceChartApiResponse {
  data: FloorPriceDataPoint[];
  collection_id: string;
  duration_days: number;
  currency: string;
}

// Duration options for the UI
export interface DurationOption {
  label: string;    // e.g., "7D", "1M", "1Y"
  value: number;    // Duration in days
}

export const FLOOR_PRICE_DURATION_OPTIONS: DurationOption[] = [
  { label: "7D", value: 7 },
  { label: "30D", value: 30 },
  { label: "90D", value: 90 },
  { label: "180D", value: 180 },
  { label: "1Y", value: 365 },
  // { label: "All", value: 0 }, // Or a very large number to signify all available data
];

export const DEFAULT_FLOOR_PRICE_DURATION_DAYS = 30;
