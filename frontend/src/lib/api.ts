/**
 * API client for SilverSignal backend.
 * Falls back to seeded data if the backend is unavailable.
 */

const API_BASE = '/api';

export interface APISignalsResponse {
  signals: Record<string, any>;
  regime: {
    regime: 'RISK ON' | 'NEUTRAL' | 'RISK OFF';
    avgScore: number;
    volatilityElevated: boolean;
    justification: string;
  };
  lastUpdated: string;
  tickerCount: number;
}

export interface RefreshStatus {
  isRefreshing: boolean;
  progress: Array<{ ticker: string; status: string; message: string }>;
  lastCompleted: string | null;
  totalTickers: number;
  completedTickers: number;
}

export async function fetchSignals(): Promise<APISignalsResponse | null> {
  try {
    const res = await fetch(`${API_BASE}/signals`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch {
    return null;
  }
}

export async function triggerRefresh(): Promise<{ status: string }> {
  try {
    const res = await fetch(`${API_BASE}/refresh`, { method: 'POST' });
    return await res.json();
  } catch {
    return { status: 'error' };
  }
}

export async function fetchRefreshStatus(): Promise<RefreshStatus | null> {
  try {
    const res = await fetch(`${API_BASE}/refresh/status`);
    return await res.json();
  } catch {
    return null;
  }
}

export async function fetchOHLC(ticker: string, timeframe: string = 'daily'): Promise<any[] | null> {
  try {
    const res = await fetch(`${API_BASE}/ohlc/${ticker}/${timeframe}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return data.data || [];
  } catch {
    return null;
  }
}
