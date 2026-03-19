import { useState, useCallback } from 'react';
import { BarChart3, TrendingUp, Cpu, Grid3X3, LineChart } from 'lucide-react';
import { DataProvider } from './data/DataProvider';
import Overview from './pages/Overview';
import IndicesMacro from './pages/IndicesMacro';
import Mag7Tech from './pages/Mag7Tech';
import Sectors from './pages/Sectors';
import ChartPage from './pages/ChartPage';
import ChartDrawer from './components/ChartDrawer';

const TABS = [
  { id: 'overview', label: 'Overview', icon: BarChart3 },
  { id: 'indices', label: 'Indices & Macro', icon: TrendingUp },
  { id: 'mag7', label: 'Mag 7 + Tech', icon: Cpu },
  { id: 'sectors', label: 'Sectors', icon: Grid3X3 },
  { id: 'chart', label: 'Chart', icon: LineChart },
] as const;

type TabId = (typeof TABS)[number]['id'];

function AppContent() {
  const [activeTab, setActiveTab] = useState<TabId>('overview');
  const [drawerTicker, setDrawerTicker] = useState<string | null>(null);

  const handleTickerClick = useCallback((symbol: string) => {
    setDrawerTicker(symbol);
  }, []);

  const handleCloseDrawer = useCallback(() => {
    setDrawerTicker(null);
  }, []);

  const handleChartNav = useCallback((symbol: string) => {
    setDrawerTicker(null);
    setActiveTab('chart');
    setTimeout(() => {
      window.dispatchEvent(new CustomEvent('select-ticker', { detail: symbol }));
    }, 100);
  }, []);

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* Desktop top nav */}
      <nav className="hidden md:flex items-center h-12 bg-signal-surface border-b border-signal-border px-4 gap-1 shrink-0">
        <div className="flex items-center gap-2 mr-6">
          <SilverSignalLogo />
          <span className="text-sm font-semibold tracking-tight text-signal-text">
            SilverSignal
          </span>
        </div>
        {TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              data-testid={`tab-${tab.id}`}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                isActive
                  ? 'bg-signal-green/10 text-signal-green'
                  : 'text-signal-text-secondary hover:text-signal-text hover:bg-white/5'
              }`}
            >
              <Icon size={14} />
              {tab.label}
            </button>
          );
        })}
      </nav>

      {/* Main content area */}
      <main className="flex-1 overflow-y-auto overflow-x-hidden custom-scrollbar pb-16 md:pb-0">
        {activeTab === 'overview' && <Overview onTickerClick={handleTickerClick} />}
        {activeTab === 'indices' && <IndicesMacro onTickerClick={handleTickerClick} />}
        {activeTab === 'mag7' && <Mag7Tech onTickerClick={handleTickerClick} />}
        {activeTab === 'sectors' && <Sectors onTickerClick={handleTickerClick} />}
        {activeTab === 'chart' && <ChartPage />}
      </main>

      {/* Mobile bottom nav */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 h-14 bg-signal-surface border-t border-signal-border flex items-center justify-around px-2 z-50">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              data-testid={`mobile-tab-${tab.id}`}
              className={`flex flex-col items-center justify-center gap-0.5 min-w-[56px] min-h-[44px] rounded-lg transition-all ${
                isActive
                  ? 'text-signal-green'
                  : 'text-signal-text-muted'
              }`}
            >
              <Icon size={18} />
              <span className="text-[10px] font-medium leading-none">{tab.label.split(' ')[0]}</span>
            </button>
          );
        })}
      </nav>

      {/* Chart Drawer */}
      {drawerTicker && (
        <ChartDrawer
          symbol={drawerTicker}
          onClose={handleCloseDrawer}
          onNavigateToChart={handleChartNav}
        />
      )}
    </div>
  );
}

export default function App() {
  return (
    <DataProvider>
      <AppContent />
    </DataProvider>
  );
}

function SilverSignalLogo() {
  return (
    <svg width="22" height="22" viewBox="0 0 32 32" fill="none" aria-label="SilverSignal logo">
      <rect x="3" y="20" width="5" height="9" rx="1" fill="#00d4a0" opacity="0.5" />
      <rect x="10" y="14" width="5" height="15" rx="1" fill="#00d4a0" opacity="0.7" />
      <rect x="17" y="7" width="5" height="22" rx="1" fill="#00d4a0" opacity="0.9" />
      <rect x="24" y="3" width="5" height="26" rx="1" fill="#00d4a0" />
      <circle cx="26.5" cy="3" r="2" fill="#00d4a0" />
    </svg>
  );
}
