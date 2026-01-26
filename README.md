# Fixed Fee Analysis

Compare total execution costs (average slippage + fees) across perpetual DEXs.

## Overview

This tool compares average slippage and trading fees across 8 decentralized perpetual exchanges:

- **Hyperliquid** - Orderbook-based DEX
- **Lighter** - Orderbook-based DEX  
- **Aster** - Orderbook-based DEX
- **Extended** - Orderbook-based DEX 
- **Avantis** - Oracle-based DEX
- **Ostium** - Oracle-based DEX

## Methodology

### Fundamentals
- **Mid Price**: `(Best Bid + Best Ask) / 2`
- **Average Slippage**: `|Avg Execution Price - Mid Price| / Mid Price × 10000` (in bps, which retrieved from buy side and sell side)
  

### 1. Calculation Components

1. **Average Slippage Calculation**:
   - **Orderbook DEXs**: Uses live orderbook depth to simulate partial or full fills.
     - Formula: `(Buy Slippage + Sell Slippage) / 2`
   - **Oracle DEXs**: Uses the price from the oracle.

2. **Total Cost Calculation**:
   - Formula: `Total Cost = Average Slippage + Opening Fee + Closing Fee`
   - **Opening/Closing Fees**: Determined by the selected Order Type (Taker or Maker).

### 2. Fee Structure
Fees are applied for both opening and closing positions.

| Exchange | Taker Fee | Maker Fee | Notes |
|----------|-----------|-----------|-------|
| **Hyperliquid** | 0.9 bps | 0.3 bps | xyz |
| **Lighter** | 0.0 bps | 0.0 bps | |
| **Aster** | 4.0 bps | 0.5 bps | |
| **Extended** | 2.5 bps | 0.0 bps | |
| **Avantis** | Variable based on assets | Variable based on assets | |
| **Ostium** | 3-20 bps | 0.0 bps | |

### 3. Total Cost
The final result is expressed in bps: `Effective Spread + Fees`.

*Effective Spread = 2 × Average Slippage* (estimated round-trip cost).

## Supported Assets

| Category | Assets |
|----------|--------|
| **Commodities** | Gold (XAU), Silver (XAG) |
| **Forex** | EUR/USD, GBP/USD, USD/JPY |
| **Indices** | SPY, QQQ |
| **Stocks (MAG7)** | AAPL, MSFT, GOOG, AMZN, META, NVDA, TSLA |
| **Other** | COIN |

## Installation

```bash
# Install dependencies
pip install flask flask-cors requests

# Run the server
python rwa_fee_comparisson.py
```

## Usage

1. Open `---` in your browser
2. Select an asset from the dropdown
3. Choose order size ($10K, $100K, $1M, $10M)
4. Results auto-refresh on selection

## Project Structure

```
├── rwa_fee_comparisson.py # Backend API server (main)
├── static/
│   └── styles.css        
└── templates/
    └── index.html        
```

## License

MIT
