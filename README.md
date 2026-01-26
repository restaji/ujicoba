# Fixed Fee Analysis

Compare total execution costs (slippage + fees) across perpetual DEXs.

## Overview

This tool compares slippage and trading fees across 8 decentralized perpetual exchanges:

- **Hyperliquid** - Orderbook-based DEX
- **Lighter** - Orderbook-based DEX  
- **Aster** - Orderbook-based DEX
- **Avantis** - Oracle-based DEX
- **Ostium** - Oracle-based DEX
- **Extended** - Orderbook-based DEX (Starknet)

## Methodology

The tool calculates the **Total Cost (basis points)** for executing a market order of a specified size (e.g., $100K).

### Formula
$$ \text{Total Cost (bps)} = \text{Slippage (bps)} + \text{Open Fee (bps)} + \text{Close Fee (bps)} $$

### Mid Price Calculation
For all exchanges, the **mid-price** is calculated as:
```
Mid Price = (Best Bid + Best Ask) / 2
```

### Slippage Calculation
Slippage measures price impact and is calculated as:
```
Slippage (bps) = ((Avg Execution Price - Mid Price) / Mid Price) × 10000
```

- **Orderbook DEXs (Hyperliquid, Lighter, Aster, Extended)**:
  - Fetches the full orderbook snapshot.
  - Simulates walking down the orderbook to fill the requested size.
  - Calculates average execution price from filled levels.

- **Non-Orderbook DEXs (Avantis, Ostium)**:
  - Uses fixed spread/fee parameters from documentation.
  - Assumes zero price impact for supported sizes.

### 2. Fee Structure (Taker Fees)
Fees are applied for both opening and closing positions.

- **Hyperliquid**: 4.5 bps
- **Lighter**: 0.0 bps (currently)
- **Aster**: 4.0 bps
- **Avantis**: Variable (based on OI skew/utilization)
- **Ostium**: 3-20 bps (varies by asset class)
- **Extended**: Orderbook dependent

### 3. Total Cost
The final result is expressed in bps: `Effective Spread + Fees`.
*Effective Spread = 2 × Slippage* (approximate round-trip cost).

## Supported Assets

| Category | Assets |
|----------|--------|
| **Commodities** | Gold (XAU), Silver (XAG) |
| **Forex** | EUR/USD, GBP/USD, USD/JPY |
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
│   └── styles.css         # Stylesheet
└── templates/
    └── index.html         # Frontend UI
```

## License

MIT
