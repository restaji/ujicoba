# Fixed Fee Analysis

Compare total execution costs (slippage + fees) across perpetual DEXs.

## Overview

This tool compares slippage and trading fees across 8 decentralized perpetual exchanges:

- **Hyperliquid** - Orderbook-based DEX
- **Lighter** - Orderbook-based DEX  
- **Paradex** - Orderbook-based DEX
- **Aster** - Orderbook-based DEX
- **Avantis** - Oracle-based DEX
- **Ostium** - Oracle-based DEX
- **Extended** - Orderbook-based DEX (Starknet)
- **Variational** - Peer-to-Peer DEX (RFQ)

## Methodology

The tool calculates the **Total Cost (basis points)** for executing a market order of a specified size (e.g., $100K).

### Formula
$$ \text{Total Cost (bps)} = \text{Slippage (bps)} + \text{Open Fee (bps)} + \text{Close Fee (bps)} $$

### 1. Slippage Calculation
Slippage is the difference between the **mid-market price** and the **average execution price**.

- **Orderbook DEXs (Hyperliquid, Lighter, Paradex, Aster, Extended)**:
  - Fetches the full L2 orderbook snapshot.
  - Simulates walking down the orderbook to fill the requested size.
  - Calculation: `(Avg Execution Price - Mid Price) / Mid Price`

- **Oracle DEXs (Avantis, Ostium)**:
  - Uses fixed spread/fee parameters from documentation.
  - Assumes zero price impact for supported sizes (infinite liquidity assumption up to caps) or fixed slippage models.

- **Variational (RFQ)**:
  - Fetches pre-calculated bid/ask quotes for size buckets ($1K, $100K, $1M).
  - Interpolates spread for sizes within range.
  - Extrapolates linearly for order sizes > $1M.
  - Uses `mark_price` as the reference for spread normalization.
  - Mid-price calculated as `(best_bid + best_ask) / 2` derived from the smallest quote bucket.

### 2. Fee Structure (Taker Fees)
Fees are applied for both opening and closing positions.

- **Hyperliquid**: 4.5 bps
- **Lighter**: 0.0 bps (currently)
- **Paradex**: 0.0 bps (maker/taker model dependent)
- **Aster**: 4.0 bps
- **Avantis**: Variable (based on OI skew/utilization)
- **Ostium**: 3-20 bps (varies by asset class)
- **Extended**: Orderbook dependent
- **Variational**: 0.0 bps (fees baked into spread)

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
