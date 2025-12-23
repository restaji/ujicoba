import requests
import json
from typing import Dict, List, Optional, Tuple

CLIP_SIZES = [10000, 100000, 1000000, 10000000]
TAKER_FEE_BPS = 4.5  # Hyperliquid taker fee: 0.045% = 4.5 bps


def fetch_hyperliquid_orderbook(token: str) -> Dict:
    """Fetch full orderbook from Hyperliquid API."""
    url = "https://api.hyperliquid.xyz/info"
    payload = {
        "type": "l2Book",
        "coin": token.upper()
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
    
    if not data.get('levels') or len(data['levels']) < 2:
        raise ValueError(f"Invalid orderbook for {token} on Hyperliquid")
    
    # Get all available levels (Hyperliquid returns full book by default)
    bids = [
        {"price": float(level["px"]), "qty": float(level["sz"])}
        for level in data['levels'][0]
    ]
    asks = [
        {"price": float(level["px"]), "qty": float(level["sz"])}
        for level in data['levels'][1]
    ]
    
    # Calculate total liquidity
    total_bid_liquidity = sum(b['price'] * b['qty'] for b in bids)
    total_ask_liquidity = sum(a['price'] * a['qty'] for a in asks)
    
    print(f"Fetched orderbook: {len(bids)} bid levels, {len(asks)} ask levels")
    print(f"Total bid liquidity: ${total_bid_liquidity:,.0f}")
    print(f"Total ask liquidity: ${total_ask_liquidity:,.0f}")
    
    return {"bids": bids, "asks": asks, "token": token}


def calculate_slippage(orderbook: Dict, size_usd: float, side: str = "buy") -> Dict:
    """Calculate slippage for a given order size and side. Uses ALL available orderbook levels."""
    if side == "buy":
        levels = sorted(orderbook['asks'], key=lambda x: x['price'])
    else:
        levels = sorted(orderbook['bids'], key=lambda x: x['price'], reverse=True)
    
    if not levels:
        return {"slippage": None, "filled": False, "error": "No liquidity"}
    
    best_bid = max(b['price'] for b in orderbook['bids'])
    best_ask = min(a['price'] for a in orderbook['asks'])
    mid_price = (best_bid + best_ask) / 2
    best_price = best_ask if side == "buy" else best_bid
    
    remaining_usd = size_usd
    total_qty = 0
    total_cost = 0
    levels_used = 0
    depth_used_usd = 0
    worst_price = best_price
    
    # Walk through ALL levels in the orderbook until order is filled or book exhausted
    for level in levels:
        price = level['price']
        qty_available = level['qty']
        value_at_level = qty_available * price
        levels_used += 1
        worst_price = price
        
        if remaining_usd <= value_at_level:
            # This level can complete the order
            qty_taken = remaining_usd / price
            total_qty += qty_taken
            total_cost += remaining_usd
            depth_used_usd += remaining_usd
            remaining_usd = 0
            break
        else:
            # Take entire level and continue to next
            total_qty += qty_available
            total_cost += value_at_level
            depth_used_usd += value_at_level
            remaining_usd -= value_at_level
    
    effective_spread = abs((worst_price - best_price) / best_price) * 100
    
    # Check if order was partially filled
    if remaining_usd > 0:
        filled_usd = size_usd - remaining_usd
        filled_percent = (filled_usd / size_usd) * 100
        
        if total_qty == 0:
            return {
                "slippage": None,
                "filled": False,
                "filled_percent": 0,
                "error": "No liquidity"
            }
        
        avg_price = total_cost / total_qty
        slippage = ((avg_price - mid_price) / mid_price) * 100 if side == "buy" else ((mid_price - avg_price) / mid_price) * 100
        
        return {
            "slippage": round(slippage, 6),
            "slippage_bps": round(slippage * 100, 2),
            "effective_spread_bps": round(effective_spread * 100, 2),
            "filled": False,
            "filled_percent": round(filled_percent, 2),
            "levels_used": levels_used,
            "depth_used_usd": round(depth_used_usd),
            "best_price": round(best_price, 2),
            "worst_price": round(worst_price, 2),
            "avg_price": round(avg_price, 2),
            "unfilled_usd": round(remaining_usd, 2)
        }
    
    # Order fully filled
    avg_price = total_cost / total_qty
    slippage = ((avg_price - mid_price) / mid_price) * 100 if side == "buy" else ((mid_price - avg_price) / mid_price) * 100
    
    return {
        "slippage": round(slippage, 6),
        "slippage_bps": round(slippage * 100, 2),
        "effective_spread_bps": round(effective_spread * 100, 2),
        "filled": True,
        "filled_percent": 100,
        "levels_used": levels_used,
        "depth_used_usd": round(depth_used_usd),
        "best_price": round(best_price, 2),
        "worst_price": round(worst_price, 2),
        "avg_price": round(avg_price, 2),
        "unfilled_usd": 0
    }


def analyze_orderbook(orderbook: Dict) -> Dict:
    """Analyze orderbook and calculate slippage for all clip sizes."""
    if not orderbook['bids'] or not orderbook['asks']:
        return {"error": "Empty orderbook"}
    
    best_bid = max(b['price'] for b in orderbook['bids'])
    best_ask = min(a['price'] for a in orderbook['asks'])
    mid_price = (best_bid + best_ask) / 2
    spread = ((best_ask - best_bid) / best_bid) * 100
    
    result = {
        "token": orderbook['token'],
        "mid_price": round(mid_price, 2),
        "spread_bps": round(spread * 100, 2),
        "taker_fee_bps": TAKER_FEE_BPS,
        "slippage": {}
    }
    
    for size in CLIP_SIZES:
        buy = calculate_slippage(orderbook, size, "buy")
        sell = calculate_slippage(orderbook, size, "sell")
        
        avg_slippage = None
        if buy['slippage'] is not None and sell['slippage'] is not None:
            avg_slippage = (buy['slippage'] + sell['slippage']) / 2
        elif buy['slippage'] is not None:
            avg_slippage = buy['slippage']
        elif sell['slippage'] is not None:
            avg_slippage = sell['slippage']
        
        avg_effective_spread = 0
        if 'effective_spread_bps' in buy and 'effective_spread_bps' in sell:
            avg_effective_spread = (buy['effective_spread_bps'] + sell['effective_spread_bps']) / 2
        
        avg_bps = round(avg_slippage * 100, 2) if avg_slippage is not None else None
        total_cost_bps = round(avg_bps + TAKER_FEE_BPS, 2) if avg_bps is not None else None
        
        size_key = f"${size // 1000}k"
        result['slippage'][size_key] = {
            "avg_bps": avg_bps,
            "taker_fee_bps": TAKER_FEE_BPS,
            "total_cost_bps": total_cost_bps,
            "effective_spread_bps": round(avg_effective_spread, 2),
            "filled": buy.get('filled', False) and sell.get('filled', False),
            "levels": {
                "buy": buy.get('levels_used', 0),
                "sell": sell.get('levels_used', 0)
            },
            "depth_used": {
                "buy": buy.get('depth_used_usd', 0),
                "sell": sell.get('depth_used_usd', 0)
            },
            "unfilled_usd": {
                "buy": buy.get('unfilled_usd', 0),
                "sell": sell.get('unfilled_usd', 0)
            }
        }
    
    return result


def main():
    """Main function to run the slippage analyzer."""
    print("=" * 60)
    print("HYPERLIQUID SLIPPAGE ANALYZER")
    print("=" * 60)
    
    token = input("\nEnter token symbol (e.g., BTC, ETH): ").strip().upper()
    
    # Clean token name
    token = token.replace("-PERP", "").replace("-USD", "").replace("USDT", "").replace("USDC", "")
    
    print(f"\nAnalyzing {token} on Hyperliquid...")
    print("-" * 60)
    
    try:
        orderbook = fetch_hyperliquid_orderbook(token)
        analysis = analyze_orderbook(orderbook)
        
        if 'error' in analysis:
            print(f"Error: {analysis['error']}")
            return
        
        print(f"\nToken: {analysis['token']}")
        print(f"Mid Price: ${analysis['mid_price']}")
        print(f"Spread: {analysis['spread_bps']} bps")
        print(f"Taker Fee: {analysis['taker_fee_bps']} bps")
        print("\n" + "=" * 60)
        print("SLIPPAGE ANALYSIS")
        print("=" * 60)
        
        for size_key, data in analysis['slippage'].items():
            print(f"\n{size_key} Order:")
            if data['avg_bps'] is not None:
                print(f"  Avg Slippage:       {data['avg_bps']} bps")
            else:
                print(f"  Avg Slippage:       N/A")
            print(f"  Taker Fee:          {data['taker_fee_bps']} bps")
            if data['total_cost_bps'] is not None:
                print(f"  Total Cost:         {data['total_cost_bps']} bps")
            else:
                print(f"  Total Cost:         N/A")
            print(f"  Effective Spread:   {data['effective_spread_bps']} bps")
            print(f"  Filled:             {'✓ YES' if data['filled'] else '✗ PARTIAL'}")
            if not data['filled'] and 'unfilled_usd' in data:
                buy_unfilled = data['unfilled_usd']['buy']
                sell_unfilled = data['unfilled_usd']['sell']
                if buy_unfilled > 0 or sell_unfilled > 0:
                    print(f"  Unfilled Amount:    Buy: ${buy_unfilled:,.2f}, Sell: ${sell_unfilled:,.2f}")
            print(f"  Levels Used:        Buy: {data['levels']['buy']}, Sell: {data['levels']['sell']}")
            print(f"  Depth Used (USD):   Buy: ${data['depth_used']['buy']:,}, Sell: ${data['depth_used']['sell']:,}")
        
        print("\n" + "=" * 60)
        print("Analysis complete!")
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
    