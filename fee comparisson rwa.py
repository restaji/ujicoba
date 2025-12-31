import requests
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

# Fee structures per asset (in basis points)
OPENING_FEES = {
    'HYPERLIQUID': {
        'XAU': 0.45, 'XAG': 0.45, 'QQQ': 0.45, 'SPY': 0.45, 
        'HOOD': 0.45, 'NVDA': 0.45, 'AAPL': 0.45, 'AMZN': 0.45,
        'GOOG': 0.45, 'MSFT': 0.45, 'META': 0.45, 'TSLA': 0.45,
        'USDJPY': 0.45, 'GBPUSD': 0.45, 'EURUSD': 0.45
    },
    'LIGHTER': {
        'XAU': 0, 'XAG': 0, 'QQQ': 0, 'SPY': 0,
        'HOOD': 0, 'NVDA': 0, 'AAPL': 0, 'AMZN': 0,
        'GOOG': 0, 'MSFT': 0, 'META': 0, 'TSLA': 0,
        'USDJPY': 0, 'GBPUSD': 0, 'EURUSD': 0
    }
}

CLOSING_FEES = {
    'HYPERLIQUID': {
        'XAU': 0, 'XAG': 0, 'QQQ': 0, 'SPY': 0,
        'HOOD': 0, 'NVDA': 0, 'AAPL': 0, 'AMZN': 0,
        'GOOG': 0, 'MSFT': 0, 'META': 0, 'TSLA': 0,
        'USDJPY': 0, 'GBPUSD': 0, 'EURUSD': 0
    },
    'LIGHTER': {
        'XAU': 0, 'XAG': 0, 'QQQ': 0, 'SPY': 0,
        'HOOD': 0, 'NVDA': 0, 'AAPL': 0, 'AMZN': 0,
        'GOOG': 0, 'MSFT': 0, 'META': 0, 'TSLA': 0,
        'USDJPY': 0, 'GBPUSD': 0, 'EURUSD': 0
    }
}

@dataclass
class AssetConfig:
    name: str
    symbol_key: str  # Key for fee tables
    hyperliquid_symbol: Optional[str]
    lighter_market_id: Optional[int]

# Asset configurations
ASSETS = {
    'GOLD': AssetConfig('Gold (PAXG)', 'XAU', 'PAXG', 92),
    'SILVER': AssetConfig('Silver (XAG)', 'XAG', 'XAG', 93),
    'HOOD': AssetConfig('Robinhood', 'HOOD', 'HOOD', 108),
    'NVDA': AssetConfig('NVIDIA', 'NVDA', 'NVDA', 110),
    'GOOG': AssetConfig('Google', 'GOOG', 'GOOG', 116),
    'META': AssetConfig('Meta', 'META', 'META', 117),
    'MSFT': AssetConfig('Microsoft', 'MSFT', 'MSFT', 115),
    'AMZN': AssetConfig('Amazon', 'AMZN', 'AMZN', 114),
    'AAPL': AssetConfig('Apple', 'AAPL', 'AAPL', 113),
    'TSLA': AssetConfig('Tesla', 'TSLA', 'TSLA', 112),
    'SPX': AssetConfig('S&P 500', 'SPY', 'SPX500', 42),
    'EURUSD': AssetConfig('EUR/USD', 'EURUSD', 'EUR', 96),
    'GBPUSD': AssetConfig('GBP/USD', 'GBPUSD', 'GBP', 97),
    'USDJPY': AssetConfig('USD/JPY', 'USDJPY', 'JPY', 98),
}

# Taker fees (in basis points)
HYPERLIQUID_TAKER_FEE_BPS = 4.5
LIGHTER_TAKER_FEE_BPS = 0.0

class HyperliquidAPI:
    def __init__(self):
        self.base_url = "https://api.hyperliquid.xyz/info"
        self.headers = {'Content-Type': 'application/json'}
    
    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get orderbook from Hyperliquid"""
        payload = {
            "type": "l2Book",
            "coin": symbol
        }
        
        try:
            response = requests.post(self.base_url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching Hyperliquid orderbook for {symbol}: {e}")
            return None
    
    def calculate_execution_cost(self, orderbook: Dict, order_size_usd: float) -> Optional[Dict]:
        """Calculate the execution cost for a given order size"""
        if not orderbook:
            return None
        
        levels = orderbook.get('levels', [[], []])
        bids = levels[0] if len(levels) > 0 else []
        asks = levels[1] if len(levels) > 1 else []
        
        if not asks or not bids:
            return None
        
        best_bid = float(bids[0].get('px', 0))
        best_ask = float(asks[0].get('px', 0))
        mid_price = (best_bid + best_ask) / 2
        
        # Calculate for buy side
        buy_result = self._calculate_side(asks, order_size_usd, mid_price, 'buy')
        # Calculate for sell side
        sell_result = self._calculate_side(bids, order_size_usd, mid_price, 'sell')
        
        # Average the results
        if buy_result and sell_result:
            avg_slippage = (buy_result['slippage_bps'] + sell_result['slippage_bps']) / 2
            filled = buy_result['filled'] and sell_result['filled']
            
            return {
                'executed': True if filled else 'PARTIAL',
                'best_bid': best_bid,
                'best_ask': best_ask,
                'mid_price': mid_price,
                'slippage_bps': avg_slippage,
                'fee_bps': HYPERLIQUID_TAKER_FEE_BPS,
                'total_cost_bps': avg_slippage + HYPERLIQUID_TAKER_FEE_BPS,
                'buy': buy_result,
                'sell': sell_result,
                'filled': filled
            }
        
        return None
    
    def _calculate_side(self, levels, order_size_usd, mid_price, side):
        """Calculate execution for one side of the book"""
        if side == 'buy':
            levels = sorted(levels, key=lambda x: float(x.get('px', 0)))
        else:
            levels = sorted(levels, key=lambda x: float(x.get('px', 0)), reverse=True)
        
        best_price = float(levels[0].get('px', 0))
        total_qty = 0
        remaining_usd = order_size_usd
        levels_used = 0
        
        for level in levels:
            price = float(level.get('px', 0))
            size = float(level.get('sz', 0))
            
            if price <= 0:
                continue
            
            value_available = price * size
            
            if remaining_usd <= value_available:
                qty_needed = remaining_usd / price
                total_qty += qty_needed
                remaining_usd = 0
                levels_used += 1
                break
            else:
                total_qty += size
                remaining_usd -= value_available
                levels_used += 1
        
        filled_usd = order_size_usd - remaining_usd
        avg_price = filled_usd / total_qty if total_qty > 0 else 0
        
        if side == 'buy':
            slippage_bps = ((avg_price - mid_price) / mid_price) * 10000 if mid_price > 0 else 0
        else:
            slippage_bps = ((mid_price - avg_price) / mid_price) * 10000 if mid_price > 0 else 0
        
        return {
            'filled': remaining_usd == 0,
            'filled_usd': filled_usd,
            'unfilled_usd': remaining_usd,
            'levels_used': levels_used,
            'avg_price': avg_price,
            'slippage_bps': slippage_bps
        }

class LighterAPI:
    def __init__(self):
        self.base_url = "https://mainnet.zklighter.elliot.ai/api/v1"
        self.headers = {'Content-Type': 'application/json'}
    
    def get_orderbook(self, market_id: int) -> Optional[Dict]:
        """Get orderbook from Lighter"""
        url = f"{self.base_url}/orderBookOrders?market_id={market_id}&limit=100"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching Lighter orderbook for market {market_id}: {e}")
            return None
    
    def calculate_execution_cost(self, orderbook: Dict, order_size_usd: float) -> Optional[Dict]:
        """Calculate the execution cost for a given order size"""
        if not orderbook:
            return None
        
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not asks or not bids:
            return None
        
        best_bid = float(bids[0].get('price', 0))
        best_ask = float(asks[0].get('price', 0))
        mid_price = (best_bid + best_ask) / 2
        
        # Calculate for buy side
        buy_result = self._calculate_side(asks, order_size_usd, mid_price, 'buy')
        # Calculate for sell side
        sell_result = self._calculate_side(bids, order_size_usd, mid_price, 'sell')
        
        # Average the results
        if buy_result and sell_result:
            avg_slippage = (buy_result['slippage_bps'] + sell_result['slippage_bps']) / 2
            filled = buy_result['filled'] and sell_result['filled']
            
            return {
                'executed': True if filled else 'PARTIAL',
                'best_bid': best_bid,
                'best_ask': best_ask,
                'mid_price': mid_price,
                'slippage_bps': avg_slippage,
                'fee_bps': LIGHTER_TAKER_FEE_BPS,
                'total_cost_bps': avg_slippage + LIGHTER_TAKER_FEE_BPS,
                'buy': buy_result,
                'sell': sell_result,
                'filled': filled
            }
        
        return None
    
    def _calculate_side(self, levels, order_size_usd, mid_price, side):
        """Calculate execution for one side of the book"""
        if side == 'buy':
            levels = sorted(levels, key=lambda x: float(x.get('price', 0)))
        else:
            levels = sorted(levels, key=lambda x: float(x.get('price', 0)), reverse=True)
        
        best_price = float(levels[0].get('price', 0))
        total_qty = 0
        remaining_usd = order_size_usd
        levels_used = 0
        
        for level in levels:
            price = float(level.get('price', 0))
            size = float(level.get('remaining_base_amount', 0))
            
            if price <= 0:
                continue
            
            value_available = price * size
            
            if remaining_usd <= value_available:
                qty_needed = remaining_usd / price
                total_qty += qty_needed
                remaining_usd = 0
                levels_used += 1
                break
            else:
                total_qty += size
                remaining_usd -= value_available
                levels_used += 1
        
        filled_usd = order_size_usd - remaining_usd
        avg_price = filled_usd / total_qty if total_qty > 0 else 0
        
        if side == 'buy':
            slippage_bps = ((avg_price - mid_price) / mid_price) * 10000 if mid_price > 0 else 0
        else:
            slippage_bps = ((mid_price - avg_price) / mid_price) * 10000 if mid_price > 0 else 0
        
        return {
            'filled': remaining_usd == 0,
            'filled_usd': filled_usd,
            'unfilled_usd': remaining_usd,
            'levels_used': levels_used,
            'avg_price': avg_price,
            'slippage_bps': slippage_bps
        }

class FeeComparator:
    def __init__(self):
        self.hyperliquid = HyperliquidAPI()
        self.lighter = LighterAPI()
    
    def compare_asset(self, asset_key: str, order_size_usd: float) -> Dict:
        """Compare costs for a specific asset and order size"""
        config = ASSETS.get(asset_key.upper())
        if not config:
            return None
        
        result = {
            'asset': config.name,
            'symbol_key': config.symbol_key,
            'order_size_usd': order_size_usd,
            'hyperliquid': None,
            'lighter': None
        }
        
        # Get Hyperliquid data
        if config.hyperliquid_symbol:
            print(f"  Fetching Hyperliquid orderbook for {config.hyperliquid_symbol}...")
            hl_orderbook = self.hyperliquid.get_orderbook(config.hyperliquid_symbol)
            result['hyperliquid'] = self.hyperliquid.calculate_execution_cost(hl_orderbook, order_size_usd)
        
        # Get Lighter data
        if config.lighter_market_id:
            print(f"  Fetching Lighter orderbook for market {config.lighter_market_id}...")
            lighter_orderbook = self.lighter.get_orderbook(config.lighter_market_id)
            result['lighter'] = self.lighter.calculate_execution_cost(lighter_orderbook, order_size_usd)
        
        return result
    
    def print_result(self, result: Dict):
        """Print formatted comparison result"""
        symbol_key = result['symbol_key']
        order_size = result['order_size_usd']
        
        print("\n" + "=" * 80)
        print(f"SLIPPAGE ANALYSIS - {result['asset']}")
        print("=" * 80)
        
        hl = result.get('hyperliquid')
        lt = result.get('lighter')
        
        # Get fixed fees
        hl_opening = OPENING_FEES['HYPERLIQUID'].get(symbol_key, 0)
        hl_closing = CLOSING_FEES['HYPERLIQUID'].get(symbol_key, 0)
        lt_opening = OPENING_FEES['LIGHTER'].get(symbol_key, 0)
        lt_closing = CLOSING_FEES['LIGHTER'].get(symbol_key, 0)
        
        # Print Hyperliquid results
        if hl:
            print(f"\n--- HYPERLIQUID ---")
            print(f"${order_size:,.0f} Order:")
            slippage_bps = hl.get('slippage_bps', 0)
            print(f"  Avg Slippage:        {slippage_bps:.2f} bps")
            print(f"  Taker Fee:           {hl.get('fee_bps', 0):.2f} bps")
            print(f"  Opening Fee:         {hl_opening:.2f} bps")
            print(f"  Closing Fee:         {hl_closing:.2f} bps")
            
            buy_data = hl.get('buy', {})
            sell_data = hl.get('sell', {})
            
            # Effective Spread = 2 × Slippage (round-trip cost)
            effective_spread = 2 * slippage_bps
            print(f"  Effective Spread:    {effective_spread:.2f} bps")
            
            # Total cost including all fees (using effective spread instead of slippage)
            total_with_fixed = effective_spread + hl.get('fee_bps', 0) + hl_opening + hl_closing
            print(f"  Total Cost:          {total_with_fixed:.2f} bps")
            
            if hl.get('executed') == 'PARTIAL':
                print(f"  Filled:              X PARTIAL")
                buy_unfilled = buy_data.get('unfilled_usd', 0)
                sell_unfilled = sell_data.get('unfilled_usd', 0)
                print(f"  Unfilled Amount:     Buy: ${buy_unfilled:,.2f}, Sell: ${sell_unfilled:,.2f}")
            else:
                print(f"  Filled:              ✓ YES")
            
            buy_levels = buy_data.get('levels_used', 0)
            sell_levels = sell_data.get('levels_used', 0)
            print(f"  Levels Used:         Buy: {buy_levels}, Sell: {sell_levels}")
            
            buy_filled = buy_data.get('filled_usd', order_size)
            sell_filled = sell_data.get('filled_usd', order_size)
            print(f"  Depth Used (USD):    Buy: ${buy_filled:,.0f}, Sell: ${sell_filled:,.0f}")
        
        # Print Lighter results
        if lt:
            print(f"\n--- LIGHTER ---")
            print(f"${order_size:,.0f} Order:")
            slippage_bps = lt.get('slippage_bps', 0)
            print(f"  Avg Slippage:        {slippage_bps:.2f} bps")
            print(f"  Taker Fee:           {lt.get('fee_bps', 0):.2f} bps")
            print(f"  Opening Fee:         {lt_opening:.2f} bps")
            print(f"  Closing Fee:         {lt_closing:.2f} bps")
            
            buy_data = lt.get('buy', {})
            sell_data = lt.get('sell', {})
            
            # Effective Spread = 2 × Slippage (round-trip cost)
            effective_spread = 2 * slippage_bps
            print(f"  Effective Spread:    {effective_spread:.2f} bps")
            
            # Total cost including all fees (using effective spread instead of slippage)
            total_with_fixed = effective_spread + lt.get('fee_bps', 0) + lt_opening + lt_closing
            print(f"  Total Cost:          {total_with_fixed:.2f} bps")
            
            if lt.get('executed') == 'PARTIAL':
                print(f"  Filled:              X PARTIAL")
                buy_unfilled = buy_data.get('unfilled_usd', 0)
                sell_unfilled = sell_data.get('unfilled_usd', 0)
                print(f"  Unfilled Amount:     Buy: ${buy_unfilled:,.2f}, Sell: ${sell_unfilled:,.2f}")
            else:
                print(f"  Filled:              ✓ YES")
            
            buy_levels = buy_data.get('levels_used', 0)
            sell_levels = sell_data.get('levels_used', 0)
            print(f"  Levels Used:         Buy: {buy_levels}, Sell: {sell_levels}")
            
            buy_filled = buy_data.get('filled_usd', order_size)
            sell_filled = sell_data.get('filled_usd', order_size)
            print(f"  Depth Used (USD):    Buy: ${buy_filled:,.0f}, Sell: ${sell_filled:,.0f}")
        
        print()
        
        # Winner determination
        if hl and lt:
            # Calculate total using effective spread (2 × slippage)
            hl_effective_spread = 2 * hl.get('slippage_bps', 0)
            lt_effective_spread = 2 * lt.get('slippage_bps', 0)
            
            hl_total = hl_effective_spread + hl.get('fee_bps', 0) + hl_opening + hl_closing
            lt_total = lt_effective_spread + lt.get('fee_bps', 0) + lt_opening + lt_closing
            
            hl_partial = hl.get('executed') == 'PARTIAL'
            lt_partial = lt.get('executed') == 'PARTIAL'
            
            if hl_partial or lt_partial:
                print(f"⚠️  Note: Partial fills detected. Compare liquidity availability carefully.")
            else:
                if hl_total < lt_total:
                    winner = "Hyperliquid"
                    savings_bps = lt_total - hl_total
                    savings_usd = order_size * savings_bps / 10000
                elif lt_total < hl_total:
                    winner = "Lighter"
                    savings_bps = hl_total - lt_total
                    savings_usd = order_size * savings_bps / 10000
                else:
                    winner = "Tie"
                    savings_bps = 0
                    savings_usd = 0
                
                print(f"🏆 WINNER: {winner}")
                if savings_bps > 0:
                    print(f"   Savings: {savings_bps:.2f} bps (${savings_usd:,.2f} on ${order_size:,} order)")
        
        print("=" * 80)
        print("\nDefinitions:")
        print("  • Slippage: One-way price impact from mid-price")
        print("  • Effective Spread: Round-trip cost (2 × Slippage)")
        print("  • Total Cost: Effective Spread + Taker Fee + Opening Fee + Closing Fee")
        print("=" * 80)

def print_available_assets():
    """Print list of available assets"""
    print("\nAvailable Assets:")
    print("-" * 60)
    for key, config in ASSETS.items():
        hl_status = "✓" if config.hyperliquid_symbol else "✗"
        lt_status = "✓" if config.lighter_market_id else "✗"
        print(f"{key:<10} - {config.name:<25} [HL: {hl_status}] [Lighter: {lt_status}]")
    print("-" * 60)

def main():
    comparator = FeeComparator()
    
    print("=" * 60)
    print("PERP DEX FEE COMPARISON TOOL")
    print("Hyperliquid vs Lighter (with Opening/Closing Fees)")
    print("=" * 60)
    
    while True:
        print("\n")
        print_available_assets()
        
        # Get asset from user
        asset_input = input("\nEnter asset symbol (or 'quit' to exit): ").strip()
        
        if asset_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if asset_input.upper() not in ASSETS:
            print(f"❌ Asset '{asset_input}' not found. Please choose from the list above.")
            continue
        
        # Get order size from user
        try:
            order_size_input = input("Enter order size in USD (e.g., 10000, 100000, 1000000): ").strip()
            order_size = float(order_size_input.replace(',', ''))
            
            if order_size <= 0:
                print("❌ Order size must be greater than 0")
                continue
        except ValueError:
            print("❌ Invalid order size. Please enter a number.")
            continue
        
        # Perform comparison
        print(f"\nAnalyzing {asset_input.upper()} with ${order_size:,.0f} order size...")
        result = comparator.compare_asset(asset_input, order_size)
        
        if result:
            comparator.print_result(result)
        else:
            print("❌ Error comparing asset")
        
        # Ask if user wants to continue
        continue_input = input("\nCompare another asset? (y/n): ").strip().lower()
        if continue_input != 'y':
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()