import requests
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# --- CONSTANTS ---
HYPERLIQUID_TAKER_FEE_BPS = 4.5
LIGHTER_TAKER_FEE_BPS = 0.0
PARADEX_TAKER_FEE_BPS = 0.0
ASTER_TAKER_FEE_BPS = 0.0 

@dataclass
class AssetConfig:
    name: str
    symbol_key: str
    hyperliquid_symbol: Optional[str]
    lighter_market_id: Optional[int]
    paradex_symbol: Optional[str]
    aster_symbol: Optional[str]

# ASSETS CONFIGURATION
ASSETS = {
    'QQQ': AssetConfig('Invesco QQQ', 'QQQ', None, None, None, 'QQQUSDT'),
    'GOLD': AssetConfig('Gold (PAXG)', 'XAU', 'PAXG', 92, 'PAXG', 'XAUUSDT'),
    'SILVER': AssetConfig('Silver (XAG)', 'XAG', 'SILVER', 93, None, 'XAGUSDT'),
    'HOOD': AssetConfig('Robinhood', 'HOOD', 'HOOD', 108, None, 'HOODUSDT'),
    'NVDA': AssetConfig('NVIDIA', 'NVDA', 'NVDA', 110, None, 'NVDAUSDT'),
    'GOOG': AssetConfig('Google', 'GOOG', 'GOOGL', 116, None, 'GOOGUSDT'),
    'META': AssetConfig('Meta', 'META', 'META', 117, None, 'METAUSDT'),
    'MSFT': AssetConfig('Microsoft', 'MSFT', 'MSFT', 115, None, 'MSFTUSDT'),
    'AMZN': AssetConfig('Amazon', 'AMZN', 'AMZN', 114, None, 'AMZNUSDT'),
    'AAPL': AssetConfig('Apple', 'AAPL', 'AAPL', 113, None, 'AAPLUSDT'),
    'TSLA': AssetConfig('Tesla', 'TSLA', 'TSLA', 112, None, 'TSLAUSDT'),
    'COIN': AssetConfig('Coinbase', 'COIN', 'COIN', 109, None, 'COINUSDT'),
    'EURUSD': AssetConfig('EUR/USD', 'EURUSD', 'EUR', 96, None, None),
    'GBPUSD': AssetConfig('GBP/USD', 'GBPUSD', 'GBP', 97, None, None),
    'USDJPY': AssetConfig('USD/JPY', 'USDJPY', 'JPY', 98, None, None),
}


class HyperliquidAPI:
    def __init__(self):
        self.base_url = "https://api.hyperliquid.xyz/info"
        self.headers = {'Content-Type': 'application/json'}

    def normalize_symbol(self, symbol: str) -> str:
        s = symbol.upper()
        if s == "NDX": return "kPW"
        return s

    def _fetch_coin(self, coin: str, n_sig_figs: Optional[int]) -> Optional[Dict]:
        payload = {"type": "l2Book", "coin": coin}
        if n_sig_figs is not None:
            payload["nSigFigs"] = n_sig_figs

        try:
            response = requests.post(self.base_url, json=payload, headers=self.headers, timeout=10)
            if response.status_code != 200: 
                print(f"  > HTTP {response.status_code} for {coin}")
                return None
            data = response.json()
            if not data: return None

            levels = data.get('levels', [])
            if not isinstance(levels, list) or len(levels) < 2: return None
            
            bids = levels[0] if isinstance(levels[0], list) else []
            asks = levels[1] if isinstance(levels[1], list) else []
            if not bids or not asks: return None
            
            formatted_bids = []
            formatted_asks = []
            
            for bid in bids:
                if isinstance(bid, dict): formatted_bids.append(bid)
                elif isinstance(bid, list) and len(bid) >= 2:
                    formatted_bids.append({'px': str(bid[0]), 'sz': str(bid[1])})
            
            for ask in asks:
                if isinstance(ask, dict): formatted_asks.append(ask)
                elif isinstance(ask, list) and len(ask) >= 2:
                    formatted_asks.append({'px': str(ask[0]), 'sz': str(ask[1])})
            
            return {'levels': [formatted_bids, formatted_asks]}
            
        except Exception as e:
            print(f"  > Error fetching {coin}: {e}")
            return None

    def get_orderbook(self, symbol: str, n_sig_figs: Optional[int] = None) -> Tuple[Optional[Dict], bool]:
        raw_symbol = self.normalize_symbol(symbol)
        
        # Try XYZ (RWA) version first
        rwa_coin = f"xyz:{raw_symbol}"
        print(f"  > Trying {rwa_coin}...")
        book = self._fetch_coin(rwa_coin, n_sig_figs)
        if book:
            print(f"  > Success with {rwa_coin}")
            return book, True
        
        # Fall back to regular symbol
        print(f"  > Trying {raw_symbol}...")
        book = self._fetch_coin(raw_symbol, n_sig_figs)
        if book:
            print(f"  > Success with {raw_symbol}")
            return book, False
        
        print(f"  > Failed to fetch orderbook for {symbol}")
        return None, False

    def calculate_execution_cost(self, orderbook: Dict, order_size_usd: float, anchor_mid_price: Optional[float] = None) -> Optional[Dict]:
        if not orderbook: return None
        levels = orderbook.get('levels', [[], []])
        bids = levels[0] if len(levels) > 0 else []
        asks = levels[1] if len(levels) > 1 else []
        if not asks or not bids: return None

        try:
            best_bid = float(bids[0].get('px', 0))
            best_ask = float(asks[0].get('px', 0))
        except (ValueError, AttributeError, IndexError):
            return None
        
        if best_bid <= 0 or best_ask <= 0: return None
        
        mid_price = anchor_mid_price if anchor_mid_price else (best_bid + best_ask) / 2
        
        print(f"  > Best Bid: ${best_bid:,.4f}, Best Ask: ${best_ask:,.4f}, Mid: ${mid_price:,.4f}")

        buy_result = self._calculate_side(asks, order_size_usd, mid_price, 'buy')
        sell_result = self._calculate_side(bids, order_size_usd, mid_price, 'sell')

        if buy_result and sell_result:
            avg_slippage = (buy_result['slippage_bps'] + sell_result['slippage_bps']) / 2
            filled = buy_result['filled'] and sell_result['filled']
            max_levels_hit = (buy_result['levels_used'] >= len(asks)) or (sell_result['levels_used'] >= len(bids))
            
            return {
                'executed': True if filled else 'PARTIAL',
                'best_bid': best_bid,
                'best_ask': best_ask,
                'mid_price': mid_price,
                'slippage_bps': avg_slippage,
                'fee_bps': HYPERLIQUID_TAKER_FEE_BPS,
                'buy': buy_result,
                'sell': sell_result,
                'filled': filled,
                'max_levels_hit': max_levels_hit
            }
        return None

    def _calculate_side(self, levels, order_size_usd, mid_price, side):
        levels = sorted(levels, key=lambda x: float(x.get('px', 0)), reverse=(side == 'sell'))
        total_qty = 0
        total_cost = 0
        remaining_usd = order_size_usd
        levels_used = 0
        
        for level in levels:
            try:
                price = float(level.get('px', 0))
                size = float(level.get('sz', 0))
            except (ValueError, AttributeError): continue
            
            if price <= 0 or size <= 0: continue
            value_available = price * size
            
            if remaining_usd <= value_available:
                qty_needed = remaining_usd / price
                total_qty += qty_needed
                total_cost += remaining_usd
                remaining_usd = 0
                levels_used += 1
                break
            else:
                total_qty += size
                total_cost += value_available
                remaining_usd -= value_available
                levels_used += 1

        filled_usd = order_size_usd - remaining_usd
        avg_price = total_cost / total_qty if total_qty > 0 else 0
        slippage_bps = abs((avg_price - mid_price) / mid_price) * 10000 if (mid_price > 0 and avg_price > 0) else 0
            
        return {'filled': remaining_usd == 0, 'filled_usd': filled_usd, 'unfilled_usd': remaining_usd, 'levels_used': levels_used, 'avg_price': avg_price, 'slippage_bps': slippage_bps}


class LighterAPI:
    def __init__(self):
        self.base_url = "https://mainnet.zklighter.elliot.ai/api/v1"
        self.headers = {'Content-Type': 'application/json'}

    def get_orderbook(self, market_id: int) -> Optional[Dict]:
        url = f"{self.base_url}/orderBookOrders?market_id={market_id}&limit=100"
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception: return None

    def calculate_execution_cost(self, orderbook: Dict, order_size_usd: float) -> Optional[Dict]:
        if not orderbook: return None
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        if not asks or not bids: return None

        best_bid = float(bids[0].get('price', 0))
        best_ask = float(asks[0].get('price', 0))
        mid_price = (best_bid + best_ask) / 2

        buy_result = self._calculate_side(asks, order_size_usd, mid_price, 'buy')
        sell_result = self._calculate_side(bids, order_size_usd, mid_price, 'sell')

        if buy_result and sell_result:
            avg_slippage = (buy_result['slippage_bps'] + sell_result['slippage_bps']) / 2
            filled = buy_result['filled'] and sell_result['filled']
            return {
                'executed': True if filled else 'PARTIAL',
                'mid_price': mid_price,
                'slippage_bps': avg_slippage,
                'fee_bps': LIGHTER_TAKER_FEE_BPS,
                'buy': buy_result,
                'sell': sell_result,
                'filled': filled
            }
        return None

    def _calculate_side(self, levels, order_size_usd, mid_price, side):
        levels = sorted(levels, key=lambda x: float(x.get('price', 0)), reverse=(side == 'sell'))
        total_qty = 0
        total_cost = 0
        remaining_usd = order_size_usd
        levels_used = 0
        
        for level in levels:
            price = float(level.get('price', 0))
            size = float(level.get('remaining_base_amount', 0))
            if price <= 0: continue
            value_available = price * size
            if remaining_usd <= value_available:
                qty_needed = remaining_usd / price
                total_qty += qty_needed
                total_cost += remaining_usd
                remaining_usd = 0
                levels_used += 1
                break
            else:
                total_qty += size
                total_cost += value_available
                remaining_usd -= value_available
                levels_used += 1
                
        filled_usd = order_size_usd - remaining_usd
        avg_price = total_cost / total_qty if total_qty > 0 else 0
        slippage_bps = abs((avg_price - mid_price) / mid_price) * 10000 if mid_price > 0 else 0
        return {'filled': remaining_usd == 0, 'filled_usd': filled_usd, 'unfilled_usd': remaining_usd, 'levels_used': levels_used, 'avg_price': avg_price, 'slippage_bps': slippage_bps}


class ParadexAPI:
    def __init__(self):
        self.base_url = "https://api.prod.paradex.trade/v1"
        self.headers = {'Accept': 'application/json'}

    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        pair = f"{symbol.upper()}-USD-PERP"
        url = f"{self.base_url}/orderbook/{pair}"
        params = {'depth': 100}
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            if response.status_code == 404:
                print(f"  > Market {pair} not found on Paradex")
                return None
            response.raise_for_status()
            data = response.json()
            result_data = data.get('result', data)
            if not result_data.get('bids') or not result_data.get('asks'): return None
            bids = [{'price': float(b[0]), 'qty': float(b[1])} for b in result_data['bids']]
            asks = [{'price': float(a[0]), 'qty': float(a[1])} for a in result_data['asks']]
            return {'bids': bids, 'asks': asks}
        except: return None

    def calculate_execution_cost(self, orderbook: Dict, order_size_usd: float) -> Optional[Dict]:
        if not orderbook: return None
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        if not asks or not bids: return None
        best_bid = bids[0]['price']
        best_ask = asks[0]['price']
        mid_price = (best_bid + best_ask) / 2
        buy_result = self._calculate_side(asks, order_size_usd, mid_price, 'buy')
        sell_result = self._calculate_side(bids, order_size_usd, mid_price, 'sell')
        if buy_result and sell_result:
            avg_slippage = (buy_result['slippage_bps'] + sell_result['slippage_bps']) / 2
            filled = buy_result['filled'] and sell_result['filled']
            return {
                'executed': True if filled else 'PARTIAL',
                'mid_price': mid_price,
                'slippage_bps': avg_slippage,
                'fee_bps': PARADEX_TAKER_FEE_BPS,
                'buy': buy_result,
                'sell': sell_result,
                'filled': filled
            }
        return None

    def _calculate_side(self, levels, order_size_usd, mid_price, side):
        levels = sorted(levels, key=lambda x: x['price'], reverse=(side == 'sell'))
        total_qty = 0
        total_cost = 0
        remaining_usd = order_size_usd
        levels_used = 0
        
        for level in levels:
            price = level['price']
            size = level['qty']
            if price <= 0: continue
            value_available = price * size
            if remaining_usd <= value_available:
                qty_needed = remaining_usd / price
                total_qty += qty_needed
                total_cost += remaining_usd
                remaining_usd = 0
                levels_used += 1
                break
            else:
                total_qty += size
                total_cost += value_available
                remaining_usd -= value_available
                levels_used += 1
                
        filled_usd = order_size_usd - remaining_usd
        avg_price = total_cost / total_qty if total_qty > 0 else 0
        slippage_bps = abs((avg_price - mid_price) / mid_price) * 10000 if mid_price > 0 else 0
        return {'filled': remaining_usd == 0, 'filled_usd': filled_usd, 'unfilled_usd': remaining_usd, 'levels_used': levels_used, 'avg_price': avg_price, 'slippage_bps': slippage_bps}


class AsterAPI:
    def __init__(self):
        self.base_url = "https://fapi.asterdex.com/fapi/v1/depth"
        self.headers = {'Content-Type': 'application/json'}

    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        params = {'symbol': symbol, 'limit': 50}
        try:
            response = requests.get(self.base_url, headers=self.headers, params=params, timeout=10)
            if response.status_code != 200:
                print(f"  > Aster Error {response.status_code} for {symbol}")
                return None
            data = response.json()
            if not data.get('bids') or not data.get('asks'): return None
            bids = [{'price': float(l[0]), 'qty': float(l[1])} for l in data['bids']]
            asks = [{'price': float(l[0]), 'qty': float(l[1])} for l in data['asks']]
            return {'bids': bids, 'asks': asks}
        except Exception as e:
            print(f"  > Error fetching Aster orderbook for {symbol}: {e}")
            return None

    def calculate_execution_cost(self, orderbook: Dict, order_size_usd: float) -> Optional[Dict]:
        if not orderbook: return None
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        if not asks or not bids: return None
        best_bid = bids[0]['price']
        best_ask = asks[0]['price']
        mid_price = (best_bid + best_ask) / 2
        buy_result = self._calculate_side(asks, order_size_usd, mid_price, 'buy')
        sell_result = self._calculate_side(bids, order_size_usd, mid_price, 'sell')
        if buy_result and sell_result:
            avg_slippage = (buy_result['slippage_bps'] + sell_result['slippage_bps']) / 2
            filled = buy_result['filled'] and sell_result['filled']
            return {
                'executed': True if filled else 'PARTIAL',
                'mid_price': mid_price,
                'slippage_bps': avg_slippage,
                'fee_bps': ASTER_TAKER_FEE_BPS,
                'buy': buy_result,
                'sell': sell_result,
                'filled': filled
            }
        return None

    def _calculate_side(self, levels, order_size_usd, mid_price, side):
        levels = sorted(levels, key=lambda x: x['price'], reverse=(side == 'sell'))
        total_qty = 0
        total_cost = 0
        remaining_usd = order_size_usd
        levels_used = 0
        for level in levels:
            price = level['price']
            size = level['qty']
            if price <= 0: continue
            value_available = price * size
            if remaining_usd <= value_available:
                qty_needed = remaining_usd / price
                total_qty += qty_needed
                total_cost += remaining_usd
                remaining_usd = 0
                levels_used += 1
                break
            else:
                total_qty += size
                total_cost += value_available
                remaining_usd -= value_available
                levels_used += 1
        filled_usd = order_size_usd - remaining_usd
        avg_price = total_cost / total_qty if total_qty > 0 else 0
        slippage_bps = abs((avg_price - mid_price) / mid_price) * 10000 if mid_price > 0 else 0
        return {'filled': remaining_usd == 0, 'filled_usd': filled_usd, 'unfilled_usd': remaining_usd, 'levels_used': levels_used, 'avg_price': avg_price, 'slippage_bps': slippage_bps}


class AvantisStatic:
    """
    Handles static fee and spread data for Avantis.
    No API fetching required.
    """
    def calculate_cost(self, asset_key: str, order_size_usd: float) -> Dict:
        # Default initialization
        open_fee_bps = 0.0
        close_fee_bps = 0.0
        spread_bps = 0.0
        
        key = asset_key.upper()
        
        # --- Logic Definitions ---
        forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
        indices = ['QQQ'] # SPY isn't in main config, but logic applies if added
        equities_list = ['HOOD', 'NVDA', 'AAPL', 'AMZN', 'GOOG', 'MSFT', 'META', 'TSLA', 'COIN']

        if key == 'GOLD': # XAU
            open_fee_bps = 6.0
            close_fee_bps = 0.0
            spread_bps = 0.0
        elif key == 'SILVER': # XAG
            open_fee_bps = 6.0
            close_fee_bps = 0.0
            spread_bps = 10.0
        elif key in forex_pairs:
            open_fee_bps = 3.0
            close_fee_bps = 0.0
            spread_bps = 0.0
        elif key in indices:
            open_fee_bps = 4.5
            close_fee_bps = 4.5
            spread_bps = 0.0
        elif key in equities_list:
            open_fee_bps = 4.5
            close_fee_bps = 4.5
            spread_bps = 2.5
        else:
             # Fallback for undefined assets (assume equity structure as safe default or 0)
             open_fee_bps = 4.5
             close_fee_bps = 4.5
             spread_bps = 2.5

        # Spread = 2 * Slippage. Therefore Slippage = Spread / 2
        slippage_bps = spread_bps / 2.0

        return {
            'executed': True, # Static data assumes infinite depth at quoted spread
            'mid_price': 0, # Not relevant for static calculation
            'slippage_bps': slippage_bps,
            'open_fee_bps': open_fee_bps,
            'close_fee_bps': close_fee_bps,
            'filled': True,
            'buy': {'filled_usd': order_size_usd, 'levels_used': 1},
            'sell': {'filled_usd': order_size_usd, 'levels_used': 1}
        }


class FeeComparator:
    def __init__(self):
        self.hyperliquid = HyperliquidAPI()
        self.lighter = LighterAPI()
        self.paradex = ParadexAPI()
        self.aster = AsterAPI()
        self.avantis = AvantisStatic()

    def compare_asset(self, asset_key: str, order_size_usd: float) -> Dict:
        config = ASSETS.get(asset_key.upper())
        if not config: return None

        result = {
            'asset': config.name,
            'symbol_key': config.symbol_key,
            'order_size_usd': order_size_usd,
            'hyperliquid': None,
            'lighter': None,
            'paradex': None,
            'aster': None,
            'avantis': None
        }

        # --- Hyperliquid Logic ---
        if config.hyperliquid_symbol:
            sig_figs_options = [None, 4, 3, 2]
            anchor_mid_price = None
            for sig_figs in sig_figs_options:
                sf_label = "Default" if sig_figs is None else str(sig_figs)
                print(f"  Fetching Hyperliquid {config.hyperliquid_symbol} (nSigFigs={sf_label})...")
                hl_orderbook, is_xyz = self.hyperliquid.get_orderbook(config.hyperliquid_symbol, n_sig_figs=sig_figs)
                if hl_orderbook is None: continue
                calc_result = self.hyperliquid.calculate_execution_cost(hl_orderbook, order_size_usd, anchor_mid_price=anchor_mid_price)
                if calc_result:
                    if anchor_mid_price is None:
                        anchor_mid_price = calc_result['mid_price']
                        print(f"  > Locked True Mid-Price: ${anchor_mid_price:,.4f}")
                    calc_result['is_xyz'] = is_xyz
                    if calc_result['filled']:
                        result['hyperliquid'] = calc_result
                        break
                    if calc_result.get('max_levels_hit', False):
                         print(f"  > Hit 20-level limit (Partial). Retrying with lower nSigFigs...")
                    result['hyperliquid'] = calc_result

        # --- Lighter Fetching ---
        if config.lighter_market_id:
            print(f"  Fetching Lighter orderbook for market {config.lighter_market_id}...")
            lighter_orderbook = self.lighter.get_orderbook(config.lighter_market_id)
            result['lighter'] = self.lighter.calculate_execution_cost(lighter_orderbook, order_size_usd)

        # --- Paradex Fetching ---
        if config.paradex_symbol:
            print(f"  Fetching Paradex orderbook for {config.paradex_symbol}...")
            paradex_orderbook = self.paradex.get_orderbook(config.paradex_symbol)
            result['paradex'] = self.paradex.calculate_execution_cost(paradex_orderbook, order_size_usd)

        # --- Aster Fetching ---
        if config.aster_symbol:
            print(f"  Fetching Aster orderbook for {config.aster_symbol}...")
            aster_orderbook = self.aster.get_orderbook(config.aster_symbol)
            result['aster'] = self.aster.calculate_execution_cost(aster_orderbook, order_size_usd)

        # --- Avantis Calculation (Static) ---
        # Avantis supports all assets defined in ASSETS for this specific script context
        print(f"  Calculating Avantis static costs...")
        result['avantis'] = self.avantis.calculate_cost(asset_key, order_size_usd)

        return result

    def print_result(self, result: Dict):
        order_size = result['order_size_usd']

        print("\n" + "=" * 135)
        print(f"SLIPPAGE ANALYSIS - {result['asset']}")
        print("=" * 135)

        # --- FEE STRUCTURE ---
        hl_open = HYPERLIQUID_TAKER_FEE_BPS
        hl_close = 0.0
        
        lt_open = LIGHTER_TAKER_FEE_BPS
        lt_close = 0.0

        px_open = PARADEX_TAKER_FEE_BPS
        px_close = 0.0
        
        as_open = ASTER_TAKER_FEE_BPS
        as_close = 0.0

        # Avantis fees are variable
        av = result.get('avantis')
        av_open = av['open_fee_bps'] if av else 0.0
        av_close = av['close_fee_bps'] if av else 0.0

        print("\nFee Structure (bps):")
        print(f"{'Type':<12} {'Hyperliquid':<15} {'Lighter':<15} {'Paradex':<15} {'Aster':<15} {'Avantis':<15}")
        print(f"{'-'*12} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
        print(f"{'Opening':<12} {hl_open:<15.2f} {lt_open:<15.2f} {px_open:<15.2f} {as_open:<15.2f} {av_open:<15.2f}")
        print(f"{'Closing':<12} {hl_close:<15.2f} {lt_close:<15.2f} {px_close:<15.2f} {as_close:<15.2f} {av_close:<15.2f}")
        print("=" * 135)

        hl = result.get('hyperliquid')
        lt = result.get('lighter')
        px = result.get('paradex')
        ast = result.get('aster')
        av = result.get('avantis')

        def print_exchange_data(name, data, open_fee, close_fee, tag=""):
            if not data: return
            print(f"\n--- {name}{tag} ---")
            print(f"${order_size:,.0f} Order:")
            slippage_bps = data.get('slippage_bps', 0)
            
            effective_spread = 2 * slippage_bps
            total_cost_bps = effective_spread + open_fee + close_fee
            
            print(f"  Avg Slippage:        {slippage_bps:.2f} bps")
            print(f"  Effective Spread:    {effective_spread:.2f} bps")
            print(f"  Opening Fee:         {open_fee:.2f} bps")
            print(f"  Closing Fee:         {close_fee:.2f} bps")
            print(f"  Total Cost:          {total_cost_bps:.2f} bps")

            if data.get('executed') == 'PARTIAL':
                print(f"  Filled:              X PARTIAL")
                print(f"  Unfilled Amount:     Buy: ${data['buy'].get('unfilled_usd', 0):,.2f}, Sell: ${data['sell'].get('unfilled_usd', 0):,.2f}")
            else:
                print(f"  Filled:              ✓ YES")
            
            if name != "AVANTIS":
                print(f"  Levels Used:         Buy: {data['buy'].get('levels_used', 0)}, Sell: {data['sell'].get('levels_used', 0)}")
                print(f"  Depth Used (USD):    Buy: ${data['buy'].get('filled_usd', order_size):,.0f}, Sell: ${data['sell'].get('filled_usd', order_size):,.0f}")
            else:
                print(f"  Note:                Static data (Assumes infinite depth)")

        if hl:
            hl_tag = " (XYZ)" if hl.get('is_xyz') else ""
            print_exchange_data("HYPERLIQUID", hl, hl_open, hl_close, hl_tag)
        if lt: print_exchange_data("LIGHTER", lt, lt_open, lt_close)
        if px: print_exchange_data("PARADEX", px, px_open, px_close)
        if ast: print_exchange_data("ASTER", ast, as_open, as_close)
        if av: print_exchange_data("AVANTIS", av, av_open, av_close)

        print()

        # Winner determination
        exchanges = []
        if hl and hl.get('executed') != 'PARTIAL':
            hl_total = (2 * hl.get('slippage_bps', 0)) + hl_open + hl_close
            exchanges.append(('Hyperliquid', hl_total))
        if lt and lt.get('executed') != 'PARTIAL':
            lt_total = (2 * lt.get('slippage_bps', 0)) + lt_open + lt_close
            exchanges.append(('Lighter', lt_total))
        if px and px.get('executed') != 'PARTIAL':
            px_total = (2 * px.get('slippage_bps', 0)) + px_open + px_close
            exchanges.append(('Paradex', px_total))
        if ast and ast.get('executed') != 'PARTIAL':
            as_total = (2 * ast.get('slippage_bps', 0)) + as_open + as_close
            exchanges.append(('Aster', as_total))
        if av:
            av_total = (2 * av.get('slippage_bps', 0)) + av_open + av_close
            exchanges.append(('Avantis', av_total))

        if len(exchanges) == 0:
            print(f"⚠️  Note: All exchanges have partial fills. Compare liquidity availability carefully.")
        elif len(exchanges) == 1:
            print(f"🏆 WINNER: {exchanges[0][0]} (only exchange with full liquidity)")
        else:
            winner = min(exchanges, key=lambda x: x[1])
            print(f"🏆 WINNER: {winner[0]} (Total Cost: {winner[1]:.2f} bps)")
            for name, cost in exchanges:
                if name != winner[0]:
                    savings = cost - winner[1]
                    savings_usd = order_size * savings / 10000
                    print(f"   vs {name:<12}: saves {savings:>6.2f} bps (${savings_usd:,.2f})")

        print("=" * 135)
        print("\nDefinitions:")
        print("  • Slippage: One-way price impact from mid-price")
        print("  • Effective Spread: Round-trip cost (2 × Slippage)")
        print("  • Total Cost: Effective Spread + Opening Fee + Closing Fee")
        print("=" * 135)


def print_available_assets():
    print("\nAvailable Assets:")
    print("-" * 100)
    print(f"{'Key':<10} {'Name':<25} {'HL':<5} {'Lighter':<9} {'Paradex':<9} {'Aster':<7} {'Avantis':<7}")
    print("-" * 100)
    for key, config in ASSETS.items():
        hl_status = "✓" if config.hyperliquid_symbol else "✗"
        lt_status = "✓" if config.lighter_market_id else "✗"
        px_status = "✓" if config.paradex_symbol else "✗"
        as_status = "✓" if config.aster_symbol else "✗"
        av_status = "✓" # Always true per requirement
        print(f"{key:<10} {config.name:<25} {hl_status:<5} {lt_status:<9} {px_status:<9} {as_status:<7} {av_status:<7}")
    print("-" * 100)


def main():
    comparator = FeeComparator()
    print("=" * 80)
    print("PERP DEX FEE COMPARISON TOOL")
    print("Hyperliquid vs Lighter vs Paradex vs Aster vs Avantis")
    print("=" * 80)

    while True:
        print("\n")
        print_available_assets()
        asset_input = input("\nEnter asset symbol (or 'quit' to exit): ").strip()
        if asset_input.lower() == 'quit':
            print("Goodbye!")
            break
        if asset_input.upper() not in ASSETS:
            print(f"❌ Asset '{asset_input}' not found. Please choose from the list above.")
            continue
        try:
            order_size_input = input("Enter order size in USD (e.g., 10000, 100000, 1000000): ").strip()
            order_size = float(order_size_input.replace(',', ''))
            if order_size <= 0:
                print("❌ Order size must be greater than 0")
                continue
        except ValueError:
            print("❌ Invalid order size. Please enter a number.")
            continue

        print(f"\nAnalyzing {asset_input.upper()} with ${order_size:,.0f} order size...")
        result = comparator.compare_asset(asset_input, order_size)
        if result: comparator.print_result(result)
        else: print("❌ Error comparing asset")

        continue_input = input("\nCompare another asset? (y/n): ").strip().lower()
        if continue_input != 'y':
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()