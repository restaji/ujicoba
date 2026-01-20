#!/usr/bin/env python3
"""
Multi-Exchange Slippage Comparison API

Compares execution costs across 6 perp DEXs:
- Hyperliquid
- Lighter  
- Paradex
- Aster
- Avantis
- Ostium (oracle-based)

Run with: python slippage_api.py
Visit: http://localhost:5000
"""

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import requests
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

app = Flask(__name__)
CORS(app)

# --- CONSTANTS ---
HYPERLIQUID_TAKER_FEE_BPS = 4.5
LIGHTER_TAKER_FEE_BPS = 0.0
PARADEX_TAKER_FEE_BPS = 0.0
ASTER_TAKER_FEE_BPS = 4.0
VARIATIONAL_TAKER_FEE_BPS = 0.0  

# Ostium fees vary by asset class (from docs)
OSTIUM_FEES_BPS = {
    # Commodities
    'XAUUSD': 3.0,   # Gold
    'XAGUSD': 15.0,  # Silver
    'XPTUSD': 20.0,  # Platinum
    'XPDUSD': 20.0,  # Palladium
    'CLUSD': 10.0,   # Oil
    'HGUSD': 15.0,   # Copper
    # Forex
    'EURUSD': 3.0,
    'GBPUSD': 3.0,
    'USDJPY': 3.0,
    'USDCAD': 3.0,
    'USDCHF': 3.0,
    'AUDUSD': 3.0,
    'NZDUSD': 3.0,
    'USDMXN': 5.0,  # Exception
    # Indices
    'SPXUSD': 5.0,
    'NDXUSD': 5.0,
    'DJIUSD': 5.0,
    'DAXEUR': 5.0,
    'NIKJPY': 5.0,
    'HSIHKD': 5.0,
    'FTSEGBP': 5.0,
    # Stocks
    'NVDAUSD': 5.0,
    'GOGUSD': 5.0,
    'METAUSD': 5.0,
    'MSFTUSD': 5.0,
    'AMZNUSD': 5.0,
    'AAPLUSD': 5.0,
    'TSLAUSD': 5.0,
    'COINUSD': 5.0,
    'HOODUSD': 5.0,
    'PLTRUSD': 5.0,
    'AMDUSD': 5.0,
    'NFLXUSD': 5.0,
    'ORCLUSD': 5.0,
    'COSTUSD': 5.0,
    'XOMUSD': 5.0,
    'CVXUSD': 5.0,
}


# ASSETS - MAG7 + COIN + Commodities + Forex
# extended_symbol is for Extended Exchange (Starknet)
@dataclass
class AssetConfig:
    name: str
    symbol_key: str
    asset_class: str  # 'commodity', 'forex', 'index', 'stock'
    hyperliquid_symbol: Optional[str]
    lighter_market_id: Optional[int]
    paradex_symbol: Optional[str]
    aster_symbol: Optional[str]
    ostium_symbol: Optional[str]
    extended_symbol: Optional[str] = None  # Extended Exchange symbol
    variational_symbol: Optional[str] = None  # Variational symbol

ASSETS = {
    # Commodities
    'GOLD': AssetConfig('Gold', 'XAU', 'commodity', 'PAXG', 92, 'PAXG', 'XAUUSDT', 'XAUUSD', 'XAU-USD', 'PAXG'),
    'SILVER': AssetConfig('Silver', 'XAG', 'commodity', 'SILVER', 93, None, 'XAGUSDT', 'XAGUSD', 'XAG-USD'),
    
    # Forex
    'EURUSD': AssetConfig('EUR/USD', 'EURUSD', 'forex', 'EUR', 96, None, None, 'EURUSD', 'EUR-USD'),
    'GBPUSD': AssetConfig('GBP/USD', 'GBPUSD', 'forex', 'GBP', 97, None, None, 'GBPUSD', None),  # Not on Extended
    'USDJPY': AssetConfig('USD/JPY', 'USDJPY', 'forex', 'JPY', 98, None, None, 'USDJPY', 'USDJPY-USD'),
    
    # MAG7 Stocks (Not on Extended Exchange)
    'AAPL': AssetConfig('Apple', 'AAPL', 'stock', 'AAPL', 113, None, 'AAPLUSDT', 'AAPLUSD', None),
    'MSFT': AssetConfig('Microsoft', 'MSFT', 'stock', 'MSFT', 115, None, 'MSFTUSDT', 'MSFTUSD', None),
    'GOOG': AssetConfig('Google', 'GOOG', 'stock', 'GOOGL', 116, None, 'GOOGUSDT', 'GOGUSD', None),
    'AMZN': AssetConfig('Amazon', 'AMZN', 'stock', 'AMZN', 114, None, 'AMZNUSDT', 'AMZNUSD', None),
    'META': AssetConfig('Meta', 'META', 'stock', 'META', 117, None, 'METAUSDT', 'METAUSD', None),
    'NVDA': AssetConfig('NVIDIA', 'NVDA', 'stock', 'NVDA', 110, None, 'NVDAUSDT', 'NVDAUSD', None),
    'TSLA': AssetConfig('Tesla', 'TSLA', 'stock', 'TSLA', 112, None, 'TSLAUSDT', 'TSLAUSD', None),
    
    # Other
    'COIN': AssetConfig('Coinbase', 'COIN', 'stock', 'COIN', 109, None, 'COINUSDT', 'COINUSD', None),
}




class OstiumAPI:
    """Client for interacting with Ostium's REST API."""
    
    BASE_URL = "https://metadata-backend.ostium.io"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def get_fee_bps(self, ostium_symbol: str) -> float:
        """Get the opening fee for an Ostium asset based on their fee schedule."""
        return OSTIUM_FEES_BPS.get(ostium_symbol, 5.0)  # Default to 5 bps
    
    def get_latest_price(self, asset: str, max_retries: int = 3) -> Optional[Dict]:
        """Get the latest price for a specific asset with retry logic."""
        url = f"{self.BASE_URL}/PricePublish/latest-price"
        params = {"asset": asset}
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if data and data.get('mid', 0) > 0:
                        return data
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)  # Wait 1 second before retry
                else:
                    print(f"  > Ostium error for {asset} after {max_retries} attempts: {e}")
        return None
    
    def calculate_execution_cost(self, asset: str, order_size_usd: float) -> Optional[Dict]:
        """
        Calculate execution cost based on oracle spread + opening fee.
        
        Ostium uses oracle-provided bid/ask, so spread is static
        regardless of order size (no orderbook depth).
        Opening fee varies by asset class.
        """
        price_data = self.get_latest_price(asset)
        if not price_data:
            return None
        
        bid = price_data.get('bid', 0)
        ask = price_data.get('ask', 0)
        mid = price_data.get('mid', 0)
        
        if bid <= 0 or ask <= 0 or mid <= 0:
            return None
        
        spread = ask - bid
        spread_bps = (spread / mid) * 10000
        slippage_bps = spread_bps / 2  # Half spread on entry
        
        # Get asset-specific fee
        fee_bps = self.get_fee_bps(asset)
        
        return {
            'executed': True,
            'best_bid': bid,
            'best_ask': ask,
            'mid_price': mid,
            'slippage_bps': slippage_bps,
            'fee_bps': fee_bps,
            'is_market_open': price_data.get('isMarketOpen', False),
            'filled': True,
            'buy': {
                'filled': True,
                'filled_usd': order_size_usd,
                'unfilled_usd': 0,
                'levels_used': 1,
                'avg_price': ask,
                'slippage_bps': slippage_bps
            },
            'sell': {
                'filled': True,
                'filled_usd': order_size_usd,
                'unfilled_usd': 0,
                'levels_used': 1,
                'avg_price': bid,
                'slippage_bps': slippage_bps
            }
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
            
        except Exception:
            return None

    def get_orderbook(self, symbol: str, n_sig_figs: Optional[int] = None) -> Tuple[Optional[Dict], bool]:
        raw_symbol = self.normalize_symbol(symbol)
        
        # Try XYZ (RWA) version first
        rwa_coin = f"xyz:{raw_symbol}"
        book = self._fetch_coin(rwa_coin, n_sig_figs)
        if book:
            return book, True
        
        # Fall back to regular symbol
        book = self._fetch_coin(raw_symbol, n_sig_figs)
        if book:
            return book, False
        
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
        url = f"{self.base_url}/orderBookOrders?market_id={market_id}&limit=150"
        try:
            response = requests.get(url, headers=self.headers, timeout=1000)
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
            response = requests.get(url, headers=self.headers, params=params, timeout=1000)
            if response.status_code == 404:
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
                return None
            data = response.json()
            if not data.get('bids') or not data.get('asks'): return None
            bids = [{'price': float(l[0]), 'qty': float(l[1])} for l in data['bids']]
            asks = [{'price': float(l[0]), 'qty': float(l[1])} for l in data['asks']]
            return {'bids': bids, 'asks': asks}
        except Exception:
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
    """Handles static fee and spread data for Avantis."""
    def calculate_cost(self, asset_key: str, order_size_usd: float) -> Dict:
        open_fee_bps = 0.0
        close_fee_bps = 0.0
        spread_bps = 0.0
        
        key = asset_key.upper()
        
        forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
        indices = ['QQQ']
        equities_list = ['HOOD', 'NVDA', 'AAPL', 'AMZN', 'GOOG', 'MSFT', 'META', 'TSLA', 'COIN']

        if key == 'GOLD':
            open_fee_bps = 6.0
            close_fee_bps = 0.0
            spread_bps = 0.0
        elif key == 'SILVER':
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
            open_fee_bps = 4.5
            close_fee_bps = 4.5
            spread_bps = 2.5

        slippage_bps = spread_bps / 2.0

        return {
            'executed': True,
            'mid_price': 0,
            'slippage_bps': slippage_bps,
            'open_fee_bps': open_fee_bps,
            'close_fee_bps': close_fee_bps,
            'filled': True,
            'buy': {'filled_usd': order_size_usd, 'levels_used': 1},
            'sell': {'filled_usd': order_size_usd, 'levels_used': 1}
        }


class ExtendedAPI:
    """Client for Extended Exchange (Starknet) orderbook data."""
    
    BASE_URL = "https://api.starknet.extended.exchange/api/v1"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def get_orderbook(self, market: str) -> Optional[Dict]:
        """Fetch orderbook for a given market (e.g., 'XAU-USD')."""
        try:
            url = f"{self.BASE_URL}/info/markets/{market}/orderbook"
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                return None
            data = response.json()
            if data.get('status') != 'OK':
                return None
            return data.get('data')
        except Exception as e:
            print(f"Extended API error for {market}: {e}")
            return None
    
    def calculate_execution_cost(self, orderbook: Dict, order_size_usd: float) -> Optional[Dict]:
        """Calculate execution cost from Extended orderbook."""
        if not orderbook:
            return None
        
        bids = orderbook.get('bid', [])
        asks = orderbook.get('ask', [])
        
        if not bids or not asks:
            return None
        
        # Get mid price
        best_bid = float(bids[0]['price'])
        best_ask = float(asks[0]['price'])
        mid_price = (best_bid + best_ask) / 2
        spread_bps = ((best_ask - best_bid) / mid_price) * 10000
        
        # Calculate buy and sell execution
        buy_result = self._calculate_side(asks, order_size_usd, mid_price)
        sell_result = self._calculate_side(bids, order_size_usd, mid_price)
        
        if not buy_result or not sell_result:
            return None
        
        avg_slippage = (abs(buy_result['slippage_bps']) + abs(sell_result['slippage_bps'])) / 2
        filled = buy_result['filled'] and sell_result['filled']
        
        # Extended Exchange fees: 2.5 bps taker fee
        open_fee_bps = 2.5
        close_fee_bps = 2.5
        effective_spread_bps = spread_bps / 2
        total_cost_bps = avg_slippage + effective_spread_bps + open_fee_bps + close_fee_bps
        
        return {
            'executed': 'FULL' if filled else 'PARTIAL',
            'mid_price': mid_price,
            'spread_bps': spread_bps,
            'slippage_bps': avg_slippage,
            'effective_spread_bps': effective_spread_bps,
            'open_fee_bps': open_fee_bps,
            'close_fee_bps': close_fee_bps,
            'total_cost_bps': total_cost_bps,
            'filled': filled,
            'buy': buy_result,
            'sell': sell_result
        }
    
    def _calculate_side(self, levels: List, order_size_usd: float, mid_price: float) -> Optional[Dict]:
        """Calculate execution for one side of the book."""
        if not levels:
            return None
        
        remaining_usd = order_size_usd
        total_qty = 0
        total_cost = 0
        levels_used = 0
        
        for level in levels:
            price = float(level['price'])
            qty = float(level['qty'])
            value_available = price * qty
            
            if value_available >= remaining_usd:
                qty_needed = remaining_usd / price
                total_qty += qty_needed
                total_cost += remaining_usd
                remaining_usd = 0
                levels_used += 1
                break
            else:
                total_qty += qty
                total_cost += value_available
                remaining_usd -= value_available
                levels_used += 1
        
        filled_usd = order_size_usd - remaining_usd
        avg_price = total_cost / total_qty if total_qty > 0 else 0
        slippage_bps = abs((avg_price - mid_price) / mid_price) * 10000 if mid_price > 0 else 0
        
        return {
            'filled': remaining_usd == 0,
            'filled_usd': filled_usd,
            'unfilled_usd': remaining_usd,
            'levels_used': levels_used,
            'avg_price': avg_price,
            'slippage_bps': slippage_bps
        }


class VariationalAPI:
    """Client for Variational DEX market data."""
    
    BASE_URL = "https://omni-client-api.prod.ap-northeast-1.variational.io"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        self._stats_cache = None
        self._cache_time = 0
    
    def get_stats(self) -> Optional[Dict]:
        """Fetch market stats (cached for 1000 seconds)."""
        import time
        current_time = time.time()
        if self._stats_cache and (current_time - self._cache_time) < 1000:
            return self._stats_cache
        
        try:
            url = f"{self.BASE_URL}/metadata/stats"
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                return None
            self._stats_cache = response.json()
            self._cache_time = current_time
            return self._stats_cache
        except Exception as e:
            print(f"Variational API error: {e}")
            return None
    
    def calculate_execution_cost(self, ticker: str, order_size_usd: float) -> Optional[Dict]:
        """Calculate execution cost using pre-calculated bid/ask quotes with interpolation."""
        stats = self.get_stats()
        if not stats:
            return None
        
        # Find the listing for the given ticker
        listings = stats.get('listings', [])
        listing = None
        for l in listings:
            if l.get('ticker') == ticker:
                listing = l
                break
        
        if not listing:
            return None
        
        quotes = listing.get('quotes', {})
        mark_price = float(listing.get('mark_price', 0))
        
        if mark_price <= 0:
            return None
        
        # Build a list of (size, spread_bps) tuples from available quotes
        size_spread_data = []
        for size_key, size_val in [('size_1k', 1000), ('size_100k', 100000), ('size_1m', 1000000)]:
            quote = quotes.get(size_key)
            if quote:
                bid = float(quote.get('bid', 0))
                ask = float(quote.get('ask', 0))
                if bid > 0 and ask > 0:
                    spread = ask - bid
                    spread_bps = (spread / mark_price) * 10000
                    size_spread_data.append((size_val, spread_bps))
        
        if not size_spread_data:
            return None
        
        # Sort by size
        size_spread_data.sort(key=lambda x: x[0])
        
        # Interpolate/extrapolate to find spread for the given order size
        if order_size_usd <= size_spread_data[0][0]:
            # Use smallest bucket
            spread_bps = size_spread_data[0][1]
        elif order_size_usd >= size_spread_data[-1][0]:
            # Extrapolate beyond largest bucket using slope from last two points
            if len(size_spread_data) >= 2:
                s1, sp1 = size_spread_data[-2]
                s2, sp2 = size_spread_data[-1]
                slope = (sp2 - sp1) / (s2 - s1) if s2 != s1 else 0
                spread_bps = sp2 + slope * (order_size_usd - s2)
            else:
                spread_bps = size_spread_data[-1][1]
        else:
            # Interpolate between two brackets
            for i in range(len(size_spread_data) - 1):
                s1, sp1 = size_spread_data[i]
                s2, sp2 = size_spread_data[i + 1]
                if s1 <= order_size_usd <= s2:
                    ratio = (order_size_usd - s1) / (s2 - s1)
                    spread_bps = sp1 + ratio * (sp2 - sp1)
                    break
        
        # Slippage is half the spread (entry only)
        slippage_bps = spread_bps / 2
        
        # Get best bid/ask from smallest bucket (Top of Book) for mid_price calculation
        quote_1k = quotes.get('size_1k') or quotes.get('size_100k') or quotes.get('size_1m')
        
        bid = float(quote_1k.get('bid', 0)) if quote_1k else mark_price
        ask = float(quote_1k.get('ask', 0)) if quote_1k else mark_price
        
        if bid > 0 and ask > 0:
            mid_price = (bid + ask) / 2
        else:
            mid_price = mark_price
        
        return {
            'executed': True,
            'mid_price': mid_price,
            'mark_price': mark_price,
            'best_bid': bid,
            'best_ask': ask,
            'slippage_bps': slippage_bps,
            'fee_bps': VARIATIONAL_TAKER_FEE_BPS,
            'filled': True,
            'buy': {
                'filled': True,
                'filled_usd': order_size_usd,
                'unfilled_usd': 0,
                'levels_used': 1,
                'avg_price': ask,
                'slippage_bps': slippage_bps
            },
            'sell': {
                'filled': True,
                'filled_usd': order_size_usd,
                'unfilled_usd': 0,
                'levels_used': 1,
                'avg_price': bid,
                'slippage_bps': slippage_bps
            }
        }

class FeeComparator:
    def __init__(self):
        self.hyperliquid = HyperliquidAPI()
        self.lighter = LighterAPI()
        self.paradex = ParadexAPI()
        self.aster = AsterAPI()
        self.avantis = AvantisStatic()
        self.ostium = OstiumAPI()
        self.extended = ExtendedAPI()
        self.variational = VariationalAPI()

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
            'avantis': None,
            'ostium': None,
            'extended': None,
            'variational': None,
            # Include symbol info for display
            'symbols': {
                'hyperliquid': config.hyperliquid_symbol,
                'lighter': config.symbol_key if config.lighter_market_id else None,
                'paradex': f"{config.paradex_symbol}-USD-PERP" if config.paradex_symbol else None,
                'aster': config.aster_symbol,
                'avantis': config.symbol_key,
                'ostium': config.ostium_symbol,
                'extended': config.extended_symbol,
                'variational': config.variational_symbol
            }
        }

        # --- Hyperliquid ---
        if config.hyperliquid_symbol:
            sig_figs_options = [None, 4, 3, 2]
            anchor_mid_price = None
            for sig_figs in sig_figs_options:
                hl_orderbook, is_xyz = self.hyperliquid.get_orderbook(config.hyperliquid_symbol, n_sig_figs=sig_figs)
                if hl_orderbook is None: continue
                calc_result = self.hyperliquid.calculate_execution_cost(hl_orderbook, order_size_usd, anchor_mid_price=anchor_mid_price)
                if calc_result:
                    if anchor_mid_price is None:
                        anchor_mid_price = calc_result['mid_price']
                    calc_result['is_xyz'] = is_xyz
                    calc_result['symbol'] = f"xyz:{config.hyperliquid_symbol}" if is_xyz else config.hyperliquid_symbol
                    if calc_result['filled']:
                        result['hyperliquid'] = calc_result
                        # Update symbol in symbols dict
                        result['symbols']['hyperliquid'] = calc_result['symbol']
                        break
                    result['hyperliquid'] = calc_result
                    result['symbols']['hyperliquid'] = calc_result['symbol']

        # --- Lighter ---
        if config.lighter_market_id:
            lighter_orderbook = self.lighter.get_orderbook(config.lighter_market_id)
            lighter_result = self.lighter.calculate_execution_cost(lighter_orderbook, order_size_usd)
            if lighter_result:
                lighter_result['symbol'] = config.symbol_key
            result['lighter'] = lighter_result

        # --- Paradex ---
        if config.paradex_symbol:
            paradex_orderbook = self.paradex.get_orderbook(config.paradex_symbol)
            paradex_result = self.paradex.calculate_execution_cost(paradex_orderbook, order_size_usd)
            if paradex_result:
                paradex_result['symbol'] = f"{config.paradex_symbol}-USD-PERP"
            result['paradex'] = paradex_result

        # --- Aster ---
        if config.aster_symbol:
            aster_orderbook = self.aster.get_orderbook(config.aster_symbol)
            aster_result = self.aster.calculate_execution_cost(aster_orderbook, order_size_usd)
            if aster_result:
                aster_result['symbol'] = config.aster_symbol
            result['aster'] = aster_result

        # --- Avantis (Static) ---
        avantis_result = self.avantis.calculate_cost(asset_key, order_size_usd)
        if avantis_result:
            avantis_result['symbol'] = config.symbol_key
        result['avantis'] = avantis_result

        # --- Ostium ---
        if config.ostium_symbol:
            ostium_result = self.ostium.calculate_execution_cost(config.ostium_symbol, order_size_usd)
            if ostium_result:
                ostium_result['symbol'] = config.ostium_symbol
            result['ostium'] = ostium_result

        # --- Extended Exchange ---
        if config.extended_symbol:
            extended_orderbook = self.extended.get_orderbook(config.extended_symbol)
            extended_result = self.extended.calculate_execution_cost(extended_orderbook, order_size_usd)
            if extended_result:
                extended_result['symbol'] = config.extended_symbol
            result['extended'] = extended_result

        # --- Variational ---
        if config.variational_symbol:
            variational_result = self.variational.calculate_execution_cost(config.variational_symbol, order_size_usd)
            if variational_result:
                variational_result['symbol'] = config.variational_symbol
            result['variational'] = variational_result

        return result


# Initialize comparator
comparator = FeeComparator()


# --- FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/assets', methods=['GET'])
def get_assets():
    """Return list of available assets."""
    assets_list = []
    for key, config in ASSETS.items():
        assets_list.append({
            'key': key,
            'name': config.name,
            'symbol': config.symbol_key,
            'exchanges': {
                'hyperliquid': config.hyperliquid_symbol is not None,
                'lighter': config.lighter_market_id is not None,
                'paradex': config.paradex_symbol is not None,
                'aster': config.aster_symbol is not None,
                'avantis': True,
                'ostium': config.ostium_symbol is not None
            }
        })
    return jsonify({'assets': assets_list})


@app.route('/api/compare', methods=['POST'])
def compare():
    """Compare slippage across exchanges for given asset and order size."""
    data = request.json
    asset = data.get('asset', '').upper()
    order_size = float(data.get('order_size', 100000))
    
    if asset not in ASSETS:
        return jsonify({'error': f'Asset {asset} not found'}), 400
    
    result = comparator.compare_asset(asset, order_size)
    
    if not result:
        return jsonify({'error': 'Failed to compare asset'}), 500
    
    # Calculate totals and determine winner
    exchanges = []
    
    # Fee structures
    fee_structure = {
        'hyperliquid': {'open': HYPERLIQUID_TAKER_FEE_BPS, 'close': 0.0},
        'lighter': {'open': LIGHTER_TAKER_FEE_BPS, 'close': 0.0},
        'paradex': {'open': PARADEX_TAKER_FEE_BPS, 'close': 0.0},
        'aster': {'open': ASTER_TAKER_FEE_BPS, 'close': 0.0},
    }
    
    # Ostium has variable fees per asset
    os_data = result.get('ostium')
    if os_data:
        fee_structure['ostium'] = {
            'open': os_data.get('fee_bps', 5.0),
            'close': 0.0
        }
    
    # Avantis has variable fees
    av = result.get('avantis')
    if av:
        fee_structure['avantis'] = {
            'open': av.get('open_fee_bps', 0),
            'close': av.get('close_fee_bps', 0)
        }
    
    for exchange_name in ['hyperliquid', 'lighter', 'paradex', 'aster', 'avantis', 'ostium', 'variational']:
        ex_data = result.get(exchange_name)
        if ex_data:
            fees = fee_structure.get(exchange_name, {'open': 0, 'close': 0})
            slippage = ex_data.get('slippage_bps', 0)
            effective_spread = 2 * slippage
            total_cost = effective_spread + fees['open'] + fees['close']
            
            ex_data['effective_spread_bps'] = effective_spread
            ex_data['open_fee_bps'] = fees['open']
            ex_data['close_fee_bps'] = fees['close']
            ex_data['total_cost_bps'] = total_cost
            ex_data['exchange'] = exchange_name
            
            if ex_data.get('executed') != 'PARTIAL':
                exchanges.append({
                    'name': exchange_name,
                    'total_cost': total_cost,
                    'filled': ex_data.get('filled', True)
                })
    
    # Determine winner
    winner = None
    if exchanges:
        winner = min(exchanges, key=lambda x: x['total_cost'])
        result['winner'] = winner['name']
        result['winner_cost_bps'] = winner['total_cost']
    
    return jsonify(result)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 SLIPPAGE COMPARISON API SERVER")
    print("=" * 60)
    print("Open http://127.0.0.1:5001 in your browser")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5001)
