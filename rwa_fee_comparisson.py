#!/usr/bin/env python3
from __future__ import annotations  # Enable forward references for type hints
"""
Multi-Exchange Slippage Comparison API

Compares execution costs across 6 perp DEXs:
- Hyperliquid
- Lighter  
- Aster
- Avantis
- Ostium 
- Extended

Run with: python rwa_fee_comparisson.py
    
"""

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import requests
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Taker Fees
HYPERLIQUID_TAKER_FEE_BPS = 0.9
LIGHTER_TAKER_FEE_BPS = 0.0
ASTER_TAKER_FEE_BPS = 4.0
EXTENDED_TAKER_FEE_BPS = 2.5

# Maker Fees 
HYPERLIQUID_MAKER_FEE_BPS = 0.3
LIGHTER_MAKER_FEE_BPS = 0.0
ASTER_MAKER_FEE_BPS = 0.5
EXTENDED_MAKER_FEE_BPS = 0.0

# Ostium fees vary by asset class
OSTIUM_FEES_BPS = {
    # Commodities
    'XAUUSD': 3.0,   
    'XAGUSD': 15.0, 
    # Forex
    'EURUSD': 3.0,
    'GBPUSD': 3.0,
    'USDJPY': 1.5,  
    # Indices
    'SPXUSD': 5.0,
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
    aster_symbol: Optional[str]
    ostium_symbol: Optional[str]
    extended_symbol: Optional[str] = None  # Extended Exchange symbol

ASSETS = {
    # Commodities
    'XAU': AssetConfig('XAU/USD', 'XAU', 'commodity', None, 92, 'XAUUSDT', 'XAUUSD', 'XAU-USD'),
    'XAG': AssetConfig('XAG/USD', 'XAG', 'commodity', 'SILVER', 93, 'XAGUSDT', 'XAGUSD', 'XAG-USD'),
    
    # Forex
    'EURUSD': AssetConfig('EUR/USD', 'EURUSD', 'forex', 'EUR', 96, None, 'EURUSD', 'EUR-USD'),
    'GBPUSD': AssetConfig('GBP/USD', 'GBPUSD', 'forex', 'GBP', 97, None, 'GBPUSD', None),  
    'USDJPY': AssetConfig('USD/JPY', 'USDJPY', 'forex', 'JPY', 98, None, 'USDJPY', 'USDJPY-USD'),
    
    # MAG7 Stocks 
    'AAPL': AssetConfig('AAPL/USD', 'AAPL', 'stock', 'AAPL', 113, 'AAPLUSDT', 'AAPLUSD', None),
    'MSFT': AssetConfig('MSFT/USD', 'MSFT', 'stock', 'MSFT', 115, 'MSFTUSDT', 'MSFTUSD', None),
    'GOOG': AssetConfig('GOOG/USD', 'GOOG', 'stock', 'GOOGL', 116, 'GOOGUSDT', 'GOGUSD', None),
    'AMZN': AssetConfig('AMZN/USD', 'AMZN', 'stock', 'AMZN', 114, 'AMZNUSDT', 'AMZNUSD', None),
    'META': AssetConfig('META/USD', 'META', 'stock', 'META', 117, 'METAUSDT', 'METAUSD', None),
    'NVDA': AssetConfig('NVDA/USD', 'NVDA', 'stock', 'NVDA', 110, 'NVDAUSDT', 'NVDAUSD', None),
    'TSLA': AssetConfig('TSLA/USD', 'TSLA', 'stock', 'TSLA', 112, 'TSLAUSDT', 'TSLAUSD', None),
    
    # Indices
    'SPY': AssetConfig('SPY/USD', 'SPY', 'index', None, 128, None, 'SPYUSD', None),
    'QQQ': AssetConfig('QQQ/USD', 'QQQ', 'index', None, 129, None, 'QQQUSD', None),
    
    # Other
    'COIN': AssetConfig('COIN/USD', 'COIN', 'stock', 'COIN', 109, 'COINUSDT', 'COINUSD', None),
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
        # Disable SSL verification for macOS certificate issues
        self.session.verify = False
        # Suppress SSL warning
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    def get_fee_bps(self, ostium_symbol: str) -> float:
        """Get the opening fee for an Ostium asset based on their fee schedule."""
        return OSTIUM_FEES_BPS.get(ostium_symbol, 5.0)  
    
    def get_latest_price(self, asset: str, max_retries: int = 5) -> Optional[Dict]:
        """Get the latest price for a specific asset with retry logic."""
        url = f"{self.BASE_URL}/PricePublish/latest-price"
        params = {"asset": asset}
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=1000)
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if data and data.get('mid', 0) > 0:
                            return data
                    except ValueError:
                        pass
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)  
                else:
                    print(f"  > Ostium error for {asset} after {max_retries} attempts: {e}")
        return None
    
    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """
        Fetch price and create a "synthetic" orderbook.
        Ostium is oracle based not and orderbook.
        """
        price_data = self.get_latest_price(symbol)
        if not price_data:
            return None
        return price_data

    def normalize_orderbook(self, orderbook: Dict, depth_usd: float) -> Optional[StandardizedOrderbook]:
        """
        Normalize Ostium 'orderbook' (price data) to standard format.
        Uses depth_usd to simulate available liquidity at the oracle price.
        """
        if not orderbook:
            return None
        
        # Ostium returns bid, ask, mid on its API endpoint
        bid = float(orderbook.get('bid', 0))
        ask = float(orderbook.get('ask', 0))
        mid = float(orderbook.get('mid', 0))
        
        if bid <= 0 or ask <= 0:
            return None
            
        # Use requested depth for liquidity
        std_bids = [{'price': bid, 'qty': depth_usd / bid}]
        std_asks = [{'price': ask, 'qty': depth_usd / ask}]

        return StandardizedOrderbook(
            bids=std_bids,
            asks=std_asks,
            best_bid=bid,
            best_ask=ask,
            mid_price=mid,
            timestamp=time.time()
        )

    def calculate_execution_cost(self, asset: str, order_size_usd: float) -> Optional[Dict]:
        """Calculate execution cost using shared ExecutionCalculator."""
        # 1. Get synthetic orderbook
        raw_data = self.get_orderbook(asset)
        # Use order_size_usd + small buffer as depth to ensure full fill
        std_orderbook = self.normalize_orderbook(raw_data, depth_usd=order_size_usd * 1.01)
        
        if not std_orderbook:
            return None
        
        # 2. Get fees
        open_fee_bps = self.get_fee_bps(asset)
        close_fee_bps = 0.0
        
        # 3. Calculate using shared logic
        result = ExecutionCalculator.calculate_execution_cost(
            std_orderbook,
            order_size_usd,
            open_fee_bps=open_fee_bps,
            close_fee_bps=close_fee_bps
        )
        
        if result:
            # Add Ostium-specific metadata
            result['fee_bps'] = open_fee_bps
            result['maker_fee_bps'] = 0.0 
            if raw_data:
                result['is_market_open'] = raw_data.get('isMarketOpen', False)
        
        return result


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
            response = requests.post(self.base_url, json=payload, headers=self.headers, timeout=1000)
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

    def get_orderbook(self, symbol: str, n_sig_figs: Optional[int] = None) -> Optional[Dict]:
        raw_symbol = self.normalize_symbol(symbol)
        
        # Directly use XYZ (RWA) version
        coin = raw_symbol if raw_symbol.startswith("xyz:") else f"xyz:{raw_symbol}"
        return self._fetch_coin(coin, n_sig_figs)

    def normalize_orderbook(self, orderbook: Dict) -> Optional[StandardizedOrderbook]:
        """Normalize Hyperliquid orderbook to standard format."""
        if not orderbook:
            return None
        
        levels = orderbook.get('levels', [[], []])
        bids = levels[0] if len(levels) > 0 else []
        asks = levels[1] if len(levels) > 1 else []
        
        if not asks or not bids:
            return None
        
        try:
            best_bid = float(bids[0].get('px', 0))
            best_ask = float(asks[0].get('px', 0))
        except (ValueError, AttributeError, IndexError):
            return None
        
        if best_bid <= 0 or best_ask <= 0:
            return None
        
        # Convert to standard format: [{'price': float, 'qty': float}, ...]
        std_bids = []
        for b in bids:
            try:
                std_bids.append({'price': float(b.get('px', 0)), 'qty': float(b.get('sz', 0))})
            except (ValueError, AttributeError):
                continue
        
        std_asks = []
        for a in asks:
            try:
                std_asks.append({'price': float(a.get('px', 0)), 'qty': float(a.get('sz', 0))})
            except (ValueError, AttributeError):
                continue
        
        mid_price = (best_bid + best_ask) / 2
        
        return StandardizedOrderbook(
            bids=std_bids,
            asks=std_asks,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            timestamp=time.time()
        )

    def calculate_execution_cost(self, orderbook: Dict, order_size_usd: float, anchor_mid_price: Optional[float] = None) -> Optional[Dict]:
        """Calculate execution cost using shared ExecutionCalculator."""
        std_orderbook = self.normalize_orderbook(orderbook)
        if not std_orderbook:
            return None
        
        # Override mid_price if anchor provided (for consistent comparison across sig figs)
        if anchor_mid_price:
            std_orderbook = StandardizedOrderbook(
                bids=std_orderbook.bids,
                asks=std_orderbook.asks,
                best_bid=std_orderbook.best_bid,
                best_ask=std_orderbook.best_ask,
                mid_price=anchor_mid_price,
                timestamp=std_orderbook.timestamp
            )
        
        result = ExecutionCalculator.calculate_execution_cost(
            std_orderbook,
            order_size_usd,
            open_fee_bps=HYPERLIQUID_TAKER_FEE_BPS,
            close_fee_bps=0.0
        )
        
        if result:
            # Add Hyperliquid-specific fields
            result['fee_bps'] = HYPERLIQUID_TAKER_FEE_BPS
            result['maker_fee_bps'] = HYPERLIQUID_MAKER_FEE_BPS
            max_levels_hit = (result['buy']['levels_used'] >= len(std_orderbook.asks)) or \
                           (result['sell']['levels_used'] >= len(std_orderbook.bids))
            result['max_levels_hit'] = max_levels_hit
        
        return result

    def get_optimal_execution(self, symbol: str, order_size_usd: float) -> Optional[Dict]:
        """
        Calculates execution cost by cascading through orderbook precisions.
        Flow:
        1. Try Max Precision (None). If it fills the order, stop and return.
        2. If not, try 4 Significant Figures (deeper). If filled, stop and return.
        """
        
        # Precisions to try in order of preference: Max -> 4 due to slippage meassure accuracy
        # Max precision gives best price accuracy. Lower sig figs give more depth.
        precisions_to_try = [None, 4] 
        
        final_result = None
        
        for n_sig in precisions_to_try:
            # 1. Fetch
            raw_book = self.get_orderbook(symbol, n_sig_figs=n_sig)
            if not raw_book: continue
            
            # 2. Normalize
            std_book = self.normalize_orderbook(raw_book)
            if not std_book: continue
            
            # 3. Calculate
            result = ExecutionCalculator.calculate_execution_cost(
                std_book,
                order_size_usd,
                open_fee_bps=HYPERLIQUID_TAKER_FEE_BPS
            )
            
            if result:
                # Store this as the current best result
                # If we don't find a full fill later, this (or the next iteration's result) will be returned
                final_result = result
                final_result['fee_bps'] = HYPERLIQUID_TAKER_FEE_BPS
                final_result['maker_fee_bps'] = HYPERLIQUID_MAKER_FEE_BPS
                
                # Label the precision used
                if n_sig is None:
                    final_result['sig_figs'] = "Maximum"
                else:
                    final_result['sig_figs'] = n_sig
                
                # If it's a full fill, we are done. Stop looking.
                if result['filled']:
                    break
        
        # If loop finishes and we only have a partial fill (from 4 sig figs), final_result corresponds to that.
        
        if final_result:
            # Metadata
            final_result['is_xyz'] = True
            # Ensure symbol has xyz: prefix for display
            display_symbol = symbol if "xyz" in str(symbol) else f"xyz:{symbol}"
            final_result['symbol'] = display_symbol
            
        return final_result


class LighterAPI:
    def __init__(self):
        self.base_url = "https://mainnet.zklighter.elliot.ai/api/v1"
        self.headers = {'Content-Type': 'application/json'}

    def get_orderbook(self, market_id: int) -> Optional[Dict]:
        url = f"{self.base_url}/orderBookOrders?market_id={market_id}&limit=100"
        try:
            response = requests.get(url, headers=self.headers, timeout=1000)
            response.raise_for_status()
            return response.json()
        except Exception: return None

    def normalize_orderbook(self, orderbook: Dict) -> Optional[StandardizedOrderbook]:
        """Normalize Lighter orderbook to standard format."""
        if not orderbook:
            return None
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        if not asks or not bids:
            return None

        best_bid = float(bids[0].get('price', 0))
        best_ask = float(asks[0].get('price', 0))
        if best_bid <= 0 or best_ask <= 0:
            return None
        
        mid_price = (best_bid + best_ask) / 2

        # Convert to standard format
        std_bids = [{'price': float(b.get('price', 0)), 'qty': float(b.get('remaining_base_amount', 0))} for b in bids]
        std_asks = [{'price': float(a.get('price', 0)), 'qty': float(a.get('remaining_base_amount', 0))} for a in asks]

        return StandardizedOrderbook(
            bids=std_bids,
            asks=std_asks,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            timestamp=time.time()
        )

    def calculate_execution_cost(self, orderbook: Dict, order_size_usd: float) -> Optional[Dict]:
        """Calculate execution cost using shared ExecutionCalculator."""
        std_orderbook = self.normalize_orderbook(orderbook)
        if not std_orderbook:
            return None
        
        result = ExecutionCalculator.calculate_execution_cost(
            std_orderbook,
            order_size_usd,
            open_fee_bps=LIGHTER_TAKER_FEE_BPS,
            close_fee_bps=0.0
        )
        
        if result:
            result['fee_bps'] = LIGHTER_TAKER_FEE_BPS
            result['maker_fee_bps'] = LIGHTER_MAKER_FEE_BPS
        
        return result

class AsterAPI:
    def __init__(self):
        self.base_url = "https://fapi.asterdex.com/fapi/v1/depth"
        self.headers = {'Content-Type': 'application/json'}

    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        params = {'symbol': symbol, 'limit': 1000}  
        try:
            response = requests.get(self.base_url, headers=self.headers, params=params, timeout=1000)
            if response.status_code != 200:
                return None
            data = response.json()
            if not data.get('bids') or not data.get('asks'): return None
            bids = [{'price': float(l[0]), 'qty': float(l[1])} for l in data['bids']]
            asks = [{'price': float(l[0]), 'qty': float(l[1])} for l in data['asks']]
            return {'bids': bids, 'asks': asks}
        except Exception:
            return None

    def normalize_orderbook(self, orderbook: Dict) -> Optional[StandardizedOrderbook]:
        """Normalize Aster orderbook to standard format."""
        if not orderbook:
            return None
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        if not asks or not bids:
            return None
        
        best_bid = bids[0]['price']
        best_ask = asks[0]['price']
        if best_bid <= 0 or best_ask <= 0:
            return None
        
        mid_price = (best_bid + best_ask) / 2

        return StandardizedOrderbook(
            bids=bids,  
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            timestamp=time.time()
        )

    def calculate_execution_cost(self, orderbook: Dict, order_size_usd: float) -> Optional[Dict]:
        """Calculate execution cost using shared ExecutionCalculator."""
        std_orderbook = self.normalize_orderbook(orderbook)
        if not std_orderbook:
            return None
        
        result = ExecutionCalculator.calculate_execution_cost(
            std_orderbook,
            order_size_usd,
            open_fee_bps=ASTER_TAKER_FEE_BPS,
            close_fee_bps=0.0
        )
        
        if result:
            result['fee_bps'] = ASTER_TAKER_FEE_BPS
            result['maker_fee_bps'] = ASTER_MAKER_FEE_BPS
        
        return result


class AvantisStatic:
    """Handles static fee and spread data for Avantis."""
    def calculate_cost(self, asset_key: str, order_size_usd: float) -> Dict:
        open_fee_bps = 0.0
        close_fee_bps = 0.0
        spread_bps = 0.0  # Total spread (will be split into buy/sell)
        
        key = asset_key.upper()
        
        forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
        indices = ['QQQ', 'SPY']
        equities_list = ['HOOD', 'NVDA', 'AAPL', 'AMZN', 'GOOG', 'MSFT', 'META', 'TSLA', 'COIN']

        if key == 'XAU':
            open_fee_bps = 6.0
            close_fee_bps = 0.0
            spread_bps = 0.0  
        elif key == 'XAG':
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
            spread_bps = 5.0 
        else:
            open_fee_bps = 4.5
            close_fee_bps = 4.5
            spread_bps = 5.0

        buy_slippage_bps = spread_bps / 2.0
        sell_slippage_bps = spread_bps / 2.0
        total_slippage_bps = buy_slippage_bps + sell_slippage_bps  # Same as spread_bps
        total_cost_bps = total_slippage_bps + open_fee_bps + close_fee_bps

        return {
            'executed': True,
            'mid_price': 0,
            'slippage_bps': total_slippage_bps,
            'buy_slippage_bps': buy_slippage_bps,
            'sell_slippage_bps': sell_slippage_bps,
            'open_fee_bps': open_fee_bps,
            'close_fee_bps': close_fee_bps,
            'maker_fee_bps': 0.0,
            'total_cost_bps': total_cost_bps,
            'filled': True,
            'buy': {'filled': True, 'filled_usd': order_size_usd, 'unfilled_usd': 0, 'levels_used': 1, 'slippage_bps': buy_slippage_bps},
            'sell': {'filled': True, 'filled_usd': order_size_usd, 'unfilled_usd': 0, 'levels_used': 1, 'slippage_bps': sell_slippage_bps},
            'timestamp': time.time()
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
        try:
            url = f"{self.BASE_URL}/info/markets/{market}/orderbook"
            response = self.session.get(url, timeout=1000)
            if response.status_code != 200:
                return None
            data = response.json()
            if data.get('status') != 'OK':
                return None
            return data.get('data')
        except Exception as e:
            print(f"Extended API error for {market}: {e}")
            return None
    
    def normalize_orderbook(self, orderbook: Dict) -> Optional[StandardizedOrderbook]:
        if not orderbook:
            return None
        
        bids = orderbook.get('bid', [])
        asks = orderbook.get('ask', [])
        
        if not bids or not asks:
            return None
        
        best_bid = float(bids[0]['price'])
        best_ask = float(asks[0]['price'])
        
        if best_bid <= 0 or best_ask <= 0:
            return None
        
        mid_price = (best_bid + best_ask) / 2
        
        # Convert to standard format
        std_bids = [{'price': float(b['price']), 'qty': float(b['qty'])} for b in bids]
        std_asks = [{'price': float(a['price']), 'qty': float(a['qty'])} for a in asks]

        return StandardizedOrderbook(
            bids=std_bids,
            asks=std_asks,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            timestamp=time.time()
        )

    def calculate_execution_cost(self, orderbook: Dict, order_size_usd: float) -> Optional[Dict]:
        """Calculate execution cost using shared ExecutionCalculator."""
        std_orderbook = self.normalize_orderbook(orderbook)
        if not std_orderbook:
            return None
        
        open_fee_bps = EXTENDED_TAKER_FEE_BPS
        close_fee_bps = EXTENDED_TAKER_FEE_BPS
        
        result = ExecutionCalculator.calculate_execution_cost(
            std_orderbook,
            order_size_usd,
            open_fee_bps=open_fee_bps,
            close_fee_bps=close_fee_bps
        )
        
        if result:
            result['maker_fee_bps'] = EXTENDED_MAKER_FEE_BPS

        return result


# =============================================================================
# SLIPPAGE EXECUTION CALCULATOR
# =============================================================================
# All orderbook-based exchanges normalize their data and delegate here.
# This ensures consistent calculation across all exchanges.
# =============================================================================

@dataclass
class StandardizedOrderbook:
    """
    Common orderbook format for all exchanges.
    
    All exchange APIs should normalize their orderbook data to this format
    before passing to ExecutionCalculator.
    """
    bids: List[Dict[str, float]]  # [{'price': float, 'qty': float}, ...] sorted best to worst
    asks: List[Dict[str, float]]  # [{'price': float, 'qty': float}, ...] sorted best to worst (lowest first)
    best_bid: float
    best_ask: float
    mid_price: float
    timestamp: float = 0.0


class ExecutionCalculator:
    """
    Shared calculation logic for all orderbook-based exchanges.
    
    Formulas:
        mid_price = (best_bid + best_ask) / 2
        avg_execution_price = total_cost / total_qty  (from walking the book)
        slippage_bps = abs((avg_execution_price - mid_price) / mid_price) * 10000
        total_cost_bps = (2 * slippage_bps) + open_fee_bps + close_fee_bps
    """
    
    @staticmethod
    def calculate_execution_cost(
        orderbook: 'StandardizedOrderbook',
        order_size_usd: float,
        open_fee_bps: float = 0.0,
        close_fee_bps: float = 0.0
    ) -> Optional[Dict]:
        """
        Calculate execution cost from a standardized orderbook.
        
        Args:
            orderbook: Standardized orderbook with bids/asks
            order_size_usd: Order size in USD
            open_fee_bps: Opening fee in basis points
            close_fee_bps: Closing fee in basis points
            
        Returns:
            Standardized result dict with slippage, fees, and execution details
        """
        if not orderbook or not orderbook.bids or not orderbook.asks:
            return None
        
        mid_price = orderbook.mid_price
        
        # Calculate execution for both sides
        buy_result = ExecutionCalculator._walk_book(
            orderbook.asks, order_size_usd, mid_price, side='buy'
        )
        sell_result = ExecutionCalculator._walk_book(
            orderbook.bids, order_size_usd, mid_price, side='sell'
        )
        
        if not buy_result or not sell_result:
            return None
        
        # Slippage from both sides
        buy_slippage_bps = buy_result['slippage_bps']
        sell_slippage_bps = sell_result['slippage_bps']
        avg_slippage_bps = (buy_slippage_bps + sell_slippage_bps) / 2
        filled = buy_result['filled'] and sell_result['filled']
        
        # Determine which side is unfilled (if any)
        buy_unfilled = buy_result['unfilled_usd']
        sell_unfilled = sell_result['unfilled_usd']
        buy_partial = not buy_result['filled']
        sell_partial = not sell_result['filled']
        
        if buy_partial and sell_partial:
            unfilled_side = 'both'
        elif buy_partial:
            unfilled_side = 'buy'
        elif sell_partial:
            unfilled_side = 'sell'
        else:
            unfilled_side = None  # Both fully filled
        
        # Calculate total cost
        total_cost_bps = avg_slippage_bps + open_fee_bps + close_fee_bps
        
        return {
            'executed': True if filled else 'PARTIAL',
            'mid_price': mid_price,
            'best_bid': orderbook.best_bid,
            'best_ask': orderbook.best_ask,
            'slippage_bps': avg_slippage_bps,
            'buy_slippage_bps': buy_slippage_bps,
            'sell_slippage_bps': sell_slippage_bps,
            'open_fee_bps': open_fee_bps,
            'close_fee_bps': close_fee_bps,
            'total_cost_bps': total_cost_bps,
            'filled': filled,
            'order_size_usd': order_size_usd,
            'filled_usd': min(buy_result['filled_usd'], sell_result['filled_usd']),
            'unfilled_usd': max(buy_unfilled, sell_unfilled),
            'unfilled_side': unfilled_side,
            'buy': buy_result,
            'sell': sell_result,
            'timestamp': orderbook.timestamp
        }
    
    @staticmethod
    def _walk_book(
        levels: List[Dict[str, float]],
        order_size_usd: float,
        mid_price: float,
        side: str = 'buy'
    ) -> Optional[Dict]:
        
        """
        Walk through orderbook levels to calculate execution.
        
        Args:
            levels: List of {'price': float, 'qty': float} dicts
            order_size_usd: Order size in USD
            mid_price: Mid price for slippage calculation
            side: 'buy' or 'sell'
            
        Returns:
            Execution result with avg_price, slippage_bps, filled status
        """

        if not levels:
            return None
        
        # Sort levels: asks ascending (best=lowest), bids descending (best=highest)
        sorted_levels = sorted(
            levels,
            key=lambda x: x['price'],
            reverse=(side == 'sell')
        )
        
        # Calculate total cost 
        # Gathered from walking the orderbook till the order is filled

        unfilled_order_amount_usd = order_size_usd
        total_qty = 0.0
        total_cost = 0.0
        levels_used = 0
        
        for level in sorted_levels:
            price = level['price']
            qty = level['qty']
            
            if price <= 0 or qty <= 0:
                continue
            
            value_available = price * qty
            
            if unfilled_order_amount_usd <= value_available:
                # This level can fill the remaining order
                qty_needed = unfilled_order_amount_usd / price
                total_qty += qty_needed
                total_cost += unfilled_order_amount_usd
                unfilled_order_amount_usd = 0
                levels_used += 1
                break
            else:
                # Consume entire level
                total_qty += qty
                total_cost += value_available
                unfilled_order_amount_usd -= value_available
                levels_used += 1
        
        # Calculate results
        filled_usd = order_size_usd - unfilled_order_amount_usd
        avg_price = total_cost / total_qty 
        
        # Slippage = abs((avg_execution_price - mid_price) / mid_price) * 10000
        slippage_bps = abs((avg_price - mid_price) / mid_price) * 10000 
        
        return {
            'filled': unfilled_order_amount_usd == 0,
            'filled_usd': filled_usd,
            'unfilled_usd': unfilled_order_amount_usd,
            'levels_used': levels_used,
            'avg_price': avg_price,
            'slippage_bps': slippage_bps
        }

    @staticmethod
    def calculate_hybrid_execution_cost(
        primary_book: 'StandardizedOrderbook',
        secondary_book: 'StandardizedOrderbook',
        order_size_usd: float,
        open_fee_bps: float = 0.0,
        close_fee_bps: float = 0.0
    ) -> Optional[Dict]:
        """
        Calculate execution cost by filling from primary book first, then secondary.
        Straightforward "Stitch" logic: Fill what you can from Primary, then fill remainder from Secondary.
        """
        if not primary_book and not secondary_book:
            return None
            
        # If primary missing, treat secondary as primary
        if not primary_book:
            return ExecutionCalculator.calculate_execution_cost(secondary_book, order_size_usd, open_fee_bps, close_fee_bps)
        
        # If secondary missing, normal calc on primary
        if not secondary_book:
             return ExecutionCalculator.calculate_execution_cost(primary_book, order_size_usd, open_fee_bps, close_fee_bps)

        mid_price = primary_book.mid_price # Anchor to primary (fairer) price

        # --- Helper for Hybrid Walk ---
        def walk_hybrid(prim_levels, sec_levels, side):
            # 1. Fill from Primary
            prim_res = ExecutionCalculator._walk_book(prim_levels, order_size_usd, mid_price, side)
            
            # If fully filled, we are done
            if prim_res['filled']:
                return prim_res 
            
            # 2. Fill Remainder from Secondary
            unfilled = prim_res['unfilled_usd']
            filled_amount = prim_res['filled_usd']
            
            # Cost so far
            avg_prim = prim_res['avg_price'] if prim_res['avg_price'] else 0
            cost_prim = filled_amount # filled_usd is already the value in USD
            qty_prim = filled_amount / avg_prim if avg_prim > 0 else 0
            
            # Determine threshold price from primary to filter secondary
            # Sort primary levels same way _walk_book did
            sorted_prim = sorted(prim_levels, key=lambda x: x['price'], reverse=(side == 'sell'))
            levels_used = prim_res.get('levels_used', 0)
            
            last_prim_price = 0
            if levels_used > 0 and levels_used <= len(sorted_prim):
                last_prim_price = sorted_prim[levels_used - 1]['price']
            else:
                 # If full fill or something odd, use worst price
                 last_prim_price = sorted_prim[-1]['price'] if sorted_prim else 0

            # Filter secondary to avoid double counting / crossing
            # If Buy: only take asks > last_prim_price
            # If Sell: only take bids < last_prim_price

            # Find qty used at last price in primary to deduct from secondary if overlap
            qty_at_boundary = 0
            for l in prim_levels:
                if l['price'] == last_prim_price:
                    qty_at_boundary += l['qty']

            filtered_sec = []
            for lvl in sec_levels:
                price = lvl['price']
                qty = lvl['qty']
                include = False
                adjusted_qty = qty
                
                if side == 'buy':
                    if price > last_prim_price:
                        include = True
                    elif price == last_prim_price:
                        # Overlap
                        adjusted_qty = max(0, qty - qty_at_boundary)
                        if adjusted_qty > 0: include = True
                elif side == 'sell':
                    if price < last_prim_price:
                        include = True
                    elif price == last_prim_price:
                         # Overlap
                        adjusted_qty = max(0, qty - qty_at_boundary)
                        if adjusted_qty > 0: include = True
                
                if include:
                    new_lvl = lvl.copy()
                    new_lvl['qty'] = adjusted_qty
                    filtered_sec.append(new_lvl)
            
            sec_res = ExecutionCalculator._walk_book(filtered_sec, unfilled, mid_price, side)
            
            if sec_res['filled']:
                # Combine
                cost_sec = sec_res['filled_usd'] # Value in USD
                qty_sec = sec_res['filled_usd'] / sec_res['avg_price'] if sec_res['avg_price'] > 0 else 0
                
                total_qty = qty_prim + qty_sec
                total_cost = cost_prim + cost_sec
                final_avg = total_cost / total_qty if total_qty > 0 else 0
                
                slip = abs((final_avg - mid_price) / mid_price) * 10000
                
                return {
                    'filled': True,
                    'filled_usd': order_size_usd,
                    'avg_price': final_avg,
                    'slippage_bps': slip
                }
            else:
                return {'filled': False, 'slippage_bps': 0}

        buy_result = walk_hybrid(primary_book.asks, secondary_book.asks, 'buy')
        sell_result = walk_hybrid(primary_book.bids, secondary_book.bids, 'sell')

        if not buy_result or not sell_result:
            return None

        buy_slippage_bps = buy_result['slippage_bps']
        sell_slippage_bps = sell_result['slippage_bps']
        avg_slippage_bps = (buy_slippage_bps + sell_slippage_bps) / 2
        filled = buy_result['filled'] and sell_result['filled']
        total_cost_bps = avg_slippage_bps + open_fee_bps + close_fee_bps

        return {
            'executed': True if filled else 'PARTIAL',
            'mid_price': mid_price,
            'slippage_bps': avg_slippage_bps,
            'buy_slippage_bps': buy_slippage_bps,
            'sell_slippage_bps': sell_slippage_bps,
            'open_fee_bps': open_fee_bps,
            'close_fee_bps': close_fee_bps,
            'total_cost_bps': total_cost_bps,
            'filled': filled,
            'buy': buy_result,
            'sell': sell_result,
            'timestamp': primary_book.timestamp
        }




class FeeComparator:
    def __init__(self):

        self.hyperliquid = HyperliquidAPI()
        self.lighter = LighterAPI()
        self.aster = AsterAPI()
        self.avantis = AvantisStatic()
        self.ostium = OstiumAPI()
        self.extended = ExtendedAPI()

    def compare_asset(self, asset_key: str, order_size_usd: float, order_type: str = 'taker') -> Dict:
        config = ASSETS.get(asset_key.upper())
        if not config: return None



        result = {
            'asset': config.name,
            'symbol_key': config.symbol_key,
            'order_size_usd': order_size_usd,
            'hyperliquid': None,
            'lighter': None,
            'aster': None,
            'avantis': None,
            'ostium': None,
            'extended': None,
            # Include symbol info for display
            'symbols': {
                'hyperliquid': config.hyperliquid_symbol,
                'lighter': config.symbol_key if config.lighter_market_id else None,
                'aster': config.aster_symbol,
                'avantis': config.symbol_key,
                'ostium': config.ostium_symbol,
                'extended': config.extended_symbol
            }
        }

        # --- Hyperliquid ---
        if config.hyperliquid_symbol:
            hyperliquid_result = self.hyperliquid.get_optimal_execution(config.hyperliquid_symbol, order_size_usd)
            if hyperliquid_result:
                result['hyperliquid'] = hyperliquid_result
                result['symbols']['hyperliquid'] = hyperliquid_result['symbol']

        # --- Lighter ---
        if config.lighter_market_id:
            lighter_orderbook = self.lighter.get_orderbook(config.lighter_market_id)
            lighter_result = self.lighter.calculate_execution_cost(lighter_orderbook, order_size_usd)
            if lighter_result:
                lighter_result['symbol'] = config.symbol_key
            result['lighter'] = lighter_result



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

        # Override for Maker orders (Zero Slippage)
        if order_type == 'maker':
            for ex in ['hyperliquid', 'lighter', 'aster', 'avantis', 'ostium', 'extended']:
                if result.get(ex):
                    result[ex]['slippage_bps'] = 0.0
                    if 'buy' in result[ex]: result[ex]['buy']['slippage_bps'] = 0.0
                    if 'sell' in result[ex]: result[ex]['sell']['slippage_bps'] = 0.0

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
    order_size = float(data.get('order_size', 1000000))
    
    order_type = data.get('order_type', 'taker').lower()
    
    if asset not in ASSETS:
        return jsonify({'error': f'Asset {asset} not found'}), 400
    
    result = comparator.compare_asset(asset, order_size, order_type=order_type)
    
    if not result:
        return jsonify({'error': 'Failed to compare asset'}), 500
    
    # Calculate totals and determine winner
    exchanges = []
    
    # Fee structures
    if order_type == 'maker':
        # MAKER Mode: Use Maker Fees for Orderbook, Taker/Flat for Oracle
        fee_structure = {
            'hyperliquid': {'open': HYPERLIQUID_MAKER_FEE_BPS, 'close': HYPERLIQUID_MAKER_FEE_BPS},
            'lighter': {'open': LIGHTER_MAKER_FEE_BPS, 'close': LIGHTER_MAKER_FEE_BPS},
            'aster': {'open': ASTER_MAKER_FEE_BPS, 'close': ASTER_MAKER_FEE_BPS},
            # Extended uses Maker Fee (0.0)
            'extended': {'open': EXTENDED_MAKER_FEE_BPS, 'close': EXTENDED_MAKER_FEE_BPS} 
        }
    else:
        # TAKER Mode: Standard Taker Fees
        fee_structure = {
            'hyperliquid': {'open': HYPERLIQUID_TAKER_FEE_BPS, 'close': HYPERLIQUID_TAKER_FEE_BPS},
            'lighter': {'open': LIGHTER_TAKER_FEE_BPS, 'close': 0.0},
            'aster': {'open': ASTER_TAKER_FEE_BPS, 'close': 0.0},
            'extended': {'open': EXTENDED_TAKER_FEE_BPS, 'close': EXTENDED_TAKER_FEE_BPS}
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
    
    for exchange_name in ['hyperliquid', 'lighter', 'aster', 'avantis', 'ostium', 'extended']:
        ex_data = result.get(exchange_name)
        if ex_data:
            fees = fee_structure.get(exchange_name, {'open': 0, 'close': 0})
            slippage = ex_data.get('slippage_bps', 0)
            
            # Avantis: slippage only occurs once 
            effective_spread = slippage if exchange_name == 'avantis' else 2 * slippage
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


@app.route('/api/compare/<asset>', methods=['GET'])
def compare_get(asset):
    """
    GET endpoint for comparing slippage across exchanges.
    
    URL: /api/compare/<asset>?size=1000000&order_type=taker
    
    Parameters:
        asset (path): Asset symbol (e.g., XAU, XAG, AAPL, NVDA)
        size (query): Order size in USD (default: 1000000)
        order_type (query): 'taker' or 'maker' (default: taker)
    
    Example:
        GET /api/compare/XAU?size=50000
        GET /api/compare/NVDA?size=1000000&order_type=maker
    """
    asset = asset.upper()
    order_size = float(request.args.get('size', 1000000))
    order_type = request.args.get('order_type', 'taker').lower()
    
    if asset not in ASSETS:
        return jsonify({'error': f'Asset {asset} not found', 'available_assets': list(ASSETS.keys())}), 400
    
    result = comparator.compare_asset(asset, order_size, order_type=order_type)
    
    if not result:
        return jsonify({'error': 'Failed to compare asset'}), 500
    
    # Calculate totals and determine winner (same logic as POST endpoint)
    exchanges = []
    
    if order_type == 'maker':
        fee_structure = {
            'hyperliquid': {'open': HYPERLIQUID_MAKER_FEE_BPS, 'close': HYPERLIQUID_MAKER_FEE_BPS},
            'lighter': {'open': LIGHTER_MAKER_FEE_BPS, 'close': LIGHTER_MAKER_FEE_BPS},
            'aster': {'open': ASTER_MAKER_FEE_BPS, 'close': ASTER_MAKER_FEE_BPS},
            'extended': {'open': EXTENDED_MAKER_FEE_BPS, 'close': EXTENDED_MAKER_FEE_BPS}
        }
    else:
        fee_structure = {
            'hyperliquid': {'open': HYPERLIQUID_TAKER_FEE_BPS, 'close': HYPERLIQUID_TAKER_FEE_BPS},
            'lighter': {'open': LIGHTER_TAKER_FEE_BPS, 'close': 0.0},
            'aster': {'open': ASTER_TAKER_FEE_BPS, 'close': 0.0},
            'extended': {'open': EXTENDED_TAKER_FEE_BPS, 'close': EXTENDED_TAKER_FEE_BPS}
        }
    
    os_data = result.get('ostium')
    if os_data:
        fee_structure['ostium'] = {'open': os_data.get('fee_bps', 5.0), 'close': 0.0}
    
    av = result.get('avantis')
    if av:
        fee_structure['avantis'] = {'open': av.get('open_fee_bps', 0), 'close': av.get('close_fee_bps', 0)}
    
    for exchange_name in ['hyperliquid', 'lighter', 'aster', 'avantis', 'ostium', 'extended']:
        ex_data = result.get(exchange_name)
        if ex_data:
            fees = fee_structure.get(exchange_name, {'open': 0, 'close': 0})
            slippage = ex_data.get('slippage_bps', 0)
            effective_spread = slippage if exchange_name == 'avantis' else 2 * slippage
            total_cost = effective_spread + fees['open'] + fees['close']
            
            ex_data['effective_spread_bps'] = effective_spread
            ex_data['open_fee_bps'] = fees['open']
            ex_data['close_fee_bps'] = fees['close']
            ex_data['total_cost_bps'] = total_cost
            ex_data['exchange'] = exchange_name
            
            if ex_data.get('executed') != 'PARTIAL':
                exchanges.append({'name': exchange_name, 'total_cost': total_cost, 'filled': ex_data.get('filled', True)})
    
    if exchanges:
        winner = min(exchanges, key=lambda x: x['total_cost'])
        result['winner'] = winner['name']
        result['winner_cost_bps'] = winner['total_cost']
    
    return jsonify(result)


# =============================================================================
# WEBSOCKET EVENT HANDLERS
# =============================================================================

@socketio.on('compare')
def handle_compare(data):
    """Handle WebSocket compare request."""
    try:
        asset = data.get('asset')
        order_size = data.get('order_size', 1000000)
        order_type = data.get('order_type', 'taker')
        
        if not asset or asset not in ASSETS:
            emit('compare_error', {'error': f'Unknown asset: {asset}'})
            return
        
        config = ASSETS[asset]
        result = {
            'asset': asset,
            'name': config.name,
            'order_size_usd': order_size,
            'order_type': order_type
        }
        
        # Fetch data from all exchanges (reusing existing logic)
        hl_api = HyperliquidAPI()
        lighter_api = LighterAPI()
        aster_api = AsterAPI()
        avantis_static = AvantisStatic()
        ostium_api = OstiumAPI()
        extended_api = ExtendedAPI()
        
        # Hyperliquid
        if config.hyperliquid_symbol:
            hl_result = hl_api.get_optimal_execution(config.hyperliquid_symbol, order_size)
            if hl_result:
                result['hyperliquid'] = hl_result
        
        # Lighter
        if config.lighter_market_id:
            lighter_book = lighter_api.get_orderbook(config.lighter_market_id)
            if lighter_book:
                lighter_result = lighter_api.calculate_execution_cost(lighter_book, order_size)
                if lighter_result:
                    result['lighter'] = lighter_result
        
        # Aster
        if config.aster_symbol:
            aster_book = aster_api.get_orderbook(config.aster_symbol)
            if aster_book:
                aster_result = aster_api.calculate_execution_cost(aster_book, order_size)
                if aster_result:
                    result['aster'] = aster_result
        
        # Avantis (static fees)
        avantis_result = avantis_static.calculate_cost(asset, order_size)
        if avantis_result:
            result['avantis'] = avantis_result
        
        # Ostium
        if config.ostium_symbol:
            ostium_result = ostium_api.calculate_execution_cost(config.ostium_symbol, order_size)
            if ostium_result:
                result['ostium'] = ostium_result
        
        # Extended
        if config.extended_symbol:
            extended_book = extended_api.get_orderbook(config.extended_symbol)
            if extended_book:
                extended_result = extended_api.calculate_execution_cost(extended_book, order_size)
                if extended_result:
                    result['extended'] = extended_result
        
        # Calculate total costs and determine winner (same logic as HTTP endpoint)
        exchanges = []
        
        if order_type == 'maker':
            fee_structure = {
                'hyperliquid': {'open': HYPERLIQUID_MAKER_FEE_BPS, 'close': HYPERLIQUID_MAKER_FEE_BPS},
                'lighter': {'open': LIGHTER_MAKER_FEE_BPS, 'close': LIGHTER_MAKER_FEE_BPS},
                'aster': {'open': ASTER_MAKER_FEE_BPS, 'close': ASTER_MAKER_FEE_BPS},
                'extended': {'open': EXTENDED_MAKER_FEE_BPS, 'close': EXTENDED_MAKER_FEE_BPS}
            }
        else:
            fee_structure = {
                'hyperliquid': {'open': HYPERLIQUID_TAKER_FEE_BPS, 'close': HYPERLIQUID_TAKER_FEE_BPS},
                'lighter': {'open': LIGHTER_TAKER_FEE_BPS, 'close': 0.0},
                'aster': {'open': ASTER_TAKER_FEE_BPS, 'close': 0.0},
                'extended': {'open': EXTENDED_TAKER_FEE_BPS, 'close': EXTENDED_TAKER_FEE_BPS}
            }
        
        os_data = result.get('ostium')
        if os_data:
            fee_structure['ostium'] = {'open': os_data.get('fee_bps', 5.0), 'close': 0.0}
        
        av = result.get('avantis')
        if av:
            fee_structure['avantis'] = {'open': av.get('open_fee_bps', 0), 'close': av.get('close_fee_bps', 0)}
        
        for exchange_name in ['hyperliquid', 'lighter', 'aster', 'avantis', 'ostium', 'extended']:
            ex_data = result.get(exchange_name)
            if ex_data:
                fees = fee_structure.get(exchange_name, {'open': 0, 'close': 0})
                slippage = ex_data.get('slippage_bps', 0)
                effective_spread = slippage if exchange_name == 'avantis' else 2 * slippage
                total_cost = effective_spread + fees['open'] + fees['close']
                
                ex_data['effective_spread_bps'] = effective_spread
                ex_data['open_fee_bps'] = fees['open']
                ex_data['close_fee_bps'] = fees['close']
                ex_data['total_cost_bps'] = total_cost
                ex_data['exchange'] = exchange_name
                
                if ex_data.get('executed') != 'PARTIAL':
                    exchanges.append({'name': exchange_name, 'total_cost': total_cost, 'filled': ex_data.get('filled', True)})
        
        if exchanges:
            winner = min(exchanges, key=lambda x: x['total_cost'])
            result['winner'] = winner['name']
            result['winner_cost_bps'] = winner['total_cost']
        
        emit('compare_result', result)
        
    except Exception as e:
        emit('compare_error', {'error': str(e)})


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 FIXED FEE & AVERAGE SLIPPAGE COMPARISON API SERVER")
    print("=" * 60)
    print("Open http://127.0.0.1:5001 in your browser")
    print("WebSocket support enabled")
    print("=" * 60 + "\n")
    socketio.run(app, debug=True, port=5001)
