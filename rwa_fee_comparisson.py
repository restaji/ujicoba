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
ASTER_TAKER_FEE_BPS = 4.0

# Maker Fees
HYPERLIQUID_MAKER_FEE_BPS = 0.3
ASTER_MAKER_FEE_BPS = 0.5

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
    'XAU': AssetConfig('XAU/USD', 'XAU', 'commodity', 'GOLD', 92, 'XAUUSDT', 'XAUUSD', 'XAU-USD'),
    'XAG': AssetConfig('XAG/USD', 'XAG', 'commodity', 'SILVER', 93, 'XAGUSDT', 'XAGUSD', 'XAG-USD'),
    
    # Forex
    'EURUSD': AssetConfig('EUR/USD', 'EURUSD', 'forex', 'EUR', 96, None, 'EURUSD', 'EUR-USD'),
    'GBPUSD': AssetConfig('GBP/USD', 'GBPUSD', 'forex', 'GBP', 97, None, 'GBPUSD', None),  
    'USDJPY': AssetConfig('USD/JPY', 'USDJPY', 'forex', 'JPY', 98, None, 'USDJPY', 'USDJPY-USD'),
    
    # MAG7 Stocks 
    'AAPL': AssetConfig('AAPL/USD', 'AAPL', 'stock', 'AAPL', 113, 'AAPLUSDT', 'AAPLUSD', None),
    'MSFT': AssetConfig('MSFT/USD', 'MSFT', 'stock', 'MSFT', 115, 'MSFTUSDT', 'MSFTUSD', None),
    'GOOG': AssetConfig('GOOG/USD', 'GOOG', 'stock', 'GOOGL', 116, 'GOOGUSDT', 'GOOGUSD', None),
    'AMZN': AssetConfig('AMZN/USD', 'AMZN', 'stock', 'AMZN', 114, 'AMZNUSDT', 'AMZNUSD', None),
    'META': AssetConfig('META/USD', 'META', 'stock', 'META', 117, 'METAUSDT', 'METAUSD', None),
    'NVDA': AssetConfig('NVDA/USD', 'NVDA', 'stock', 'NVDA', 110, 'NVDAUSDT', 'NVDAUSD', None),
    'TSLA': AssetConfig('TSLA/USD', 'TSLA', 'stock', 'TSLA', 112, 'TSLAUSDT', 'TSLAUSD', None),
    
    # Indices
    'SPY': AssetConfig('SPY/USD', 'SPY', 'index', None, 128, None, 'SPYUSD', None),
    'QQQ': AssetConfig('QQQ/USD', 'QQQ', 'index', None, 129, 'QQQUSDT', 'QQQUSD', None),
    
    # Other
    'COIN': AssetConfig('COIN/USD', 'COIN', 'stock', 'COIN', 109, 'COINUSDT', 'COINUSD', None),
}


class OstiumAPI:
    """Client for interacting with Ostium's REST API."""
    
    BASE_URL = "https://metadata-backend.ostium.io"
    PAIRS_URL = "https://app.ostium.com/api/pairs"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        # Disable SSL verification for macOS certificate issues
        self.session.verify = False
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Load metadata from Ostium pairs API (no fallbacks)
        self.metadata_cache = self._load_cache()
    
    def _load_cache(self):
        """
        Load fee and leverage metadata from Ostium `pairs` API.
        Also fetches 'seasons' data to override fees with 'newFee' if applicable.
        """
        cache = {}
        pair_id_map = {} # id -> symbol
        
        # 1. Load Pairs (Base Metadata)
        try:
            response = self.session.get(self.PAIRS_URL, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    for pair in data:
                        base = pair.get('from')
                        quote = pair.get('to')
                        p_id = pair.get('id')
                        if not base or not quote:
                            continue
                        symbol = f"{base}{quote}"
                        
                        # Store ID mapping
                        if p_id is not None:
                            pair_id_map[p_id] = symbol

                        maker_fee_p = pair.get('makerFeeP')
                        taker_fee_p = pair.get('takerFeeP')
                        group = pair.get('group') or {}

                        # Get max leverage
                        pair_max_lev = pair.get('maxLeverage')
                        maker_max_lev = pair.get('makerMaxLeverage')
                        group_max_lev = group.get('maxLeverage')
                        
                        max_lev = None
                        for lev_val in [pair_max_lev, maker_max_lev, group_max_lev]:
                            if lev_val is not None:
                                try:
                                    lev_float = float(lev_val)
                                    if lev_float > 0:
                                        max_lev = lev_float
                                        break
                                except (TypeError, ValueError):
                                    continue

                        if max_lev is not None:
                            max_lev = max_lev / 100.0

                        taker_fee_bps = None
                        maker_fee_bps = None

                        if isinstance(taker_fee_p, (int, float, str)):
                            try:
                                taker_fee_bps = float(taker_fee_p) / 10000.0
                            except (TypeError, ValueError):
                                pass
                        if isinstance(maker_fee_p, (int, float, str)):
                            try:
                                maker_fee_bps = float(maker_fee_p) / 10000.0
                            except (TypeError, ValueError):
                                pass

                        if taker_fee_bps is not None:
                            cache[symbol] = {
                                'fee_bps': taker_fee_bps,
                                'maker_fee_bps': maker_fee_bps if maker_fee_bps is not None else 0.0,
                                'max_leverage': float(max_lev) if max_lev is not None else None
                            }
        except Exception as e:
            print(f"Error loading Ostium metadata from pairs API: {e}")

        # 2. Load Seasons (Fee Overrides)
        try:
            seasons_url = "https://onlypoints.ostium.io/api/seasons/current"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                'Accept': 'application/json',
                'Referer': 'https://app.ostium.io/'
            }
            # Disable verification for this specific call if needed or global?
            # Global verification was disabled in __init__, so it should apply here.
            response = self.session.get(seasons_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                s_data = response.json()
                season = s_data.get('season', {})
                mode = season.get('mode', {})
                assets = mode.get('assets', [])
                
                if isinstance(assets, list):
                    for asset_item in assets:
                        a_id = asset_item.get('assetId')
                        new_fee = asset_item.get('newFee') # e.g. 0.05
                        
                        if a_id is not None and new_fee is not None and a_id in pair_id_map:
                            symbol = pair_id_map[a_id]
                            if symbol in cache:
                                # Convert newFee to bps. Assuming 0.05 means 0.05% -> 5 bps
                                new_fee_bps = float(new_fee) * 100.0
                                cache[symbol]['fee_bps'] = new_fee_bps
                                cache[symbol]['maker_fee_bps'] = new_fee_bps
                                # print(f"Ostium: Overridden fee for {symbol} (ID {a_id}) to {new_fee_bps} bps")
        except Exception as e:
            print(f"Error loading Ostium seasons data: {e}")

        return cache
    
    def get_fee_bps(self, ostium_symbol: str) -> Optional[float]:
        """Get the opening fee for an Ostium asset. Returns None if not available."""
        data = self.metadata_cache.get(ostium_symbol)
        if data:
            return data.get('fee_bps')
        return None
    
    def get_maker_fee_bps(self, ostium_symbol: str) -> Optional[float]:
        """Get the maker fee for an Ostium asset. Returns None if not available."""
        data = self.metadata_cache.get(ostium_symbol)
        if data:
            return data.get('maker_fee_bps')
        return None
        
    def get_max_leverage(self, ostium_symbol: str) -> Optional[float]:
        """Get max leverage."""
        data = self.metadata_cache.get(ostium_symbol)
        if data:
            return data.get('max_leverage')
        return None
    
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
        # 1. Get synthetic orderbook (for oracle bid/ask spread)
        raw_data = self.get_orderbook(asset)
        # Use order_size_usd 
        std_orderbook = self.normalize_orderbook(raw_data, depth_usd=order_size_usd * 1.01)
        
        if not std_orderbook:
            return None
        
        # 2. Get fees from API (no fallbacks)
        open_fee_bps = self.get_fee_bps(asset)
        maker_fee_bps = self.get_maker_fee_bps(asset)
        # Ostium charges fee only on open (close fee is 0)
        close_fee_bps = 0.0
        
        # If no fee data available from API, return None
        if open_fee_bps is None:
            return None
        
        # 3. Calculate using shared logic (uses oracle spread by default)
        result = ExecutionCalculator.calculate_execution_cost(
            std_orderbook,
            order_size_usd,
            open_fee_bps=open_fee_bps,
            close_fee_bps=close_fee_bps
        )
        
        if result:
            # Add Ostium-specific metadata
            result['fee_bps'] = open_fee_bps
            result['maker_fee_bps'] = maker_fee_bps if maker_fee_bps is not None else 0.0
            if raw_data:
                result['is_market_open'] = raw_data.get('isMarketOpen', False)
            
            # Inject Max Leverage
            result['max_leverage'] = self.get_max_leverage(asset)
            
            # Use oracle bid/ask spread from metadata for slippage calculation
            # (priceImpactK method removed - numbers weren't accurate)
            # result['price_impact_k_used'] = False

            return result

class HyperliquidAPI:
    def __init__(self):
        self.base_url = "https://api.hyperliquid.xyz/info"
        self.headers = {'Content-Type': 'application/json'}
        self.max_leverages_cache = {}
        self.last_metadata_fetch = 0
        self.metadata_cache_ttl = 300  # 5 minutes

    def _fetch_metadata(self):
        """Fetch metadata to get max leverage info."""
        if time.time() - self.last_metadata_fetch < self.metadata_cache_ttl and self.max_leverages_cache:
            return

        try:
            # Use dex='xyz' as discovered
            payload = {"type": "metaAndAssetCtxs", "dex": "xyz"}
            response = requests.post(self.base_url, json=payload, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                universe = []
                if isinstance(data, list) and len(data) >= 1:
                    universe = data[0].get("universe", [])
                elif isinstance(data, dict):
                    universe = data.get("universe", [])
                
                # Update cache
                self.max_leverages_cache = {}
                for item in universe:
                    name = item.get("name")
                    max_lev = item.get("maxLeverage")
                    if name:
                        self.max_leverages_cache[name] = max_lev
                        # Also handle xyz: prefix stripping if needed or just cache what we have
                        # The symbols we use usually come as "xyz:GOLD" or just "GOLD"? 
                        # In the script we saw "xyz:GOLD"
                        
                        # Store both variants to be safe
                        if name.startswith("xyz:"):
                            self.max_leverages_cache[name.replace("xyz:", "")] = max_lev
                        else:
                            self.max_leverages_cache[f"xyz:{name}"] = max_lev
                
                self.last_metadata_fetch = time.time()
        except Exception as e:
            print(f"Error fetching HL metadata: {e}")

    def get_max_leverage(self, symbol: str) -> Optional[float]:
        self._fetch_metadata()
        
        # Try xyz:
        if f"xyz:{symbol}" in self.max_leverages_cache:
            return self.max_leverages_cache[f"xyz:{symbol}"]
            
        return None

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
            
            # Inject Max Leverage
            final_result['max_leverage'] = self.get_max_leverage(symbol)
            
        return final_result


class LighterAPI:
    # Hardcoded max leverage from Lighter UI (by market_id)
    MAX_LEVERAGE = {
        92: 20,   # XAU
        93: 20,   # XAG
        128: 20,  # SPY
        129: 20,  # QQQ
        109: 10,  # COIN
        110: 10,  # NVDA
        112: 10,  # TSLA
        113: 10,  # AAPL
        114: 10,  # AMZN
        115: 10,  # MSFT
        116: 10,  # GOOG
        117: 10,  # META
        96: 25,   # EURUSD
        97: 25,   # GBPUSD
        98: 25,   # USDJPY
    }
    
    def __init__(self):
        self.base_url = "https://mainnet.zklighter.elliot.ai/api/v1"
        self.headers = {'Content-Type': 'application/json'}
        self.fee_cache = {}  # market_id -> {taker_fee, maker_fee}
        self.fee_cache_loaded = False
    
    def _load_fee_cache(self):
        """Load fees from orderBooks API for all perp markets."""
        if self.fee_cache_loaded:
            return
        
        try:
            url = f"{self.base_url}/orderBooks?filter=perp"
            response = requests.get(url, headers=self.headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                markets = data.get('order_books', [])
                for m in markets:
                    market_id = m.get('market_id')
                    if market_id is not None:
                        # Fees are in percentage format (e.g., "0.0000" = 0%)
                        taker = float(m.get('taker_fee', '0')) * 100  # Convert to bps
                        maker = float(m.get('maker_fee', '0')) * 100  # Convert to bps
                        self.fee_cache[market_id] = {
                            'taker_fee_bps': taker,
                            'maker_fee_bps': maker
                        }
                self.fee_cache_loaded = True
        except Exception as e:
            print(f"Error loading Lighter fee cache: {e}")
    
    def get_fees(self, market_id: int) -> tuple:
        """Get taker and maker fees for a market_id."""
        self._load_fee_cache()
        fees = self.fee_cache.get(market_id, {})
        return (
            fees.get('taker_fee_bps', 0.0),
            fees.get('maker_fee_bps', 0.0)
        )
    
    def get_max_leverage(self, market_id: int) -> Optional[int]:
        """Get max leverage for a market_id from hardcoded lookup."""
        return self.MAX_LEVERAGE.get(market_id)

    def get_orderbook(self, market_id: int) -> Optional[Dict]:
        url = f"{self.base_url}/orderBookOrders?market_id={market_id}&limit=250"
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

    def calculate_execution_cost(self, orderbook: Dict, order_size_usd: float, market_id: int = None) -> Optional[Dict]:
        """Calculate execution cost using shared ExecutionCalculator."""
        std_orderbook = self.normalize_orderbook(orderbook)
        if not std_orderbook:
            return None
        
        # Get dynamic fees from API
        taker_fee_bps, maker_fee_bps = self.get_fees(market_id) if market_id else (0.0, 0.0)
        
        result = ExecutionCalculator.calculate_execution_cost(
            std_orderbook,
            order_size_usd,
            open_fee_bps=taker_fee_bps,
            close_fee_bps=0.0
        )
        
        if result:
            result['fee_bps'] = taker_fee_bps
            result['maker_fee_bps'] = maker_fee_bps
            if market_id:
                result['max_leverage'] = self.get_max_leverage(market_id)
        
        return result

class AsterAPI:
    BASE_URL = "https://fapi.asterdex.com/fapi/v1"
    
    # Hardcoded max leverage from Aster UI (API data is inaccurate)
    MAX_LEVERAGE = {
        'XAUUSDT': 75,
        'XAGUSDT': 100,
        'QQQUSDT': 10,  
        'NVDAUSDT': 10,
        'AAPLUSDT': 10,
        'AMZNUSDT': 10,
        'GOOGUSDT': 10,
        'MSFTUSDT': 10,
        'METAUSDT': 10,
        'TSLAUSDT': 10,
        'COINUSDT': None, #not available
    }
    
    def __init__(self):
        self.headers = {'Content-Type': 'application/json'}
    
    def get_max_leverage(self, symbol: str) -> Optional[int]:
        """Get max leverage for a symbol from hardcoded lookup."""
        return self.MAX_LEVERAGE.get(symbol)

    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        url = f"{self.BASE_URL}/depth"
        params = {'symbol': symbol, 'limit': 1000}  
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=1000)
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

    def calculate_execution_cost(self, orderbook: Dict, order_size_usd: float, symbol: str = None) -> Optional[Dict]:
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
            if symbol:
                result['max_leverage'] = self.get_max_leverage(symbol)
        
        return result


class AvantisAPI:
    """
    Dynamic fee and spread calculation for Avantis.
    Fetches real-time data from Avantis socket API and risk API.
    """
    
    # API Configuration
    SOCKET_API = "https://socket-api-pub.avantisfi.com/socket-api/v1/data"
    RISK_API = "https://risk-api.avantisfi.com/spread/dynamic"
    DUMMY_TRADER = "0x1234567890123456789012345678901234567890"
    
    # Pair indices
    PAIRS = {
        "XAU": 21, "XAG": 20,
        "EURUSD": 11, "GBPUSD": 13, "USDJPY": 12,
        "SPY": 78, "QQQ": 79,
        "NVDA": 81, "TSLA": 86, "AAPL": 82, "MSFT": 84,
        "AMZN": 83, "GOOG": 87, "META": 85, "COIN": 80, "HOOD": 91,
    }
    
    def __init__(self):
        self._pair_data = None
        self._group_info = None
        self._last_fetch = 0
        self._cache_ttl = 30  # Cache for 30 seconds
    
    def _fetch_socket_data(self):
        """Fetch and cache pair data from Avantis socket API."""
        now = time.time()
        if self._pair_data and (now - self._last_fetch) < self._cache_ttl:
            return
        
        try:
            resp = requests.get(self.SOCKET_API, timeout=30)
            data = resp.json().get("data", {})
            self._pair_data = data.get("pairInfos", {})
            self._group_info = data.get("groupInfo", {})
            self._last_fetch = now
        except Exception as e:
            print(f"Avantis API error: {e}")
            self._pair_data = {}
            self._group_info = {}
    
    def _get_pair_info(self, asset_key: str) -> Optional[Tuple[str, Dict]]:
        """Get pair info by asset symbol."""
        self._fetch_socket_data()
        
        key = asset_key.upper()
        pair_idx = self.PAIRS.get(key)
        
        if pair_idx is None:
            return None
        
        pair_info = self._pair_data.get(str(pair_idx))
        if not pair_info:
            return None
        
        return str(pair_idx), pair_info
    
    def _calculate_opening_fee(self, pair_info: Dict, position_size: float, is_long: bool = True) -> float:
        """
        Calculate opening fee dynamically based on skewEqParams and OI.
        Returns fee in basis points.
        """
        import math
        from decimal import Decimal
        
        long_oi = pair_info.get("openInterest", {}).get("long", 0)
        short_oi = pair_info.get("openInterest", {}).get("short", 0)
        skew_params = pair_info.get("skewEqParams", [[0, 450]])
        
        # Calculate OI percentage
        if is_long:
            new_long_oi = long_oi + position_size
            divisor = new_long_oi + short_oi
            open_interest_pct = math.floor((100 * short_oi) / (divisor if divisor != 0 else 1))
        else:
            new_short_oi = short_oi + position_size
            divisor = new_short_oi + long_oi
            open_interest_pct = math.floor((100 * long_oi) / (divisor if divisor != 0 else 1))
        
        # Get pctIndex
        pct_index = min(
            int(Decimal(str(open_interest_pct)) / Decimal('10')),
            len(skew_params) - 1
        )
        
        # Get params
        param1 = skew_params[pct_index][0]
        param2 = skew_params[pct_index][1]
        
        # Calculate fee: (param1 * openInterestPct + param2) / 10000 * 100 (to get bps)
        skew_adjusted = (param1 * open_interest_pct + param2) / 10000
        fee_bps = skew_adjusted * 100
        
        
        return fee_bps
    
    def _fetch_dynamic_spread(self, pair_index: int, position_size: float, is_long: bool, is_pnl: bool) -> Optional[float]:
        """Fetch dynamic spread from risk API. Returns spread in bps."""
        params = {
            "pairIndex": pair_index,
            "positionSizeUsdc": int(position_size * (10 ** 18)),
            "isLong": str(is_long).lower(),
            "isPnl": str(is_pnl).lower(),
            "trader": self.DUMMY_TRADER
        }
        try:
            resp = requests.get(self.RISK_API, params=params, timeout=30)
            data = resp.json()
            spread_raw = float(data.get("spreadP", 0))
            # Convert from 10^10 scaled to percentage, then to bps
            spread_pct = spread_raw / (10 ** 10)
            return spread_pct * 100  # Convert to bps
        except Exception:
            return None
    
    def _get_spread(self, pair_idx: str, pair_info: Dict, position_size: float, is_long: bool = True) -> float:
        """Get spread based on whether it's dynamic or constant."""
        group_index = pair_info.get("groupIndex", 0)
        group = self._group_info.get(str(group_index), {})
        is_spread_dynamic = group.get("isSpreadDynamic", False)
        
        # Always use isPnl=False for slippage/spread measurement
        # (isPnl=True would be for max leverage calculations only)
        is_pnl = False
        
        if is_spread_dynamic:
            # Fetch dynamic spread from API
            spread = self._fetch_dynamic_spread(int(pair_idx), position_size, is_long, is_pnl)
            if spread is not None:
                return spread
        
        # Use constant spread from pair info
        spread_pct = pair_info.get("spreadP", 0)
        return spread_pct * 100  # Convert percentage to bps
    
    def _get_close_fee(self, pair_info: Dict) -> float:
        """Get closing fee from pair info."""
        close_fee_pct = pair_info.get("closeFeeP", 0)
        return close_fee_pct  # Already in bps-like units, but check
    
    def calculate_cost(self, asset_key: str, order_size_usd: float, is_long: bool = True) -> Dict:
        """
        Calculate execution cost for Avantis dynamically.
        Uses real-time data from Avantis APIs.
        """
        result = self._get_pair_info(asset_key)
        
        if not result:
            # Asset not found in Avantis
            return None
        
        pair_idx, pair_info = result
        
        # Position size for fee calculation (Assume 1x leverage for comparison)
        position_size = order_size_usd
        
        # Calculate opening fee dynamically
        open_fee_bps = self._calculate_opening_fee(pair_info, position_size, is_long=is_long)
        
        # Get closing fee (from API, converted from percentage)
        close_fee_pct = pair_info.get("closeFeeP", 0)
        close_fee_bps = close_fee_pct  # closeFeeP is already stored as bps-equivalent
        
        # Get spread (opening slippage)
        spread_bps = self._get_spread(pair_idx, pair_info, position_size, is_long=is_long)
        
        # Avantis: Spread only occurs on opening, closing has 0 slippage
        opening_slippage_bps = spread_bps
        closing_slippage_bps = 0.0
        total_slippage_bps = opening_slippage_bps + closing_slippage_bps
        total_cost_bps = total_slippage_bps + open_fee_bps + close_fee_bps
        
        # Get max leverage
        leverages = pair_info.get("leverages", {})
        max_lev = leverages.get("pnlMaxLeverage")
        if max_lev is None:
            max_lev = leverages.get("maxLeverage")
        
        return {
            'max_leverage': max_lev,
            'executed': True,
            'mid_price': 0,
            'slippage_bps': total_slippage_bps,
            'opening_slippage_bps': opening_slippage_bps,
            'closing_slippage_bps': closing_slippage_bps,
            'buy_slippage_bps': opening_slippage_bps,
            'sell_slippage_bps': closing_slippage_bps,
            'slippage_type': 'opening_closing',
            'open_fee_bps': open_fee_bps,
            'close_fee_bps': close_fee_bps,
            'maker_fee_bps': 0.0,
            'total_cost_bps': total_cost_bps,
            'filled': True,
            'buy': {'filled': True, 'filled_usd': order_size_usd, 'unfilled_usd': 0, 'levels_used': 1, 'slippage_bps': opening_slippage_bps},
            'sell': {'filled': True, 'filled_usd': order_size_usd, 'unfilled_usd': 0, 'levels_used': 1, 'slippage_bps': closing_slippage_bps},
            'timestamp': time.time()
        }



class ExtendedAPI:
    """Client for Extended Exchange (Starknet) orderbook data."""
    
    BASE_URL = "https://api.starknet.extended.exchange/api/v1"
    API_KEY = "8dee2f0750d7f796699ec461b7c64219"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-API-Key": self.API_KEY,
        })
        self.market_cache = {}  # market -> {max_leverage, ...}
        self.market_cache_loaded = {}
        self.fee_cache = {}  # market -> {taker_fee_bps, maker_fee_bps}
    
    def get_fees(self, market: str) -> Tuple[Optional[float], Optional[float]]:
        """Get taker and maker fees for a market from /api/v1/user/fees?market={market}."""
        if market in self.fee_cache:
            c = self.fee_cache[market]
            return (c.get('taker_fee_bps'), c.get('maker_fee_bps'))
        try:
            url = f"{self.BASE_URL}/user/fees?market={market}"
            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                return (None, None)
            data = response.json()
            
            # Handle list response in data.data
            raw_data = data.get('data', data)
            if isinstance(raw_data, list) and raw_data:
                raw = raw_data[0]
            elif isinstance(raw_data, dict):
                raw = raw_data
            else:
                raw = {}

            # Support common response shapes including *FeeRate
            taker = raw.get('takerFeeRate', raw.get('takerFee', raw.get('taker_fee')))
            maker = raw.get('makerFeeRate', raw.get('makerFee', raw.get('maker_fee')))
            
            # If in decimal form (e.g. 0.00025) -> bps = * 10000; if in percent (e.g. 0.025) -> bps = * 100
            if taker is not None:
                taker_val = float(taker)
                taker_bps = taker_val * 10000 if abs(taker_val) < 0.1 else taker_val * 100
            else:
                taker_bps = 0.0 # Default if field missing but call succeeded? Or None? keep 0.0 for logic safety if partial

            if maker is not None:
                maker_val = float(maker)
                maker_bps = maker_val * 10000 if abs(maker_val) < 0.1 else maker_val * 100
            else:
                maker_bps = 0.0

            # If the API call succeeded but weird data, we might have 0.0. 
            # But earlier fallback was (0,0). Detailed failure return (None, None) is preferred if explicit error.
            # If explicit None from parsing, let's trust it. 
            # Actually, let's conform to "if not retrieved just type null".
            
            self.fee_cache[market] = {'taker_fee_bps': taker_bps, 'maker_fee_bps': maker_bps}
            return (taker_bps, maker_bps)
        except Exception as e:
            print(f"Error fetching Extended fees for {market}: {e}")
            return (None, None)
    
    def _load_market_info(self, market: str):
        """Fetch and cache market info including max leverage."""
        if self.market_cache_loaded.get(market):
            return
        
        try:
            url = f"{self.BASE_URL}/info/markets?market={market}"
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'OK':
                    markets = data.get('data', [])
                    if markets:
                        market_data = markets[0]
                        trading_config = market_data.get('tradingConfig', {})
                        self.market_cache[market] = {
                            'max_leverage': float(trading_config.get('maxLeverage', 0))
                        }
                        self.market_cache_loaded[market] = True
        except Exception as e:
            print(f"Error fetching Extended market info for {market}: {e}")
    
    def get_max_leverage(self, market: str) -> Optional[float]:
        """Get max leverage for a market."""
        self._load_market_info(market)
        cache = self.market_cache.get(market, {})
        return cache.get('max_leverage')
    
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

    def calculate_execution_cost(self, orderbook: Dict, order_size_usd: float, market: str = None) -> Optional[Dict]:
        """Calculate execution cost using shared ExecutionCalculator."""
        std_orderbook = self.normalize_orderbook(orderbook)
        if not std_orderbook:
            return None
        
        taker_bps, maker_bps = self.get_fees(market) if market else (None, None)
        
        # Use 0.0 for calculation if fees are None, but result will reflect missing fees
        calc_open = taker_bps if taker_bps is not None else 0.0
        calc_close = taker_bps if taker_bps is not None else 0.0
        
        result = ExecutionCalculator.calculate_execution_cost(
            std_orderbook,
            order_size_usd,
            open_fee_bps=calc_open,
            close_fee_bps=calc_close
        )
        
        if result:
            result['maker_fee_bps'] = maker_bps
            # Add max leverage if market provided
            if market:
                result['max_leverage'] = self.get_max_leverage(market)

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
    max_leverage: Optional[float] = None


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
        self.avantis = AvantisAPI()
        self.ostium = OstiumAPI()
        self.extended = ExtendedAPI()

    def compare_asset(self, asset_key: str, order_size_usd: float, order_type: str = 'taker', direction: str = 'long') -> Optional[Dict]:
        """
        Compare execution cost across all exchanges for a given asset.
        
        Args:
            asset_key: Asset symbol (e.g. 'BTC', 'ETH')
            order_size_usd: Order size in USD
            order_type: 'taker' or 'maker'
            direction: 'long' or 'short'
        """
        if asset_key not in ASSETS:
            return None
            
        config = ASSETS[asset_key]
        result = {
            'asset': config.name,
            'symbol_key': config.symbol_key,
            'order_size_usd': order_size_usd,
            'order_type': order_type,
            'direction': direction,
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
            aster_result = self.aster.calculate_execution_cost(aster_orderbook, order_size_usd, symbol=config.aster_symbol)
            if aster_result:
                aster_result['symbol'] = config.aster_symbol
            result['aster'] = aster_result

        # --- Avantis (Static) ---
        # --- Avantis (Static) ---
        is_long = (direction.lower() == 'long')
        avantis_result = self.avantis.calculate_cost(asset_key, order_size_usd, is_long=is_long)
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
            extended_result = self.extended.calculate_execution_cost(extended_orderbook, order_size_usd, market=config.extended_symbol)
            if extended_result:
                extended_result['symbol'] = config.extended_symbol
            result['extended'] = extended_result

        # Override for Maker orders (Zero Slippage) - only for orderbook-based perp DEXes
        # Avantis and Ostium keep their slippage as they are oracle-based
        if order_type == 'maker':
            for ex in ['hyperliquid', 'lighter', 'aster', 'extended']:
                if result.get(ex):
                    result[ex]['slippage_bps'] = 0.0
                    result[ex]['buy_slippage_bps'] = 0.0
                    result[ex]['sell_slippage_bps'] = 0.0
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
    
    order_size = float(data.get('order_size', 1000000))
    
    order_type = data.get('order_type', 'taker').lower()
    direction = data.get('direction', 'long').lower()
    
    if asset not in ASSETS:
        return jsonify({'error': f'Asset {asset} not found'}), 400
    
    result = comparator.compare_asset(asset, order_size, order_type=order_type, direction=direction)
    
    if not result:
        return jsonify({'error': 'Failed to compare asset'}), 500
    
    # Calculate totals and determine winner
    exchanges = []
    
    # Get Lighter and Extended fees dynamically from API
    config = ASSETS[asset]
    lighter_taker_bps, lighter_maker_bps = comparator.lighter.get_fees(config.lighter_market_id) if config.lighter_market_id else (0.0, 0.0)
    extended_taker_bps, extended_maker_bps = comparator.extended.get_fees(config.extended_symbol) if config.extended_symbol else (0.0, 0.0)
    
    # Fee structures
    if order_type == 'maker':
        # MAKER Mode: Use Maker Fees for Orderbook, Taker/Flat for Oracle
        fee_structure = {
            'hyperliquid': {'open': HYPERLIQUID_MAKER_FEE_BPS, 'close': HYPERLIQUID_MAKER_FEE_BPS},
            'lighter': {'open': lighter_maker_bps, 'close': lighter_maker_bps},
            'aster': {'open': ASTER_MAKER_FEE_BPS, 'close': ASTER_MAKER_FEE_BPS},
            'extended': {'open': extended_maker_bps, 'close': extended_maker_bps}
        }
    else:
        # TAKER Mode: Standard Taker Fees
        fee_structure = {
            'hyperliquid': {'open': HYPERLIQUID_TAKER_FEE_BPS, 'close': HYPERLIQUID_TAKER_FEE_BPS},
            'lighter': {'open': lighter_taker_bps, 'close': 0.0},
            'aster': {'open': ASTER_TAKER_FEE_BPS, 'close': 0.0},
            'extended': {'open': extended_taker_bps, 'close': extended_taker_bps}
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
    
    # Standardize Slippage Type for all exchanges
    is_long_direction = (direction == 'long')
    
    for exchange_name in ['hyperliquid', 'lighter', 'aster', 'ostium', 'extended']:
        ex_data = result.get(exchange_name)
        if ex_data:
            # Get raw buy/sell slippage
            buy_slip = ex_data.get('buy_slippage_bps', 0.0)
            sell_slip = ex_data.get('sell_slippage_bps', 0.0)
            
            # Map to Opening/Closing based on direction
            if is_long_direction:
                # LONG: Opening = Buy, Closing = Sell
                ex_data['opening_slippage_bps'] = buy_slip
                ex_data['closing_slippage_bps'] = sell_slip
            else:
                # SHORT: Opening = Sell, Closing = Buy
                ex_data['opening_slippage_bps'] = sell_slip
                ex_data['closing_slippage_bps'] = buy_slip
            
            ex_data['slippage_type'] = 'opening_closing'

    for exchange_name in ['hyperliquid', 'lighter', 'aster', 'avantis', 'ostium', 'extended']:
        ex_data = result.get(exchange_name)
        if ex_data:
            fees = fee_structure.get(exchange_name, {'open': 0, 'close': 0})
            slippage = ex_data.get('slippage_bps', 0)
            
            # Avantis: slippage only occurs once 
            effective_spread = slippage if exchange_name == 'avantis' else 2 * slippage
            
            f_open = fees['open']
            f_close = fees['close']
            
            if f_open is None or f_close is None:
                total_cost = None
            else:
                total_cost = effective_spread + f_open + f_close
            
            ex_data['effective_spread_bps'] = effective_spread
            ex_data['open_fee_bps'] = f_open
            ex_data['close_fee_bps'] = f_close
            ex_data['total_cost_bps'] = total_cost
            ex_data['exchange'] = exchange_name
            
            if total_cost is not None and ex_data.get('executed') != 'PARTIAL':
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
    direction = request.args.get('direction', 'long').lower()
    
    if asset not in ASSETS:
        return jsonify({'error': f'Asset {asset} not found', 'available_assets': list(ASSETS.keys())}), 400
    
    result = comparator.compare_asset(asset, order_size, order_type=order_type, direction=direction)
    
    if not result:
        return jsonify({'error': 'Failed to compare asset'}), 500
    
    # Calculate totals and determine winner (same logic as POST endpoint)
    exchanges = []
    
    # Get Lighter and Extended fees dynamically from API
    config = ASSETS[asset]
    lighter_taker_bps, lighter_maker_bps = comparator.lighter.get_fees(config.lighter_market_id) if config.lighter_market_id else (0.0, 0.0)
    extended_taker_bps, extended_maker_bps = comparator.extended.get_fees(config.extended_symbol) if config.extended_symbol else (0.0, 0.0)
    
    if order_type == 'maker':
        fee_structure = {
            'hyperliquid': {'open': HYPERLIQUID_MAKER_FEE_BPS, 'close': HYPERLIQUID_MAKER_FEE_BPS},
            'lighter': {'open': lighter_maker_bps, 'close': lighter_maker_bps},
            'aster': {'open': ASTER_MAKER_FEE_BPS, 'close': ASTER_MAKER_FEE_BPS},
            'extended': {'open': extended_maker_bps, 'close': extended_maker_bps}
        }
    else:
        fee_structure = {
            'hyperliquid': {'open': HYPERLIQUID_TAKER_FEE_BPS, 'close': HYPERLIQUID_TAKER_FEE_BPS},
            'lighter': {'open': lighter_taker_bps, 'close': 0.0},
            'aster': {'open': ASTER_TAKER_FEE_BPS, 'close': 0.0},
            'extended': {'open': extended_taker_bps, 'close': extended_taker_bps}
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
            
            f_open = fees['open']
            f_close = fees['close']
            
            if f_open is None or f_close is None:
                total_cost = None
            else:
                total_cost = effective_spread + f_open + f_close
            
            ex_data['effective_spread_bps'] = effective_spread
            ex_data['open_fee_bps'] = f_open
            ex_data['close_fee_bps'] = f_close
            ex_data['total_cost_bps'] = total_cost
            ex_data['exchange'] = exchange_name
            
            if total_cost is not None and ex_data.get('executed') != 'PARTIAL':
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
        direction = data.get('direction', 'long').lower()
        
        if not asset or asset not in ASSETS:
            emit('compare_error', {'error': f'Unknown asset: {asset}'})
            return
        
        config = ASSETS[asset]
        result = {
            'asset': asset,
            'name': config.name,
            'order_size_usd': order_size,
            'order_type': order_type,
            'direction': direction
        }
        
        # Fetch data from all exchanges (reusing existing logic)
        hl_api = HyperliquidAPI()
        lighter_api = LighterAPI()
        aster_api = AsterAPI()
        avantis_static = AvantisAPI()
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
                lighter_result = lighter_api.calculate_execution_cost(lighter_book, order_size, market_id=config.lighter_market_id)
                if lighter_result:
                    result['lighter'] = lighter_result
        
        # Aster
        if config.aster_symbol:
            aster_book = aster_api.get_orderbook(config.aster_symbol)
            if aster_book:
                aster_result = aster_api.calculate_execution_cost(aster_book, order_size, symbol=config.aster_symbol)
                if aster_result:
                    result['aster'] = aster_result
        
        # Avantis (static fees)
        is_long = (direction.lower() == 'long')
        avantis_result = avantis_static.calculate_cost(asset, order_size, is_long=is_long)
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
                extended_result = extended_api.calculate_execution_cost(extended_book, order_size, market=config.extended_symbol)
                if extended_result:
                    result['extended'] = extended_result
        
        # Override for Maker orders (Zero Slippage) - only for orderbook-based perp DEXes
        # Avantis and Ostium keep their slippage as they are oracle-based
        if order_type == 'maker':
            for ex in ['hyperliquid', 'lighter', 'aster', 'extended']:
                if result.get(ex):
                    result[ex]['slippage_bps'] = 0.0
                    result[ex]['buy_slippage_bps'] = 0.0
                    result[ex]['sell_slippage_bps'] = 0.0
                    if 'buy' in result[ex]: result[ex]['buy']['slippage_bps'] = 0.0
                    if 'sell' in result[ex]: result[ex]['sell']['slippage_bps'] = 0.0
        
        # Calculate total costs and determine winner (same logic as HTTP endpoint)
        exchanges = []
        
        # Get Lighter and Extended fees dynamically from API
        lighter_taker_bps, lighter_maker_bps = lighter_api.get_fees(config.lighter_market_id) if config.lighter_market_id else (0.0, 0.0)
        extended_taker_bps, extended_maker_bps = extended_api.get_fees(config.extended_symbol) if config.extended_symbol else (0.0, 0.0)
        
        if order_type == 'maker':
            fee_structure = {
                'hyperliquid': {'open': HYPERLIQUID_MAKER_FEE_BPS, 'close': HYPERLIQUID_MAKER_FEE_BPS},
                'lighter': {'open': lighter_maker_bps, 'close': lighter_maker_bps},
                'aster': {'open': ASTER_MAKER_FEE_BPS, 'close': ASTER_MAKER_FEE_BPS},
                'extended': {'open': extended_maker_bps, 'close': extended_maker_bps}
            }
        else:
            fee_structure = {
                'hyperliquid': {'open': HYPERLIQUID_TAKER_FEE_BPS, 'close': HYPERLIQUID_TAKER_FEE_BPS},
                'lighter': {'open': lighter_taker_bps, 'close': 0.0},
                'aster': {'open': ASTER_TAKER_FEE_BPS, 'close': 0.0},
                'extended': {'open': extended_taker_bps, 'close': extended_taker_bps}
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
                
                f_open = fees['open']
                f_close = fees['close']
                
                if f_open is None or f_close is None:
                    total_cost = None
                else:
                    total_cost = effective_spread + f_open + f_close
                
                ex_data['effective_spread_bps'] = effective_spread
                ex_data['open_fee_bps'] = f_open
                ex_data['close_fee_bps'] = f_close
                ex_data['total_cost_bps'] = total_cost
                ex_data['exchange'] = exchange_name
                
                if total_cost is not None and ex_data.get('executed') != 'PARTIAL':
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
