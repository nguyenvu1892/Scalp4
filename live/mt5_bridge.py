"""
ScalForex — MT5 Bridge
Kết nối Demo Exness qua MetaTrader5. Market orders with hard SL/TP.
Exponential backoff retry on connection failures.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from utils.logger import setup_logger

log = setup_logger("live.mt5_bridge")


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OrderResult:
    """Result of an order execution."""
    success: bool
    ticket: int = 0
    price: float = 0.0
    lot: float = 0.0
    side: str = ""
    symbol: str = ""
    sl: float = 0.0
    tp: float = 0.0
    error_message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PositionInfo:
    """Active position information."""
    ticket: int
    symbol: str
    side: str
    lot: float
    open_price: float
    current_price: float
    profit: float
    sl: float
    tp: float


class MT5Bridge:
    """MetaTrader5 bridge for Demo Exness trading.

    Safety features:
        - Hard SL/TP set at order time (server-side protection)
        - Exponential backoff on connection failure
        - Max retry limit
        - All operations logged

    Args:
        login: MT5 account login.
        password: MT5 account password.
        server: Broker server name.
        mt5_path: Path to MT5 terminal executable.
        max_retries: Max connection retries.
        base_delay: Base delay for exponential backoff (seconds).
    """

    def __init__(
        self,
        login: int = 0,
        password: str = "",
        server: str = "Exness-MT5Real",
        mt5_path: Optional[str] = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
    ) -> None:
        self.login = login or int(os.getenv("MT5_LOGIN", "0"))
        self.password = password or os.getenv("MT5_PASSWORD", "")
        self.server = server or os.getenv("MT5_SERVER", "Exness-MT5Real")
        self.mt5_path = mt5_path or os.getenv("MT5_PATH", "")
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._connected = False
        self._mt5 = None  # MetaTrader5 module reference

    def connect(self) -> bool:
        """Connect to MT5 with exponential backoff retry.

        SAFETY: mt5_path is MANDATORY. Without it, mt5.initialize()
        could connect to a LIVE terminal running in parallel.

        Returns:
            True if connected successfully.
        """
        # ═══════════════════════════════════════════════════════════
        # ⚠️ CRITICAL SAFETY CHECK — DO NOT REMOVE ⚠️
        # Máy chạy song song MT5 Live + Demo.
        # Nếu gọi mt5.initialize() KHÔNG CÓ path, nó sẽ bắt vào
        # Terminal ĐẦU TIÊN tìm thấy — có thể là Live → cháy TK!
        # ═══════════════════════════════════════════════════════════
        if not self.mt5_path:
            log.critical(
                "🚨🚨🚨 MT5_PATH IS EMPTY — REFUSING TO CONNECT! 🚨🚨🚨\n"
                "  Máy đang chạy song song Live + Demo MT5.\n"
                "  Nếu gọi mt5.initialize() không có path, có thể\n"
                "  BẮT VÀO TERMINAL LIVE → TRADE NHẦM TIỀN THẬT!\n"
                "  Set mt5_path trong configs/trading_rules.yaml hoặc env MT5_PATH."
            )
            return False

        if not self.login:
            log.critical(
                "🚨 MT5_LOGIN is empty — refusing to connect. "
                "Set MT5_LOGIN env var or pass login parameter."
            )
            return False

        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
        except ImportError:
            log.error(
                "MetaTrader5 package not installed. "
                "Install with: pip install MetaTrader5"
            )
            return False

        for attempt in range(1, self.max_retries + 1):
            log.info(f"MT5 connect attempt {attempt}/{self.max_retries}...")
            log.info(f"  path={self.mt5_path}")
            log.info(f"  login={self.login}, server={self.server}")

            # ALWAYS pass path + login + server — NEVER leave empty
            if not self._mt5.initialize(
                path=self.mt5_path,
                login=self.login,
                password=self.password,
                server=self.server,
            ):
                error = self._mt5.last_error()
                log.warning(f"MT5 connect failed: {error}")

                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** (attempt - 1))
                    log.info(f"Retrying in {delay:.1f}s (exponential backoff)...")
                    time.sleep(delay)
                continue

            account_info = self._mt5.account_info()
            if account_info is not None:
                self._connected = True
                log.info(
                    f"MT5 connected: Account #{account_info.login} "
                    f"({account_info.server}) "
                    f"Balance: ${account_info.balance:.2f}"
                )
                return True

        log.error(f"MT5 connection failed after {self.max_retries} attempts")
        return False

    def disconnect(self) -> None:
        """Disconnect from MT5."""
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False
            log.info("MT5 disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._mt5 is not None

    def get_account_info(self) -> dict:
        """Get current account information.

        Returns:
            Dict with balance, equity, margin, profit, etc.
        """
        if not self.is_connected:
            return {"error": "Not connected"}

        info = self._mt5.account_info()
        if info is None:
            return {"error": "Failed to get account info"}

        return {
            "login": info.login,
            "server": info.server,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "profit": info.profit,
            "leverage": info.leverage,
        }

    def get_symbol_price(self, symbol: str) -> Optional[dict]:
        """Get current bid/ask prices.

        Args:
            symbol: Trading symbol (e.g. XAUUSDm).

        Returns:
            Dict with bid, ask, spread, or None on failure.
        """
        if not self.is_connected:
            return None

        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            log.warning(f"Failed to get tick for {symbol}")
            return None

        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": round(tick.ask - tick.bid, 5),
            "time": datetime.fromtimestamp(tick.time),
        }

    def send_market_order(
        self,
        symbol: str,
        side: OrderSide,
        lot: float,
        sl_points: float = 0,
        tp_points: float = 0,
        comment: str = "ScalForex AI",
    ) -> OrderResult:
        """Send market order with hard SL/TP on broker server.

        SL/TP are set AT ORDER TIME — server-side protection.

        Args:
            symbol: Trading symbol.
            side: BUY or SELL.
            lot: Position size.
            sl_points: Stop loss in points from entry.
            tp_points: Take profit in points from entry.
            comment: Order comment.

        Returns:
            OrderResult with execution details.
        """
        if not self.is_connected:
            return OrderResult(success=False, error_message="Not connected")

        mt5 = self._mt5
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(success=False, error_message=f"No price for {symbol}")

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return OrderResult(success=False, error_message=f"Unknown symbol {symbol}")

        point = symbol_info.point

        if side == OrderSide.BUY:
            price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY
            sl = price - sl_points * point if sl_points > 0 else 0.0
            tp = price + tp_points * point if tp_points > 0 else 0.0
        else:
            price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
            sl = price + sl_points * point if sl_points > 0 else 0.0
            tp = price - tp_points * point if tp_points > 0 else 0.0

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,  # Max slippage
            "magic": 314159,  # ScalForex magic number
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        log.info(
            f"Sending {side.value} {lot} {symbol} @ {price:.5f} "
            f"SL={sl:.5f} TP={tp:.5f}"
        )

        result = mt5.order_send(request)
        if result is None:
            error = mt5.last_error()
            log.error(f"Order send returned None: {error}")
            return OrderResult(success=False, error_message=str(error))

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(
                f"Order failed: retcode={result.retcode} "
                f"comment={result.comment}"
            )
            return OrderResult(
                success=False,
                error_message=f"retcode={result.retcode}: {result.comment}",
            )

        log.info(
            f"✅ Order executed: Ticket #{result.order} "
            f"{side.value} {lot} {symbol} @ {result.price:.5f}"
        )
        return OrderResult(
            success=True,
            ticket=result.order,
            price=result.price,
            lot=lot,
            side=side.value,
            symbol=symbol,
            sl=sl,
            tp=tp,
        )

    def close_position(self, ticket: int) -> OrderResult:
        """Close a specific position by ticket.

        Args:
            ticket: Position ticket to close.

        Returns:
            OrderResult with close details.
        """
        if not self.is_connected:
            return OrderResult(success=False, error_message="Not connected")

        mt5 = self._mt5
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return OrderResult(success=False, error_message=f"Position {ticket} not found")

        pos = position[0]
        symbol = pos.symbol
        lot = pos.volume

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(success=False, error_message=f"No price for {symbol}")

        if pos.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 314159,
            "comment": "ScalForex AI Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"✅ Closed position #{ticket} @ {result.price:.5f}")
            return OrderResult(
                success=True, ticket=ticket, price=result.price,
                lot=lot, side="CLOSE", symbol=symbol,
            )

        error = result.comment if result else str(mt5.last_error())
        log.error(f"Failed to close #{ticket}: {error}")
        return OrderResult(success=False, error_message=error)

    def close_all_positions(self) -> list[OrderResult]:
        """EMERGENCY: Close ALL open positions immediately.

        Used by Killswitch when DD exceeds threshold.

        Returns:
            List of OrderResults for each closed position.
        """
        if not self.is_connected:
            return [OrderResult(success=False, error_message="Not connected")]

        positions = self._mt5.positions_get()
        if not positions:
            log.info("No open positions to close")
            return []

        log.warning(f"🚨 CLOSING ALL {len(positions)} POSITIONS!")
        results = []
        for pos in positions:
            result = self.close_position(pos.ticket)
            results.append(result)

        return results

    def get_open_positions(self) -> list[PositionInfo]:
        """Get all currently open positions.

        Returns:
            List of PositionInfo for active positions.
        """
        if not self.is_connected:
            return []

        positions = self._mt5.positions_get()
        if not positions:
            return []

        mt5 = self._mt5
        result = []
        for pos in positions:
            result.append(PositionInfo(
                ticket=pos.ticket,
                symbol=pos.symbol,
                side="BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                lot=pos.volume,
                open_price=pos.price_open,
                current_price=pos.price_current,
                profit=pos.profit,
                sl=pos.sl,
                tp=pos.tp,
            ))
        return result
