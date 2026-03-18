"""
Order boshqaruv moduli.
Paper trading simulatsiyasi va real order bajarilishini boshqaradi.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import yfinance as yf

from config import Config
from database.trade_history import TradeHistory
from bot.risk_manager import RiskManager
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Trade:
    """Xotiradagi trade obyekti."""
    trade_id: int
    symbol: str
    direction: str
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    tp3_price: float
    lot_size: float
    open_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "OCHIQ"
    tp1_bajarildi: bool = False
    tp2_bajarildi: bool = False


class OrderManager:
    """
    Trade hayot siklini boshqaradi: ochish, kuzatish,
    qisman yopish va to'liq yopish. DEMO_MODE=True bo'lsa
    paper trading rejimida ishlaydi.
    """

    def __init__(
        self,
        config: Config,
        db: TradeHistory,
        risk_manager: RiskManager,
        telegram_bot: Any,
    ) -> None:
        self.config = config
        self.db = db
        self.risk_manager = risk_manager
        self.telegram_bot = telegram_bot
        self._paper_balans: float = config.initial_balance
        self._paper_boshlangich_balans: float = config.initial_balance
        self._ochiq_tradelar: Dict[int, Trade] = {}
        self._ulangan: bool = False

    async def connect(self) -> None:
        """Broker bilan ulanish (yoki paper mode ni tasdiqlash)."""
        if self.config.demo_mode:
            logger.info("Paper trading rejimi faol. Virtual balans: $%.2f", self._paper_balans)
            self._ulangan = True
            return

        try:
            import ccxt.async_support as ccxt  # type: ignore
            exchange_cls = getattr(ccxt, self.config.broker_name.lower(), None)
            if exchange_cls is None:
                raise ValueError(f"Noma'lum birja: {self.config.broker_name}")
            self._exchange = exchange_cls({
                "apiKey": self.config.broker_api_key,
                "secret": self.config.broker_api_secret,
            })
            await self._exchange.load_markets()
            self._ulangan = True
            logger.info("%s birjasiga ulandi", self.config.broker_name)
        except Exception as xato:
            logger.error("Broker ulanishi muvaffaqiyatsiz: %s – paper rejimga o'tildi", xato)
            self.config.__dict__["demo_mode"] = True
            self._ulangan = True

    async def get_account_balance(self) -> float:
        """Joriy balansni qaytarish (paper rejimda virtual)."""
        if self.config.demo_mode:
            return self._paper_balans
        try:
            balans = await self._exchange.fetch_balance()
            return float(balans["total"].get("USD", self._paper_balans))
        except Exception as xato:
            logger.error("Balans olishda xato: %s", xato)
            return self._paper_balans

    async def get_current_price(self, symbol: Optional[str] = None) -> float:
        """Joriy narxni yfinance orqali olish."""
        sym = symbol or self.config.yfinance_symbol
        loop = asyncio.get_event_loop()
        try:
            def _narx_ol() -> float:
                ticker = yf.Ticker(sym)
                hist = ticker.history(period="1d", interval="1m")
                if hist.empty:
                    return 0.0
                return float(hist["Close"].iloc[-1])

            narx = await loop.run_in_executor(None, _narx_ol)
            return narx
        except Exception as xato:
            logger.error("Narx olishda xato: %s", xato)
            return 0.0

    async def open_trade(self, signal) -> Optional[Trade]:
        """Signal asosida yangi trade ochish."""
        if not self._ulangan:
            await self.connect()

        balans = await self.get_account_balance()
        lot_hajmi = self.risk_manager.calculate_position_size(
            balans, signal.entry_price, signal.sl_price
        )

        kirish_narxi = signal.entry_price
        if self.config.demo_mode:
            joriy_narx = await self.get_current_price()
            if joriy_narx > 0:
                kirish_narxi = joriy_narx

        try:
            trade_id = await self.db.save_trade(
                symbol=self.config.symbol,
                direction=signal.direction,
                entry_price=kirish_narxi,
                sl_price=signal.sl_price,
                tp1_price=signal.tp1,
                tp2_price=signal.tp2,
                tp3_price=signal.tp3,
                lot_size=lot_hajmi,
                signal_strength=int(signal.strength),
                confirmations=signal.confirmations,
                metadata={"reasons": signal.reasons},
            )
        except Exception as xato:
            logger.exception("Trade ma'lumotlar bazasiga saqlashda xato: %s", xato)
            return None

        trade = Trade(
            trade_id=trade_id,
            symbol=self.config.symbol,
            direction=signal.direction,
            entry_price=kirish_narxi,
            sl_price=signal.sl_price,
            tp1_price=signal.tp1,
            tp2_price=signal.tp2,
            tp3_price=signal.tp3,
            lot_size=lot_hajmi,
        )
        self._ochiq_tradelar[trade_id] = trade

        logger.info(
            "Trade ochildi: #%d %s %s @ %.2f | SL=%.2f | TP1=%.2f | Lot=%.2f",
            trade_id,
            signal.direction,
            self.config.symbol,
            kirish_narxi,
            signal.sl_price,
            signal.tp1,
            lot_hajmi,
        )

        if self.telegram_bot:
            try:
                await self.telegram_bot.send_trade_opened(trade)
            except Exception as xato:
                logger.warning("Telegram xabari yuborishda xato: %s", xato)

        return trade

    async def close_trade(
        self,
        trade_id: int,
        sabab: str,
        chiqish_narxi: Optional[float] = None,
    ) -> bool:
        """Tradeni yopish va natijani saqlash."""
        trade = self._ochiq_tradelar.get(trade_id)
        if trade is None:
            logger.warning("close_trade: #%d trade xotirada topilmadi", trade_id)
            db_trade = await self.db.get_trade(trade_id)
            if not db_trade or db_trade["status"] != "OPEN":
                return False
            trade = Trade(
                trade_id=trade_id,
                symbol=db_trade["symbol"],
                direction=db_trade["direction"],
                entry_price=db_trade["entry_price"],
                sl_price=db_trade["sl_price"],
                tp1_price=db_trade["tp1_price"],
                tp2_price=db_trade["tp2_price"],
                tp3_price=db_trade["tp3_price"],
                lot_size=db_trade["lot_size"],
            )

        if chiqish_narxi is None:
            chiqish_narxi = await self.get_current_price()

        foyda_zarar = self._foyda_hisoblash(trade, chiqish_narxi)
        pip_foyda = (chiqish_narxi - trade.entry_price) * (
            1 if trade.direction == "BUY" else -1
        ) * 100

        if self.config.demo_mode:
            self._paper_balans += foyda_zarar

        await self.db.close_trade(trade_id, chiqish_narxi, foyda_zarar, pip_foyda, sabab)
        self._ochiq_tradelar.pop(trade_id, None)
        await self.db.upsert_daily_stats()

        natija = "FOYDA" if foyda_zarar >= 0 else "ZARAR"
        logger.info(
            "Trade yopildi: #%d | Sabab: %s | Chiqish: %.2f | %s: $%.2f",
            trade_id, sabab, chiqish_narxi, natija, foyda_zarar,
        )

        if self.telegram_bot:
            try:
                if sabab.startswith("SL"):
                    await self.telegram_bot.send_sl_hit(trade, foyda_zarar)
                elif sabab.startswith("TP"):
                    daraja = sabab.replace("TP", "").strip()
                    await self.telegram_bot.send_tp_hit(trade, daraja, foyda_zarar)
            except Exception as xato:
                logger.warning("Telegram xabari yuborishda xato: %s", xato)

        return True

    async def partial_close(self, trade_id: int, foiz: float, sabab: str) -> bool:
        """Tradeni qisman yopish (lot hajmini kamaytirish)."""
        trade = self._ochiq_tradelar.get(trade_id)
        if trade is None:
            logger.warning("partial_close: #%d trade keshda topilmadi", trade_id)
            return False

        if not (0 < foiz <= 1.0):
            logger.error("partial_close: foiz (0, 1] oralig'ida bo'lishi kerak, berilgan: %.2f", foiz)
            return False

        joriy_narx = await self.get_current_price()
        yopilayotgan_lot = round(trade.lot_size * foiz, 2)
        qolgan_lot = round(trade.lot_size - yopilayotgan_lot, 2)
        qisman_foyda = self._lot_uchun_foyda(trade, joriy_narx, yopilayotgan_lot)

        if self.config.demo_mode:
            self._paper_balans += qisman_foyda

        if qolgan_lot < 0.01:
            return await self.close_trade(trade_id, sabab, joriy_narx)

        trade.lot_size = qolgan_lot
        await self.db.update_trade(trade_id, lot_size=qolgan_lot)
        logger.info(
            "Qisman yopildi #%d: %.0f%% (%.2f lot) | Foyda: $%.2f | Qolgan: %.2f lot",
            trade_id, foiz * 100, yopilayotgan_lot, qisman_foyda, qolgan_lot,
        )
        return True

    async def update_stop_loss(self, trade_id: int, yangi_sl: float) -> bool:
        """Stop Loss narxini yangilash."""
        trade = self._ochiq_tradelar.get(trade_id)
        if trade:
            trade.sl_price = yangi_sl
        await self.db.update_trade(trade_id, sl_price=yangi_sl)
        logger.debug("SL yangilandi: #%d yangi SL=%.3f", trade_id, yangi_sl)
        return True

    async def monitor_open_trades(self) -> None:
        """
        Barcha ochiq tradelarni joriy narxga nisbatan tekshirish.
        TP yoki SL ishlagan bo'lsa yopadi.
        """
        if not self._ochiq_tradelar:
            db_tradelar = await self.db.get_open_trades()
            for t in db_tradelar:
                if t["id"] not in self._ochiq_tradelar:
                    self._ochiq_tradelar[t["id"]] = Trade(
                        trade_id=t["id"],
                        symbol=t["symbol"],
                        direction=t["direction"],
                        entry_price=t["entry_price"],
                        sl_price=t["sl_price"],
                        tp1_price=t["tp1_price"],
                        tp2_price=t["tp2_price"],
                        tp3_price=t["tp3_price"],
                        lot_size=t["lot_size"],
                    )

        if not self._ochiq_tradelar:
            return

        joriy_narx = await self.get_current_price()
        if joriy_narx <= 0:
            logger.warning("monitor_open_trades: noto'g'ri narx %.3f", joriy_narx)
            return

        atr = joriy_narx * 0.005

        for trade_id, trade in list(self._ochiq_tradelar.items()):
            yonalish = trade.direction

            # Stop Loss tekshirish
            sl_ishladi = (
                (yonalish == "BUY" and joriy_narx <= trade.sl_price)
                or (yonalish == "SELL" and joriy_narx >= trade.sl_price)
            )
            if sl_ishladi:
                await self.close_trade(trade_id, "SL ishladi", trade.sl_price)
                continue

            # Maqsad 1 – qisman yopish
            tp1_ishladi = (
                (yonalish == "BUY" and joriy_narx >= trade.tp1_price)
                or (yonalish == "SELL" and joriy_narx <= trade.tp1_price)
            )
            if tp1_ishladi and not trade.tp1_bajarildi:
                trade.tp1_bajarildi = True
                await self.partial_close(trade_id, 0.33, "TP1 ishladi")
                await self.update_stop_loss(trade_id, trade.entry_price)
                logger.info("TP1 ishladi #%d – 33%% yopildi, SL kirish narxiga ko'chirildi", trade_id)
                if self.telegram_bot:
                    try:
                        qisman_foyda = self._lot_uchun_foyda(trade, joriy_narx, trade.lot_size * 0.33)
                        await self.telegram_bot.send_tp_hit(trade, "1", qisman_foyda)
                    except Exception:
                        pass
                continue

            # Maqsad 2 – qisman yopish
            tp2_ishladi = (
                (yonalish == "BUY" and joriy_narx >= trade.tp2_price)
                or (yonalish == "SELL" and joriy_narx <= trade.tp2_price)
            )
            if tp2_ishladi and trade.tp1_bajarildi and not trade.tp2_bajarildi:
                trade.tp2_bajarildi = True
                await self.partial_close(trade_id, 0.50, "TP2 ishladi")
                logger.info("TP2 ishladi #%d – qolgan lotning 50%% yopildi", trade_id)
                if self.telegram_bot:
                    try:
                        qisman_foyda = self._lot_uchun_foyda(trade, joriy_narx, trade.lot_size * 0.50)
                        await self.telegram_bot.send_tp_hit(trade, "2", qisman_foyda)
                    except Exception:
                        pass
                continue

            # Maqsad 3 – to'liq yopish
            tp3_ishladi = (
                (yonalish == "BUY" and joriy_narx >= trade.tp3_price)
                or (yonalish == "SELL" and joriy_narx <= trade.tp3_price)
            )
            if tp3_ishladi and trade.tp2_bajarildi:
                await self.close_trade(trade_id, "TP3 ishladi", trade.tp3_price)
                continue

            # Trailing Stop yangilash
            yangi_sl = self.risk_manager.update_trailing_stop(
                {"direction": yonalish, "sl_price": trade.sl_price, "entry_price": trade.entry_price},
                joriy_narx, atr,
            )
            if yangi_sl is not None:
                await self.update_stop_loss(trade_id, yangi_sl)

            # Break-even tekshirish
            if self.risk_manager.should_breakeven(
                {"direction": yonalish, "entry_price": trade.entry_price, "sl_price": trade.sl_price},
                joriy_narx, atr=atr,
            ):
                await self.update_stop_loss(trade_id, trade.entry_price)
                logger.info("Break-even qo'llanildi: #%d", trade_id)

    async def check_time_based_exit(self) -> None:
        """48 soatdan ko'p ochiq qolgan tradelarni yopish."""
        chegara = datetime.now(timezone.utc) - timedelta(hours=48)
        for trade_id, trade in list(self._ochiq_tradelar.items()):
            open_time = trade.open_time
            if open_time.tzinfo is None:
                open_time = open_time.replace(tzinfo=timezone.utc)
            if open_time < chegara:
                logger.info("Vaqt bo'yicha yopildi: #%d (48 soatdan ko'p ochiq)", trade_id)
                await self.close_trade(trade_id, "Vaqt limiti (48 soat)")

    def _foyda_hisoblash(self, trade: Trade, chiqish_narxi: float) -> float:
        """To'liq lot uchun foyda/zarar hisoblash."""
        return self._lot_uchun_foyda(trade, chiqish_narxi, trade.lot_size)

    @staticmethod
    def _lot_uchun_foyda(trade: Trade, chiqish_narxi: float, lotlar: float) -> float:
        """
        Berilgan lot uchun foyda/zarar hisoblash.
        1 lot = 100 oz; 1 pip ($0.01) = lotga $1.
        """
        pip_farqi = (chiqish_narxi - trade.entry_price) * 100
        if trade.direction == "SELL":
            pip_farqi = -pip_farqi
        return round(pip_farqi * lotlar * 1.0, 2)

    @property
    def paper_balance(self) -> float:
        """Joriy virtual balans."""
        return self._paper_balans

    @property
    def paper_initial_balance(self) -> float:
        """Boshlang'ich virtual balans."""
        return self._paper_boshlangich_balans
