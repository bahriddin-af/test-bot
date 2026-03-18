"""
Risk boshqaruv moduli.
Pozitsiya hajmini hisoblash, SL/TP belgilash va trade tekshirishni amalga oshiradi.
"""

from __future__ import annotations

from typing import Optional, Tuple

from config import Config
from database.trade_history import TradeHistory
from utils.logger import get_logger

logger = get_logger(__name__)

# Oltin uchun pip qiymati: 1 lot = 100 oz, 1 pip = $0.01 = lotga $1
OLTIN_PIP_QIYMATI = 1.0
OLTIN_SHARTNOMA_HAJMI = 100  # oz per standart lot


class RiskManager:
    """Risk hisob-kitoblari va trade tekshirishni boshqaradi."""

    def __init__(self, config: Config, db: TradeHistory) -> None:
        self.config = config
        self.db = db

    def calculate_position_size(
        self,
        hisob_balansi: float,
        kirish_narxi: float,
        sl_narxi: float,
        risk_foiz: Optional[float] = None,
    ) -> float:
        """
        Stop Loss ishlasa aniq *risk_foiz* yo'qotilishi uchun lot hajmini hisoblash.

        Qaytaradi:
            Lot hajmi (minimum 0.01, maksimum 10.0).
        """
        rp = risk_foiz if risk_foiz is not None else self.config.risk_percent
        risk_miqdori = hisob_balansi * (rp / 100.0)
        sl_pip_masofasi = abs(kirish_narxi - sl_narxi) * 100

        if sl_pip_masofasi == 0:
            logger.warning("SL masofasi nol – minimum lot hajmi qaytariladi")
            return 0.01

        dollar_risk_per_lot = sl_pip_masofasi * OLTIN_PIP_QIYMATI
        lot_hajmi = risk_miqdori / dollar_risk_per_lot
        lot_hajmi = max(0.01, min(10.0, round(lot_hajmi, 2)))

        logger.debug(
            "Pozitsiya hajmi: balans=%.2f risk=%.1f%% SL_pip=%.1f lot=%.2f",
            hisob_balansi, rp, sl_pip_masofasi, lot_hajmi,
        )
        return lot_hajmi

    def calculate_sl_price(
        self,
        kirish_narxi: float,
        yonalish: str,
        atr: float,
        multiplikator: Optional[float] = None,
    ) -> float:
        """ATR asosida Stop Loss narxini hisoblash."""
        mult = multiplikator if multiplikator is not None else self.config.atr_multiplier_sl
        masofa = atr * mult
        if yonalish == "BUY":
            return round(kirish_narxi - masofa, 3)
        return round(kirish_narxi + masofa, 3)

    def calculate_tp_levels(
        self,
        kirish_narxi: float,
        sl_narxi: float,
        yonalish: str,
    ) -> Tuple[float, float, float]:
        """
        Uch maqsad darajasini hisoblash.
        TP1 = 1×RR, TP2 = 1.5×RR, TP3 = 2.5×RR.
        """
        sl_masofasi = abs(kirish_narxi - sl_narxi)
        rr = self.config.rr_ratio_min

        if yonalish == "BUY":
            tp1 = kirish_narxi + sl_masofasi * rr
            tp2 = kirish_narxi + sl_masofasi * rr * 1.5
            tp3 = kirish_narxi + sl_masofasi * rr * 2.5
        else:
            tp1 = kirish_narxi - sl_masofasi * rr
            tp2 = kirish_narxi - sl_masofasi * rr * 1.5
            tp3 = kirish_narxi - sl_masofasi * rr * 2.5

        return round(tp1, 3), round(tp2, 3), round(tp3, 3)

    async def check_daily_loss_limit(self, hisob_balansi: float) -> bool:
        """
        Kunlik zarar limiti oshib ketmadimi tekshirish.
        True = savdo qilsa bo'ladi, False = limit oshgan.
        """
        bugungi_foyda_zarar = await self.db.get_today_pnl()
        if bugungi_foyda_zarar >= 0:
            return True

        zarar_foiz = abs(bugungi_foyda_zarar) / hisob_balansi * 100.0
        if zarar_foiz >= self.config.max_daily_loss_percent:
            logger.warning(
                "Kunlik zarar limiti yetildi: %.2f%% (maksimum %.2f%%)",
                zarar_foiz, self.config.max_daily_loss_percent,
            )
            return False
        return True

    async def check_drawdown_limit(
        self,
        hisob_balansi: float,
        boshlangich_balans: float,
    ) -> bool:
        """
        Maksimal drawdown limiti oshib ketmadimi tekshirish.
        True = savdo qilsa bo'ladi, False = limit oshgan.
        """
        if boshlangich_balans <= 0:
            return True
        drawdown_foiz = (boshlangich_balans - hisob_balansi) / boshlangich_balans * 100.0
        if drawdown_foiz >= self.config.max_drawdown_percent:
            logger.warning(
                "Maksimal drawdown yetildi: %.2f%% (maksimum %.2f%%)",
                drawdown_foiz, self.config.max_drawdown_percent,
            )
            return False
        return True

    async def check_max_open_trades(self) -> bool:
        """
        Ochiq tradelar soni limitdan oshib ketmadimi tekshirish.
        True = yangi trade ochsa bo'ladi.
        """
        ochiq_soni = await self.db.count_open_trades()
        if ochiq_soni >= self.config.max_trades:
            logger.info(
                "Maksimal ochiq trade soni: %d/%d", ochiq_soni, self.config.max_trades
            )
            return False
        return True

    async def validate_trade(
        self,
        signal,
        hisob_balansi: float,
        boshlangich_balans: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Trade ochishdan oldin barcha risk tekshiruvlarini o'tkazish.

        Qaytaradi:
            (True, "OK") yoki (False, sabab_matni).
        """
        if boshlangich_balans is None:
            boshlangich_balans = hisob_balansi

        if not await self.check_max_open_trades():
            return False, f"Maksimal ochiq trade soni ({self.config.max_trades}) yetildi"

        if not await self.check_daily_loss_limit(hisob_balansi):
            return False, f"Kunlik zarar limiti ({self.config.max_daily_loss_percent}%) yetildi"

        if not await self.check_drawdown_limit(hisob_balansi, boshlangich_balans):
            return False, f"Maksimal drawdown ({self.config.max_drawdown_percent}%) yetildi"

        if hisob_balansi < 100.0:
            return False, "Hisob balansi minimal chegara ($100) dan past"

        if signal.sl_price <= 0:
            return False, "SL narxi noto'g'ri"

        sl_masofasi = abs(signal.entry_price - signal.sl_price)
        if sl_masofasi > signal.entry_price * 0.05:
            return False, "SL masofasi kirish narxining 5% dan oshdi – trade rad etildi"

        logger.info("Trade tekshiruvi muvaffaqiyatli: %s signali", signal.direction)
        return True, "OK"

    def update_trailing_stop(
        self,
        trade: dict,
        joriy_narx: float,
        atr: float,
        trail_multiplikator: float = 1.5,
    ) -> Optional[float]:
        """
        Trailing stop yangi narxini hisoblash.
        Foyda bo'lgan tomonga SL ni siljitadi.

        Qaytaradi:
            Yangi SL narxi yoki None (o'zgartirish kerak bo'lmasa).
        """
        yonalish = trade.get("direction", "BUY")
        joriy_sl = float(trade.get("sl_price", 0))
        trail_masofasi = atr * trail_multiplikator

        if yonalish == "BUY":
            yangi_sl = round(joriy_narx - trail_masofasi, 3)
            if yangi_sl > joriy_sl:
                logger.debug("Trailing stop BUY: %.3f → %.3f", joriy_sl, yangi_sl)
                return yangi_sl
        else:
            yangi_sl = round(joriy_narx + trail_masofasi, 3)
            if yangi_sl < joriy_sl or joriy_sl == 0:
                logger.debug("Trailing stop SELL: %.3f → %.3f", joriy_sl, yangi_sl)
                return yangi_sl
        return None

    def should_breakeven(
        self,
        trade: dict,
        joriy_narx: float,
        breakeven_atr_nisbati: float = 0.8,
        atr: Optional[float] = None,
    ) -> bool:
        """
        SL ni kirish narxiga ko'chirish kerakmi tekshirish.
        Narx ATR ning *breakeven_atr_nisbati* qismi qadar
        kirish narxidan uzoqlashganda True qaytaradi.
        """
        yonalish = trade.get("direction", "BUY")
        kirish_narxi = float(trade.get("entry_price", joriy_narx))
        joriy_sl = float(trade.get("sl_price", 0))

        if atr is None:
            atr = kirish_narxi * 0.005

        foyda_maqsadi = atr * breakeven_atr_nisbati

        if yonalish == "BUY":
            if joriy_sl >= kirish_narxi:
                return False
            return joriy_narx >= kirish_narxi + foyda_maqsadi
        else:
            if joriy_sl > 0 and joriy_sl <= kirish_narxi:
                return False
            return joriy_narx <= kirish_narxi - foyda_maqsadi
