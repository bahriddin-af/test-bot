"""
Oltin Trading Bot – asosiy ishga tushirish nuqtasi.
Tugmalar, polling va asosiy trading loopni boshqaradi.
"""

from __future__ import annotations

import asyncio
import signal
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from config import get_config, Config
from database.trade_history import TradeHistory, initialize_db
from bot.signal_engine import SignalEngine
from bot.risk_manager import RiskManager
from bot.order_manager import OrderManager
from bot.news_filter import NewsFilter
from bot.telegram_bot import TelegramBot
from utils.chart_generator import ChartGenerator
from utils.logger import get_logger

logger = get_logger(__name__)

SIKL_INTERVAL_SEKUND = 300
YURAK_URISHI_SIKL = 12
KUNLIK_HISOBOT_SOAT_UTC = 22


class GoldBot:
    """Barcha quyi tizimlarni birlashtiruvchi asosiy bot sinfi."""

    def __init__(self) -> None:
        self.config: Config = get_config()
        self.db: Optional[TradeHistory] = None
        self.signal_engine: Optional[SignalEngine] = None
        self.risk_manager: Optional[RiskManager] = None
        self.order_manager: Optional[OrderManager] = None
        self.news_filter: Optional[NewsFilter] = None
        self.telegram: Optional[TelegramBot] = None
        self.chart_gen: Optional[ChartGenerator] = None
        self._ishlamoqda: bool = False
        self._sikl_soni: int = 0
        self._oxirgi_hisobot_kuni: Optional[int] = None

    # ------------------------------------------------------------------
    # Foyda / zarar hisoblash
    # ------------------------------------------------------------------

    def _pnl_hisoblash(self, signal_obj: Any, balans: float) -> Dict:
        """Signal uchun kutilgan foyda va zarar miqdorlarini hisoblash."""
        try:
            lot = self.risk_manager.calculate_position_size(
                balans, signal_obj.entry_price, signal_obj.sl_price
            )

            def _pnl(chiqish: float) -> float:
                pip = (chiqish - signal_obj.entry_price) * 100
                if signal_obj.direction == "SELL":
                    pip = -pip
                return round(pip * lot, 2)

            sl_z  = _pnl(signal_obj.sl_price)
            tp1_f = _pnl(signal_obj.tp1)
            tp2_f = _pnl(signal_obj.tp2)
            tp3_f = _pnl(signal_obj.tp3)
            sl_d  = abs(signal_obj.entry_price - signal_obj.sl_price)
            tp1_d = abs(signal_obj.tp1 - signal_obj.entry_price)
            rr    = round(tp1_d / sl_d, 1) if sl_d > 0 else 0

            return {
                "lot": lot,
                "sl_zarar": sl_z,
                "tp1_foyda": tp1_f,
                "tp2_foyda": tp2_f,
                "tp3_foyda": tp3_f,
                "rr": rr,
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Tugma callbacklari
    # ------------------------------------------------------------------

    async def _signal_tasdiqlandi(self, signal_obj: Any) -> None:
        """Foydalanuvchi ✅ Tasdiqlash bosganda trade ochish."""
        try:
            balans = await self.order_manager.get_account_balance()
            boshlangich = self.order_manager.paper_initial_balance
            ok, sabab = await self.risk_manager.validate_trade(signal_obj, balans, boshlangich)
            if not ok:
                await self.telegram.xabar_yuborish(f"⚠️ Trade rad etildi: {sabab}")
                return
            trade = await self.order_manager.open_trade(signal_obj)
            if trade:
                logger.info("Tasdiqlangan trade #%d ochildi: %s @ %.2f",
                            trade.trade_id, trade.direction, trade.entry_price)
        except Exception as xato:
            logger.exception("Tasdiqlangan trade ochishda xato: %s", xato)
            await self.telegram.send_error_alert(str(xato))

    async def _holat_handler(self, update: Any, context: Any) -> None:
        """📊 Holat tugmasi."""
        balans = await self.order_manager.get_account_balance()
        ochiq = len(self.order_manager._ochiq_tradelar)
        boshlangich = self.order_manager.paper_initial_balance
        foyda = balans - boshlangich
        foyda_emoji = "💚" if foyda >= 0 else "❤️"
        matn = (
            f"📊 <b>BOT HOLATI</b>\n\n"
            f"🔄 Holat: Ishlamoqda ✅\n"
            f"📌 Rejim: {'PAPER (Sinov)' if self.config.demo_mode else 'REAL'}\n"
            f"💰 <b>Joriy balans:</b> <code>${balans:.2f}</code>\n"
            f"{foyda_emoji} <b>Umumiy foyda/zarar:</b> <code>${foyda:+.2f}</code>\n"
            f"📋 <b>Ochiq tradelar:</b> {ochiq}/{self.config.max_trades}\n"
            f"🔁 <b>Sikl:</b> #{self._sikl_soni}\n"
            f"🕐 <code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</code>"
        )
        await update.message.reply_text(matn, parse_mode="HTML")

    async def _balans_handler(self, update: Any, context: Any) -> None:
        """💰 Balans tugmasi."""
        balans = await self.order_manager.get_account_balance()
        boshlangich = self.order_manager.paper_initial_balance
        foyda = balans - boshlangich
        foyda_foiz = (foyda / boshlangich * 100) if boshlangich > 0 else 0
        foyda_emoji = "💚" if foyda >= 0 else "❤️"
        matn = (
            f"💰 <b>HISOB BALANSI</b>\n\n"
            f"🏦 <b>Boshlang'ich:</b> <code>${boshlangich:.2f}</code>\n"
            f"💵 <b>Joriy:</b> <code>${balans:.2f}</code>\n"
            f"{foyda_emoji} <b>Foyda/zarar:</b> <code>${foyda:+.2f} ({foyda_foiz:+.1f}%)</code>"
        )
        await update.message.reply_text(matn, parse_mode="HTML")

    async def _togtat_handler(self, update: Any, context: Any) -> None:
        """🛑 To'xtat tugmasi."""
        await self.stop()

    async def _hisobot_handler(self, update: Any, context: Any) -> None:
        """📈 Hisobot tugmasi."""
        try:
            statistika = await self.db.get_daily_stats() or {}
            kapital = await self.db.get_equity_curve()
            kapital_bytes = None
            if kapital and self.chart_gen:
                kapital_bytes = self.chart_gen.generate_equity_curve(kapital)
            await self.telegram.send_daily_report(statistika, kapital_bytes)
        except Exception as xato:
            await update.message.reply_text(f"⚠️ Hisobot olishda xato: {xato}")

    async def _ochiq_tradelar_handler(self, update: Any, context: Any) -> None:
        """📋 Ochiq tradelar — har biri uchun Yopish tugmasi bilan."""
        from telegram import InlineKeyboardMarkup, InlineKeyboardButton

        tradelar = self.order_manager._ochiq_tradelar
        if not tradelar:
            await update.message.reply_text("📋 Hozircha ochiq trade yo'q.", parse_mode="HTML")
            return

        joriy_narx = await self.order_manager.get_current_price()

        for trade_id, trade in tradelar.items():
            foyda = OrderManager._lot_uchun_foyda(trade, joriy_narx, trade.lot_size)
            f_emoji = "💚" if foyda >= 0 else "❤️"
            yo_nomi = "BUY 📈" if trade.direction == "BUY" else "SELL 📉"

            matn = (
                f"📌 <b>Trade #{trade_id}</b> — {yo_nomi}\n\n"
                f"💰 <b>Kirish:</b> <code>{trade.entry_price:.2f}</code>\n"
                f"📍 <b>Joriy narx:</b> <code>{joriy_narx:.2f}</code>\n"
                f"🛑 <b>Stop Loss:</b> <code>{trade.sl_price:.2f}</code>\n"
                f"🎯 <b>TP1:</b> <code>{trade.tp1_price:.2f}</code>\n"
                f"📦 <b>Lot:</b> {trade.lot_size:.2f}\n"
                f"{f_emoji} <b>Hozirgi foyda/zarar:</b> <code>${foyda:+.2f}</code>"
            )
            tugmalar = InlineKeyboardMarkup([[
                InlineKeyboardButton(
                    f"🔴 #{trade_id} ni yopish",
                    callback_data=f"yopish:{trade_id}"
                )
            ]])
            await update.message.reply_text(matn, parse_mode="HTML", reply_markup=tugmalar)

    async def _trade_yopish_handler(self, trade_id: int, query: Any) -> None:
        """Yopish tugmasi bosilganda trade ni yopish."""
        try:
            joriy_narx = await self.order_manager.get_current_price()
            trade = self.order_manager._ochiq_tradelar.get(trade_id)
            if not trade:
                await query.edit_message_text("⚠️ Bu trade allaqachon yopilgan.")
                return
            foyda = OrderManager._lot_uchun_foyda(trade, joriy_narx, trade.lot_size)
            f_emoji = "💚" if foyda >= 0 else "❤️"
            muvaffaqiyat = await self.order_manager.close_trade(
                trade_id, "Qo'lda yopildi", joriy_narx
            )
            if muvaffaqiyat:
                await query.edit_message_text(
                    f"✅ <b>Trade #{trade_id} yopildi</b>\n\n"
                    f"📍 Chiqish narxi: <code>{joriy_narx:.2f}</code>\n"
                    f"{f_emoji} Foyda/zarar: <code>${foyda:+.2f}</code>",
                    parse_mode="HTML"
                )
            else:
                await query.edit_message_text("⚠️ Trade yopishda xato yuz berdi.")
        except Exception as xato:
            logger.exception("Trade yopishda xato: %s", xato)
            await query.edit_message_text(f"⚠️ Xato: {xato}")

    # ------------------------------------------------------------------
    # Ishga tushirish
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Barcha quyi tizimlarni ishga tushirish."""
        logger.info("Oltin Trading Bot ishga tushirilmoqda...")

        await initialize_db(self.config.db_path)
        self.db = TradeHistory(self.config.db_path)

        self.telegram = TelegramBot(
            token=self.config.telegram_bot_token,
            chat_id=self.config.telegram_chat_id,
        )

        # Tugma callbacklarini ulash
        self.telegram.signal_tasdiqlandi_cb = self._signal_tasdiqlandi
        self.telegram.holat_cb = self._holat_handler
        self.telegram.balans_cb = self._balans_handler
        self.telegram.togtat_cb = self._togtat_handler
        self.telegram.hisobot_cb = self._hisobot_handler
        self.telegram.ochiq_tradelar_cb = self._ochiq_tradelar_handler
        self.telegram.trade_yopish_cb = self._trade_yopish_handler

        self.signal_engine = SignalEngine(self.config)
        self.risk_manager = RiskManager(self.config, self.db)
        self.order_manager = OrderManager(
            self.config, self.db, self.risk_manager, self.telegram
        )
        self.news_filter = NewsFilter(self.config)
        self.chart_gen = ChartGenerator()

        await self.order_manager.connect()

        # Telegram polling ni fon rejimida ishga tushirish
        await self.telegram.start_polling()

        rejim = "PAPER (Sinov)" if self.config.demo_mode else "REAL (Haqiqiy)"
        logger.info("Bot tayyor. Rejim: %s | Juftlik: %s", rejim, self.config.formatted_symbol)

        # Boshlang'ich xabar + menyu klaviatura
        await self.telegram.send_message(
            f"🤖 <b>Oltin Trading Bot ishga tushdi!</b>\n\n"
            f"📌 <b>Rejim:</b> {rejim}\n"
            f"💰 <b>Juftlik:</b> {self.config.formatted_symbol}\n"
            f"⏱ <b>Vaqt oralig'i:</b> {self.config.timeframe}\n"
            f"⚠️ <b>Risk per trade:</b> {self.config.risk_percent}%\n"
            f"💵 <b>Virtual balans:</b> $10,000\n\n"
            f"⬇️ Quyidagi tugmalardan foydalaning:"
        )
        await self.telegram.menyu_klaviatura_yuborish()

    async def run_cycle(self) -> None:
        """Bir to'liq trading sikli."""
        self._sikl_soni += 1
        logger.debug("--- %d-sikl boshlanmoqda ---", self._sikl_soni)

        if self.config.news_filter_enabled:
            await self.news_filter.fetch_news_calendar()

        if self.news_filter.is_news_time(self.config.symbol):
            logger.info("Yangilik bloki faol – signal yaratilmaydi")
            await self.order_manager.monitor_open_trades()
            return

        await self.order_manager.monitor_open_trades()
        await self.order_manager.check_time_based_exit()

        try:
            signal_obj = await self.signal_engine.generate_signal()
        except Exception as xato:
            logger.exception("Signal yaratishda xato: %s", xato)
            if self.telegram:
                await self.telegram.send_error_alert(f"Signal yaratishda xato:\n{xato}")
            return

        if signal_obj is None:
            logger.info("Bu siklda signal yo'q")
            return

        # Grafik yaratish
        grafik_bytes: Optional[bytes] = None
        df_1h = self.signal_engine._cached_data.get("1h")
        if df_1h is not None and not df_1h.empty and self.chart_gen:
            try:
                grafik_bytes = self.chart_gen.generate_signal_chart(df_1h, signal_obj)
            except Exception as xato:
                logger.warning("Grafik yaratishda xato: %s", xato)

        # Foyda / zarar hisoblash
        balans = await self.order_manager.get_account_balance()
        pnl_info = self._pnl_hisoblash(signal_obj, balans)

        # Signalni inline tugmalar bilan yuborish (foydalanuvchi tasdiqlaydi)
        if self.telegram:
            await self.telegram.send_signal_alert(signal_obj, grafik_bytes, pnl_info)

        if self._sikl_soni % YURAK_URISHI_SIKL == 0 and self.telegram:
            await self.telegram.send_heartbeat()

    def _kunlik_hisobot_yuborilsinmi(self) -> bool:
        hozir = datetime.now(timezone.utc)
        if hozir.hour == KUNLIK_HISOBOT_SOAT_UTC and self._oxirgi_hisobot_kuni != hozir.day:
            self._oxirgi_hisobot_kuni = hozir.day
            return True
        return False

    async def _kunlik_hisobot_yuborish(self) -> None:
        if not self.telegram or not self.db:
            return
        try:
            statistika = await self.db.get_daily_stats()
            if statistika is None:
                await self.db.upsert_daily_stats()
                statistika = await self.db.get_daily_stats() or {}
            kapital = await self.db.get_equity_curve()
            kapital_bytes = None
            if kapital and self.chart_gen:
                kapital_bytes = self.chart_gen.generate_equity_curve(kapital)
            await self.telegram.send_daily_report(statistika, kapital_bytes)
            logger.info("Kunlik hisobot yuborildi")
        except Exception as xato:
            logger.exception("Kunlik hisobot yuborishda xato: %s", xato)

    async def start(self) -> None:
        """Asosiy async loop."""
        await self.initialize()
        self._ishlamoqda = True
        logger.info("Bot ishlayapti. To'xtatish uchun Ctrl+C bosing.")

        while self._ishlamoqda:
            sikl_boshi = asyncio.get_event_loop().time()
            try:
                await self.run_cycle()
            except Exception as xato:
                logger.exception("run_cycle da xato: %s", xato)
                if self.telegram:
                    await self.telegram.send_error_alert(str(xato))

            if self._kunlik_hisobot_yuborilsinmi():
                await self._kunlik_hisobot_yuborish()

            otgan = asyncio.get_event_loop().time() - sikl_boshi
            uxlash = max(0.0, SIKL_INTERVAL_SEKUND - otgan)
            logger.debug("Sikl %.1fs davom etdi – %.1fs kutilmoqda", otgan, uxlash)
            await asyncio.sleep(uxlash)

    async def stop(self) -> None:
        """Botni muammosiz to'xtatish."""
        logger.info("Bot to'xtatilmoqda...")
        self._ishlamoqda = False

        if self.telegram:
            await self.telegram.send_message("🛑 <b>Oltin Trading Bot to'xtatildi.</b>")
            await self.telegram.stop_polling()

        if (
            self.order_manager
            and hasattr(self.order_manager, "_exchange")
            and self.order_manager._exchange
        ):
            try:
                await self.order_manager._exchange.close()
            except Exception:
                pass

        logger.info("Bot muvaffaqiyatli to'xtatildi.")


from bot.order_manager import OrderManager


def main() -> None:
    bot = GoldBot()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _togtatish(signum, frame):
        logger.info("Signal %s – bot to'xtatilmoqda", signum)
        loop.create_task(bot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _togtatish)
        except (OSError, ValueError):
            pass

    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        logger.info("Ctrl+C bosildi")
    finally:
        loop.run_until_complete(bot.stop())
        loop.close()
        logger.info("Event loop yopildi.")


if __name__ == "__main__":
    main()
