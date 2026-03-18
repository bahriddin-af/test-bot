"""
Telegram xabar moduli.
Inline tugmalar, menyu tugmalar va barcha xabarlarni boshqaradi.
"""

from __future__ import annotations

import html
import io
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


def _e(text: str) -> str:
    return html.escape(str(text))


def _hozir() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


class TelegramBot:
    """Telegram Bot — inline tugmalar va menyu bilan."""

    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token
        self.chat_id = chat_id
        self._bot: Optional[Any] = None
        self._app: Optional[Any] = None
        self._faol: bool = bool(token and chat_id)

        # Kutayotgan signallar: {callback_data: signal_obj}
        self._kutayotgan_signallar: Dict[str, Any] = {}

        # Tashqi callback — main.py dan beriladi
        self.signal_tasdiqlandi_cb: Optional[Callable] = None
        self.signal_bekor_cb: Optional[Callable] = None
        self.holat_cb: Optional[Callable] = None
        self.balans_cb: Optional[Callable] = None
        self.togtat_cb: Optional[Callable] = None
        self.hisobot_cb: Optional[Callable] = None
        self.ochiq_tradelar_cb: Optional[Callable] = None
        self.trade_yopish_cb: Optional[Callable] = None

    async def _get_bot(self) -> Optional[Any]:
        if not self._faol:
            return None
        if self._bot is None:
            try:
                from telegram import Bot
                self._bot = Bot(token=self.token)
            except Exception as xato:
                logger.error("Telegram Bot yaratishda xato: %s", xato)
                self._faol = False
                return None
        return self._bot

    # ------------------------------------------------------------------
    # Ilovani ishga tushirish (polling)
    # ------------------------------------------------------------------

    async def start_polling(self) -> None:
        """Tugma callback larni tinglash uchun polling ni ishga tushirish."""
        if not self._faol:
            return
        try:
            from telegram.ext import (
                Application, CallbackQueryHandler, MessageHandler, filters
            )
            self._app = (
                Application.builder()
                .token(self.token)
                .build()
            )
            self._app.add_handler(CallbackQueryHandler(self._inline_tugma_handler))
            self._app.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, self._menyu_handler)
            )
            await self._app.initialize()
            await self._app.start()
            await self._app.updater.start_polling(drop_pending_updates=True)
            # Application ning o'z bot instanceini ishlatamiz
            self._bot = self._app.bot
            logger.info("Telegram polling ishga tushdi")
        except Exception as xato:
            logger.error("Polling ishga tushirishda xato: %s", xato)

    async def stop_polling(self) -> None:
        """Polling ni to'xtatish."""
        if self._app:
            try:
                await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Callback handlerlari
    # ------------------------------------------------------------------

    async def _inline_tugma_handler(self, update: Any, context: Any) -> None:
        """Inline tugma bosilganda ishlaydigan handler."""
        query = update.callback_query
        await query.answer()
        data = query.data

        if data.startswith("tasdiqlash:"):
            signal_id = data.replace("tasdiqlash:", "")
            signal = self._kutayotgan_signallar.pop(signal_id, None)
            if signal and self.signal_tasdiqlandi_cb:
                await query.edit_message_reply_markup(reply_markup=None)
                await query.message.reply_text("✅ <b>Signal tasdiqlandi — trade ochilmoqda...</b>", parse_mode="HTML")
                await self.signal_tasdiqlandi_cb(signal)
            else:
                await query.edit_message_text("⏰ Signal muddati o'tdi yoki allaqachon bajarildi.")

        elif data.startswith("bekor:"):
            signal_id = data.replace("bekor:", "")
            self._kutayotgan_signallar.pop(signal_id, None)
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text("❌ <b>Signal bekor qilindi.</b>", parse_mode="HTML")

        elif data.startswith("yopish:"):
            trade_id = int(data.replace("yopish:", ""))
            if self.trade_yopish_cb:
                await self.trade_yopish_cb(trade_id, query)
            else:
                await query.edit_message_text("⚠️ Yopish funksiyasi ulanmagan.")

    async def _menyu_handler(self, update: Any, context: Any) -> None:
        """Menyu tugmasi bosilganda ishlaydigan handler."""
        if str(update.effective_chat.id) != str(self.chat_id):
            return
        matn = update.message.text

        if matn == "📊 Holat":
            if self.holat_cb:
                await self.holat_cb(update, context)
            else:
                await update.message.reply_text("📊 Bot ishlayapti ✅", parse_mode="HTML")

        elif matn == "💰 Balans":
            if self.balans_cb:
                await self.balans_cb(update, context)

        elif matn == "🛑 To'xtat":
            await update.message.reply_text("🛑 <b>Bot to'xtatilmoqda...</b>", parse_mode="HTML")
            if self.togtat_cb:
                await self.togtat_cb(update, context)

        elif matn == "📈 Hisobot":
            if self.hisobot_cb:
                await self.hisobot_cb(update, context)

        elif matn == "📋 Ochiq tradelar":
            if self.ochiq_tradelar_cb:
                await self.ochiq_tradelar_cb(update, context)

        elif matn == "🔄 Qayta ishga tushir":
            await update.message.reply_text("🔄 <b>Bot davom etmoqda...</b>", parse_mode="HTML")

    # ------------------------------------------------------------------
    # Asosiy xabar yuborish
    # ------------------------------------------------------------------

    async def xabar_yuborish(
        self,
        matn: str,
        rasm: Optional[bytes] = None,
        tugmalar: Optional[Any] = None,
    ) -> Optional[Any]:
        """HTML formatlash va ixtiyoriy tugmalar bilan xabar yuborish."""
        if not self._faol:
            return None

        bot = await self._get_bot()
        if bot is None:
            return None

        try:
            if rasm:
                buf = io.BytesIO(rasm)
                buf.name = "chart.png"
                msg = await bot.send_photo(
                    chat_id=self.chat_id,
                    photo=buf,
                    caption=matn[:1024],
                    parse_mode="HTML",
                    reply_markup=tugmalar,
                )
            else:
                msg = await bot.send_message(
                    chat_id=self.chat_id,
                    text=matn,
                    parse_mode="HTML",
                    reply_markup=tugmalar,
                )
            return msg
        except Exception as xato:
            logger.error("Telegram xabar yuborishda xato: %s", xato)
            try:
                sof = (matn
                    .replace("<b>", "").replace("</b>", "")
                    .replace("<i>", "").replace("</i>", "")
                    .replace("<code>", "").replace("</code>", ""))
                if rasm:
                    buf = io.BytesIO(rasm)
                    buf.name = "chart.png"
                    return await bot.send_photo(chat_id=self.chat_id, photo=buf, caption=sof[:1024])
                else:
                    return await bot.send_message(chat_id=self.chat_id, text=sof)
            except Exception as xato2:
                logger.error("Qayta urinish ham muvaffaqiyatsiz: %s", xato2)
                return None

    async def send_message(self, text: str, photo: Optional[bytes] = None) -> bool:
        msg = await self.xabar_yuborish(text, photo)
        return msg is not None

    # ------------------------------------------------------------------
    # Menyu klaviaturasini yuborish
    # ------------------------------------------------------------------

    async def menyu_klaviatura_yuborish(self) -> None:
        """Pastki menyu tugmalarini o'rnatish."""
        if not self._faol:
            return
        try:
            from telegram import ReplyKeyboardMarkup, KeyboardButton
            klaviatura = ReplyKeyboardMarkup(
                [
                    [KeyboardButton("📊 Holat"), KeyboardButton("💰 Balans")],
                    [KeyboardButton("📋 Ochiq tradelar"), KeyboardButton("📈 Hisobot")],
                    [KeyboardButton("🔄 Qayta ishga tushir"), KeyboardButton("🛑 To'xtat")],
                ],
                resize_keyboard=True,
            )
            bot = await self._get_bot()
            if bot:
                await bot.send_message(
                    chat_id=self.chat_id,
                    text="⌨️ <b>Boshqaruv paneli tayyor ✅</b>",
                    parse_mode="HTML",
                    reply_markup=klaviatura,
                )
                logger.info("Menyu klaviatura yuborildi")
        except Exception as xato:
            logger.error("Menyu klaviatura yuborishda xato: %s", xato)

    # ------------------------------------------------------------------
    # Signal alert — inline tugmalar bilan
    # ------------------------------------------------------------------

    async def send_signal_alert(
        self,
        signal: Any,
        chart_bytes: Optional[bytes] = None,
        pnl_info: Optional[Dict] = None,
    ) -> None:
        """Signal kelganda inline Tasdiqlash/Bekor qilish tugmalari bilan yuborish."""
        try:
            from telegram import InlineKeyboardMarkup, InlineKeyboardButton
        except ImportError:
            await self._signal_tugmasiz(signal, chart_bytes)
            return

        yo_emoji = "📈" if str(signal.direction) == "BUY" else "📉"
        yo_nomi = "SOTIB OL" if str(signal.direction) == "BUY" else "SOT"
        kuch_nomi = {
            "WEAK": "ZAIF", "MEDIUM": "O'RTA",
            "STRONG": "KUCHLI", "VERY_STRONG": "JUDA KUCHLI",
        }.get(signal.strength.name, signal.strength.name)
        yulduz = "⭐" * int(signal.strength)
        sabablar = "\n".join(f"  • {_e(r)}" for r in signal.reasons[:8])

        # Foyda/zarar qismi
        pnl_qism = ""
        if pnl_info:
            lot    = pnl_info.get("lot", 0)
            sl_z   = pnl_info.get("sl_zarar", 0)
            tp1_f  = pnl_info.get("tp1_foyda", 0)
            tp2_f  = pnl_info.get("tp2_foyda", 0)
            tp3_f  = pnl_info.get("tp3_foyda", 0)
            rr     = pnl_info.get("rr", 0)
            pnl_qism = (
                f"\n📦 <b>Lot hajmi:</b> <code>{lot:.2f}</code>\n"
                f"⚖️ <b>Risk/Reward:</b> <code>1:{rr:.1f}</code>\n\n"
                f"💸 <b>SL ishlasa zarar:</b> <code>-${abs(sl_z):.2f}</code>\n"
                f"💵 <b>TP1 da foyda:</b> <code>+${tp1_f:.2f}</code>\n"
                f"💵 <b>TP2 da foyda:</b> <code>+${tp2_f:.2f}</code>\n"
                f"💵 <b>TP3 da foyda:</b> <code>+${tp3_f:.2f}</code>\n"
            )

        # Noyob signal ID
        signal_id = str(int(datetime.now(timezone.utc).timestamp()))
        self._kutayotgan_signallar[signal_id] = signal

        matn = (
            f"{yo_emoji} <b>OLTIN SIGNALI – {yo_nomi}</b>\n\n"
            f"💪 <b>Kuch:</b> {_e(kuch_nomi)} {yulduz}\n"
            f"✅ <b>Tasdiqlashlar:</b> {signal.confirmations}\n\n"
            f"💰 <b>Kirish:</b> <code>{signal.entry_price:.2f}</code>\n"
            f"🛑 <b>Stop Loss:</b> <code>{signal.sl_price:.2f}</code>\n"
            f"🎯 <b>Maqsad 1:</b> <code>{signal.tp1:.2f}</code>\n"
            f"🎯 <b>Maqsad 2:</b> <code>{signal.tp2:.2f}</code>\n"
            f"🎯 <b>Maqsad 3:</b> <code>{signal.tp3:.2f}</code>\n"
            f"{pnl_qism}\n"
            f"<b>Sabablar:</b>\n{sabablar}\n\n"
            f"🕐 <code>{_e(signal.timestamp.strftime('%Y-%m-%d %H:%M UTC'))}</code>\n\n"
            f"⚠️ <i>Trade ochish uchun tasdiqlang:</i>"
        )

        tugmalar = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Tasdiqlash", callback_data=f"tasdiqlash:{signal_id}"),
                InlineKeyboardButton("❌ Bekor qilish", callback_data=f"bekor:{signal_id}"),
            ]
        ])

        await self.xabar_yuborish(matn, rasm=chart_bytes, tugmalar=tugmalar)

    async def _signal_tugmasiz(self, signal: Any, chart_bytes: Optional[bytes] = None) -> None:
        """Tugmalar ishlamasa oddiy xabar yuborish."""
        yo_emoji = "📈" if str(signal.direction) == "BUY" else "📉"
        yo_nomi = "SOTIB OL" if str(signal.direction) == "BUY" else "SOT"
        sabablar = "\n".join(f"  • {_e(r)}" for r in signal.reasons[:8])
        matn = (
            f"{yo_emoji} <b>OLTIN SIGNALI – {yo_nomi}</b>\n\n"
            f"💰 <b>Kirish:</b> <code>{signal.entry_price:.2f}</code>\n"
            f"🛑 <b>Stop Loss:</b> <code>{signal.sl_price:.2f}</code>\n"
            f"🎯 <b>Maqsad 1:</b> <code>{signal.tp1:.2f}</code>\n\n"
            f"<b>Sabablar:</b>\n{sabablar}"
        )
        await self.xabar_yuborish(matn, rasm=chart_bytes)

    # ------------------------------------------------------------------
    # Trade xabarlari
    # ------------------------------------------------------------------

    async def send_trade_opened(self, trade: Any) -> None:
        rang = "🟢" if trade.direction == "BUY" else "🔴"
        yo_nomi = "SOTIB OLINDI" if trade.direction == "BUY" else "SOTILDI"
        matn = (
            f"{rang} <b>TRADE OCHILDI – #{trade.trade_id}</b>\n\n"
            f"📊 <b>Juftlik:</b> {_e(trade.symbol)}\n"
            f"📌 <b>Yo'nalish:</b> {yo_nomi}\n"
            f"💰 <b>Kirish:</b> <code>{trade.entry_price:.2f}</code>\n"
            f"🛑 <b>Stop Loss:</b> <code>{trade.sl_price:.2f}</code>\n"
            f"🎯 <b>Maqsad 1:</b> <code>{trade.tp1_price:.2f}</code>\n"
            f"🎯 <b>Maqsad 2:</b> <code>{trade.tp2_price:.2f}</code>\n"
            f"🎯 <b>Maqsad 3:</b> <code>{trade.tp3_price:.2f}</code>\n"
            f"📦 <b>Lot:</b> {trade.lot_size:.2f}\n"
            f"🕐 <code>{_e(trade.open_time.strftime('%Y-%m-%d %H:%M UTC'))}</code>"
        )
        await self.xabar_yuborish(matn)

    async def send_tp_hit(self, trade: Any, tp_daraja: str, foyda: float) -> None:
        foyda_emoji = "💚" if foyda >= 0 else "❤️"
        matn = (
            f"🎯 <b>MAQSAD {tp_daraja} BAJARILDI – #{trade.trade_id}</b>\n\n"
            f"📊 {_e(trade.symbol)} | {'SOTIB OLINDI' if trade.direction == 'BUY' else 'SOTILDI'}\n"
            f"💰 Kirish: <code>{trade.entry_price:.2f}</code>\n"
            f"{foyda_emoji} <b>Foyda:</b> <code>${foyda:+.2f}</code>\n"
            f"🕐 <code>{_hozir()}</code>"
        )
        await self.xabar_yuborish(matn)

    async def send_sl_hit(self, trade: Any, zarar: float) -> None:
        matn = (
            f"🛑 <b>STOP LOSS ISHLADI – #{trade.trade_id}</b>\n\n"
            f"📊 {_e(trade.symbol)} | {'SOTIB OLINDI' if trade.direction == 'BUY' else 'SOTILDI'}\n"
            f"💰 Kirish: <code>{trade.entry_price:.2f}</code>\n"
            f"🛑 SL: <code>{trade.sl_price:.2f}</code>\n"
            f"❤️ <b>Zarar:</b> <code>${zarar:+.2f}</code>\n"
            f"🕐 <code>{_hozir()}</code>"
        )
        await self.xabar_yuborish(matn)

    async def send_daily_report(self, statistika: Dict[str, Any], kapital_grafik: Optional[bytes] = None) -> None:
        sana = statistika.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        jami = statistika.get("total_trades", 0)
        yutish = statistika.get("wins", 0)
        yutqizish = statistika.get("losses", 0)
        yutish_foiz = statistika.get("win_rate", 0.0)
        jami_foyda = statistika.get("total_pnl", 0.0)
        max_dd = statistika.get("max_drawdown", 0.0)
        foyda_emoji = "💚" if jami_foyda >= 0 else "❤️"
        matn = (
            f"📋 <b>KUNLIK HISOBOT – {_e(str(sana))}</b>\n\n"
            f"📊 <b>Jami tradelar:</b> {jami}\n"
            f"✅ <b>Yutishlar:</b> {yutish}\n"
            f"❌ <b>Yutqizishlar:</b> {yutqizish}\n"
            f"📈 <b>Yutish foizi:</b> {yutish_foiz:.1f}%\n"
            f"{foyda_emoji} <b>Jami foyda/zarar:</b> <code>${jami_foyda:+.2f}</code>\n"
            f"📉 <b>Max drawdown:</b> <code>${max_dd:.2f}</code>\n\n"
            f"🤖 <i>Oltin Trading Bot</i>"
        )
        await self.xabar_yuborish(matn, rasm=kapital_grafik)

    async def send_error_alert(self, xato_matni: str) -> None:
        matn = (
            f"⚠️ <b>BOT XATOSI</b>\n\n"
            f"<code>{_e(xato_matni[:800])}</code>\n"
            f"🕐 <code>{_hozir()}</code>"
        )
        await self.xabar_yuborish(matn)

    async def send_heartbeat(self) -> None:
        matn = (
            f"💓 <b>Bot faoliyati</b>\n"
            f"Holat: Ishlamoqda ✅\n"
            f"🕐 <code>{_hozir()}</code>"
        )
        await self.xabar_yuborish(matn)
