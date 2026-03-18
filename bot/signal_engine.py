"""
Signal yaratish moduli.
Ko'p vaqt oralig'idagi ma'lumotlarni tahlil qilib trade signallari chiqaradi.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from bot.indicators import (
    add_all_indicators,
    calculate_fibonacci_levels,
    calculate_pivot_points,
    detect_rsi_divergence,
)
from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class SignalStrength(IntEnum):
    WEAK = 1
    MEDIUM = 2
    STRONG = 3
    VERY_STRONG = 4


class SignalDirection:
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


@dataclass
class Signal:
    direction: str
    strength: SignalStrength
    confirmations: int
    reasons: List[str]
    entry_price: float
    sl_price: float
    tp1: float
    tp2: float
    tp3: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict:
        return {
            "direction": self.direction,
            "strength": self.strength.name,
            "confirmations": self.confirmations,
            "reasons": self.reasons,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "timestamp": self.timestamp.isoformat(),
        }


class SignalEngine:
    """
    Trend, momentum, support/resistance, volatillik va volume
    tahlillarini birlashtirib XAU/USD uchun signal yaratadi.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._cached_data: Dict[str, pd.DataFrame] = {}

    async def fetch_multi_timeframe_data(self) -> Dict[str, pd.DataFrame]:
        """1h, 4h va kunlik OHLCV ma'lumotlarini yfinance orqali olish."""
        symbol = self.config.yfinance_symbol
        natija: Dict[str, pd.DataFrame] = {}

        def _yuklab_olish(davr: str, interval: str) -> pd.DataFrame:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=davr, interval=interval)
            if df.empty:
                return df
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df.dropna(inplace=True)
            return df

        loop = asyncio.get_event_loop()
        try:
            df_1h = await loop.run_in_executor(None, _yuklab_olish, "5d", "60m")
            df_4h = await loop.run_in_executor(None, _yuklab_olish, "1mo", "1h")
            df_1d = await loop.run_in_executor(None, _yuklab_olish, "1y", "1d")
        except Exception as xato:
            logger.error("Ko'p vaqt oralig'idagi ma'lumot olishda xato: %s", xato)
            return {}

        if not df_1h.empty:
            natija["1h"] = add_all_indicators(df_1h)
        if not df_4h.empty:
            natija["4h"] = add_all_indicators(df_4h)
        if not df_1d.empty:
            natija["1D"] = add_all_indicators(df_1d)

        self._cached_data = natija
        return natija

    def check_trend_following(self, df: pd.DataFrame) -> Tuple[str, int, List[str]]:
        """EMA joylashuvi va Supertrend yo'nalishini tekshirish."""
        tasdiqlashlar = 0
        sabablar: List[str] = []
        yonalish = SignalDirection.NEUTRAL

        if df.empty or len(df) < 5:
            return yonalish, tasdiqlashlar, sabablar

        oxirgi = df.iloc[-1]

        if all(col in df.columns for col in ["ema_20", "ema_50", "ema_200"]):
            if oxirgi["ema_20"] > oxirgi["ema_50"] > oxirgi["ema_200"]:
                tasdiqlashlar += 1
                sabablar.append("EMA yuqoriga joylashgan (20>50>200) – o'sish tendensiyasi")
                yonalish = SignalDirection.BUY
            elif oxirgi["ema_20"] < oxirgi["ema_50"] < oxirgi["ema_200"]:
                tasdiqlashlar += 1
                sabablar.append("EMA pastga joylashgan (20<50<200) – tushish tendensiyasi")
                yonalish = SignalDirection.SELL

        if "ema_200" in df.columns:
            if oxirgi["close"] > oxirgi["ema_200"] and yonalish == SignalDirection.BUY:
                tasdiqlashlar += 1
                sabablar.append("Narx EMA 200 dan yuqorida")
            elif oxirgi["close"] < oxirgi["ema_200"] and yonalish == SignalDirection.SELL:
                tasdiqlashlar += 1
                sabablar.append("Narx EMA 200 dan pastda")

        if "supertrend_direction" in df.columns:
            st_dir = oxirgi["supertrend_direction"]
            if st_dir == 1 and yonalish in (SignalDirection.BUY, SignalDirection.NEUTRAL):
                tasdiqlashlar += 1
                sabablar.append("Supertrend – yuqoriga yo'nalgan")
                if yonalish == SignalDirection.NEUTRAL:
                    yonalish = SignalDirection.BUY
            elif st_dir == -1 and yonalish in (SignalDirection.SELL, SignalDirection.NEUTRAL):
                tasdiqlashlar += 1
                sabablar.append("Supertrend – pastga yo'nalgan")
                if yonalish == SignalDirection.NEUTRAL:
                    yonalish = SignalDirection.SELL

        return yonalish, tasdiqlashlar, sabablar

    def check_momentum(self, df: pd.DataFrame) -> Tuple[str, int, List[str]]:
        """RSI, MACD va Stochastic RSI momentum tasdiqlarini tekshirish."""
        tasdiqlashlar = 0
        sabablar: List[str] = []
        yonalish = SignalDirection.NEUTRAL

        if df.empty or len(df) < 3:
            return yonalish, tasdiqlashlar, sabablar

        oxirgi = df.iloc[-1]
        oldingi = df.iloc[-2]

        if "rsi" in df.columns:
            rsi = oxirgi["rsi"]
            if rsi > 50 and rsi < self.config.rsi_overbought:
                tasdiqlashlar += 1
                sabablar.append(f"RSI o'sish momentumi ({rsi:.1f})")
                yonalish = SignalDirection.BUY
            elif rsi < 50 and rsi > self.config.rsi_oversold:
                tasdiqlashlar += 1
                sabablar.append(f"RSI tushish momentumi ({rsi:.1f})")
                yonalish = SignalDirection.SELL

        divergence = detect_rsi_divergence(df)
        if divergence == "bullish":
            tasdiqlashlar += 1
            sabablar.append("RSI buqacha divergensiyasi aniqlandi")
            if yonalish == SignalDirection.NEUTRAL:
                yonalish = SignalDirection.BUY
        elif divergence == "bearish":
            tasdiqlashlar += 1
            sabablar.append("RSI ayiqcha divergensiyasi aniqlandi")
            if yonalish == SignalDirection.NEUTRAL:
                yonalish = SignalDirection.SELL

        if all(c in df.columns for c in ["macd", "macd_signal", "macd_hist"]):
            macd_yuqori = (
                oxirgi["macd"] > oxirgi["macd_signal"]
                and oldingi["macd"] <= oldingi["macd_signal"]
            )
            macd_quyi = (
                oxirgi["macd"] < oxirgi["macd_signal"]
                and oldingi["macd"] >= oldingi["macd_signal"]
            )
            if macd_yuqori and yonalish in (SignalDirection.BUY, SignalDirection.NEUTRAL):
                tasdiqlashlar += 1
                sabablar.append("MACD yuqoriga kesib o'tdi")
                if yonalish == SignalDirection.NEUTRAL:
                    yonalish = SignalDirection.BUY
            elif macd_quyi and yonalish in (SignalDirection.SELL, SignalDirection.NEUTRAL):
                tasdiqlashlar += 1
                sabablar.append("MACD pastga kesib o'tdi")
                if yonalish == SignalDirection.NEUTRAL:
                    yonalish = SignalDirection.SELL
            elif oxirgi["macd_hist"] > 0 and yonalish == SignalDirection.BUY:
                tasdiqlashlar += 1
                sabablar.append("MACD gistogramma musbat")
            elif oxirgi["macd_hist"] < 0 and yonalish == SignalDirection.SELL:
                tasdiqlashlar += 1
                sabablar.append("MACD gistogramma manfiy")

        if all(c in df.columns for c in ["stoch_rsi_k", "stoch_rsi_d"]):
            k = oxirgi["stoch_rsi_k"]
            d = oxirgi["stoch_rsi_d"]
            if k > d and k < 80 and yonalish == SignalDirection.BUY:
                tasdiqlashlar += 1
                sabablar.append(f"StochRSI o'sish signali (K={k:.1f}>D={d:.1f})")
            elif k < d and k > 20 and yonalish == SignalDirection.SELL:
                tasdiqlashlar += 1
                sabablar.append(f"StochRSI tushish signali (K={k:.1f}<D={d:.1f})")

        return yonalish, tasdiqlashlar, sabablar

    def check_support_resistance(self, df: pd.DataFrame) -> Tuple[str, int, List[str]]:
        """Pivot nuqtalari va Fibonacci darajalarini tekshirish."""
        tasdiqlashlar = 0
        sabablar: List[str] = []
        yonalish = SignalDirection.NEUTRAL

        if df.empty or len(df) < 10:
            return yonalish, tasdiqlashlar, sabablar

        oxirgi = df.iloc[-1]
        close = float(oxirgi["close"])
        atr = float(oxirgi.get("atr", close * 0.005))
        chegara = atr * 0.5

        pivotlar = calculate_pivot_points(df)
        if pivotlar:
            pivot = pivotlar["pivot"]
            s1, s2 = pivotlar["s1"], pivotlar["s2"]
            r1, r2 = pivotlar["r1"], pivotlar["r2"]

            tayanchga_yaqin = any(abs(close - lvl) < chegara for lvl in [s1, s2, pivot])
            qarshilikka_yaqin = any(abs(close - lvl) < chegara for lvl in [r1, r2])

            if tayanchga_yaqin:
                tasdiqlashlar += 1
                sabablar.append("Narx pivot tayanch darajasiga yaqin")
                yonalish = SignalDirection.BUY
            elif qarshilikka_yaqin:
                tasdiqlashlar += 1
                sabablar.append("Narx pivot qarshilik darajasiga yaqin")
                yonalish = SignalDirection.SELL

        orqaga_qarash = min(50, len(df))
        swing_high = float(df["high"].tail(orqaga_qarash).max())
        swing_low = float(df["low"].tail(orqaga_qarash).min())
        fiblar = calculate_fibonacci_levels(swing_high, swing_low)

        muhim_fiblar = ["0.382", "0.500", "0.618"]
        for fib_kalit in muhim_fiblar:
            fib_daraja = fiblar.get(fib_kalit, 0.0)
            if abs(close - fib_daraja) < chegara:
                trend_yonalish, _, _ = self.check_trend_following(df)
                if trend_yonalish == SignalDirection.BUY:
                    tasdiqlashlar += 1
                    sabablar.append(f"Narx Fibonacci {fib_kalit} tayanch darajasida ({fib_daraja:.2f})")
                    yonalish = SignalDirection.BUY
                elif trend_yonalish == SignalDirection.SELL:
                    tasdiqlashlar += 1
                    sabablar.append(f"Narx Fibonacci {fib_kalit} qarshilik darajasida ({fib_daraja:.2f})")
                    yonalish = SignalDirection.SELL
                break

        return yonalish, tasdiqlashlar, sabablar

    def check_volatility(self, df: pd.DataFrame) -> Tuple[str, int, List[str]]:
        """ATR va Bollinger Band volatilligini tekshirish."""
        tasdiqlashlar = 0
        sabablar: List[str] = []
        yonalish = SignalDirection.NEUTRAL

        if df.empty or len(df) < 5:
            return yonalish, tasdiqlashlar, sabablar

        oxirgi = df.iloc[-1]
        close = float(oxirgi["close"])

        if all(c in df.columns for c in ["bb_upper", "bb_lower", "bb_middle", "bb_width"]):
            bb_kenglik = oxirgi["bb_width"]
            ort_kenglik = float(df["bb_width"].tail(20).mean())
            bb_quyi = oxirgi["bb_lower"]
            bb_yuqori = oxirgi["bb_upper"]

            if close <= bb_quyi and bb_kenglik > ort_kenglik:
                tasdiqlashlar += 1
                sabablar.append("BB quyi chiziqqa tegdi – sotib olish imkoniyati")
                yonalish = SignalDirection.BUY
            elif close >= bb_yuqori and bb_kenglik > ort_kenglik:
                tasdiqlashlar += 1
                sabablar.append("BB yuqori chiziqqa tegdi – sotish imkoniyati")
                yonalish = SignalDirection.SELL
            elif bb_kenglik < ort_kenglik * 0.7:
                sabablar.append("BB siqilmoqda – tez orada kuchli harakat kutilmoqda")

        if "adx" in df.columns:
            adx_qiymati = float(oxirgi["adx"])
            if adx_qiymati >= self.config.adx_threshold:
                tasdiqlashlar += 1
                sabablar.append(f"ADX {adx_qiymati:.1f} – kuchli trend mavjud")
            else:
                sabablar.append(f"ADX {adx_qiymati:.1f} – trend zaif, bozor sideways")

        return yonalish, tasdiqlashlar, sabablar

    def check_volume(self, df: pd.DataFrame) -> Tuple[str, int, List[str]]:
        """OBV va nisbiy hajmni tekshirish."""
        tasdiqlashlar = 0
        sabablar: List[str] = []
        yonalish = SignalDirection.NEUTRAL

        if df.empty or len(df) < 10:
            return yonalish, tasdiqlashlar, sabablar

        oxirgi = df.iloc[-1]

        ort_hajm = float(df["volume"].tail(20).mean())
        joriy_hajm = float(oxirgi["volume"])
        if ort_hajm > 0:
            nisbiy_hajm = joriy_hajm / ort_hajm
            if nisbiy_hajm > 1.5:
                tasdiqlashlar += 1
                sabablar.append(f"Hajm yuqori – o'rtachadan {nisbiy_hajm:.1f}x ko'p")

        if "obv" in df.columns and len(df) >= 5:
            obv_seriya = df["obv"].tail(5)
            obv_qiyaligi = float(obv_seriya.iloc[-1]) - float(obv_seriya.iloc[0])
            narx_qiyaligi = float(df["close"].iloc[-1]) - float(df["close"].iloc[-5])

            if obv_qiyaligi > 0 and narx_qiyaligi > 0:
                tasdiqlashlar += 1
                sabablar.append("OBV narx o'sishini tasdiqlaydi")
                yonalish = SignalDirection.BUY
            elif obv_qiyaligi < 0 and narx_qiyaligi < 0:
                tasdiqlashlar += 1
                sabablar.append("OBV narx tushishini tasdiqlaydi")
                yonalish = SignalDirection.SELL

        return yonalish, tasdiqlashlar, sabablar

    def check_higher_timeframe_trend(self) -> str:
        """Kunlik taymfreymdan asosiy trend yo'nalishini aniqlash."""
        df_1d = self._cached_data.get("1D", pd.DataFrame())
        if df_1d.empty or len(df_1d) < 5:
            return SignalDirection.NEUTRAL

        oxirgi = df_1d.iloc[-1]
        if "ema_50" in df_1d.columns and "ema_200" in df_1d.columns:
            if oxirgi["close"] > oxirgi["ema_50"] > oxirgi["ema_200"]:
                return SignalDirection.BUY
            if oxirgi["close"] < oxirgi["ema_50"] < oxirgi["ema_200"]:
                return SignalDirection.SELL
        return SignalDirection.NEUTRAL

    def calculate_signal_strength(self, jami_tasdiqlashlar: int) -> SignalStrength:
        """Tasdiqlar soniga qarab signal kuchini hisoblash."""
        if jami_tasdiqlashlar >= 7:
            return SignalStrength.VERY_STRONG
        if jami_tasdiqlashlar >= 5:
            return SignalStrength.STRONG
        if jami_tasdiqlashlar >= 3:
            return SignalStrength.MEDIUM
        return SignalStrength.WEAK

    async def generate_signal(self) -> Optional[Signal]:
        """
        Barcha tekshiruvlarni bajarib, signal yaratish.
        Minimum tasdiqlashlar soni yetarli bo'lmasa None qaytaradi.
        """
        malumot = await self.fetch_multi_timeframe_data()
        if not malumot:
            logger.warning("Ma'lumot yo'q – signal yaratish o'tkazib yuborildi")
            return None

        df_1h = malumot.get("1h", pd.DataFrame())
        if df_1h.empty:
            logger.warning("1h ma'lumoti bo'sh – signal yaratib bo'lmadi")
            return None

        yuqori_tf_trend = self.check_higher_timeframe_trend()
        barcha_sabablar: List[str] = []
        buy_ovozi = 0
        sell_ovozi = 0
        jami_tasdiqlashlar = 0

        tekshiruvlar = [
            self.check_trend_following(df_1h),
            self.check_momentum(df_1h),
            self.check_support_resistance(df_1h),
            self.check_volatility(df_1h),
            self.check_volume(df_1h),
        ]

        for yonalish_, tasdiqlar, sabablar in tekshiruvlar:
            barcha_sabablar.extend(sabablar)
            jami_tasdiqlashlar += tasdiqlar
            if yonalish_ == SignalDirection.BUY:
                buy_ovozi += tasdiqlar
            elif yonalish_ == SignalDirection.SELL:
                sell_ovozi += tasdiqlar

        if yuqori_tf_trend == SignalDirection.BUY:
            barcha_sabablar.append("Kunlik trend yuqoriga (HTF moslashgan)")
            buy_ovozi += 1
            jami_tasdiqlashlar += 1
        elif yuqori_tf_trend == SignalDirection.SELL:
            barcha_sabablar.append("Kunlik trend pastga (HTF moslashgan)")
            sell_ovozi += 1
            jami_tasdiqlashlar += 1

        if buy_ovozi > sell_ovozi:
            yakuniy_yonalish = SignalDirection.BUY
        elif sell_ovozi > buy_ovozi:
            yakuniy_yonalish = SignalDirection.SELL
        else:
            logger.info("Signal neytral – ovozlar teng BUY=%d SELL=%d", buy_ovozi, sell_ovozi)
            return None

        if jami_tasdiqlashlar < self.config.min_confirmations:
            logger.info(
                "Signal juda zaif: %d tasdiqlash (minimum %d kerak)",
                jami_tasdiqlashlar,
                self.config.min_confirmations,
            )
            return None

        oxirgi = df_1h.iloc[-1]
        kirish_narxi = float(oxirgi["close"])
        atr = float(oxirgi.get("atr", kirish_narxi * 0.005))
        sl_masofa = atr * self.config.atr_multiplier_sl

        if yakuniy_yonalish == SignalDirection.BUY:
            sl_narxi = kirish_narxi - sl_masofa
        else:
            sl_narxi = kirish_narxi + sl_masofa

        tp_masofa = sl_masofa * self.config.rr_ratio_min
        if yakuniy_yonalish == SignalDirection.BUY:
            tp1 = kirish_narxi + tp_masofa
            tp2 = kirish_narxi + tp_masofa * 1.5
            tp3 = kirish_narxi + tp_masofa * 2.5
        else:
            tp1 = kirish_narxi - tp_masofa
            tp2 = kirish_narxi - tp_masofa * 1.5
            tp3 = kirish_narxi - tp_masofa * 2.5

        kuch = self.calculate_signal_strength(jami_tasdiqlashlar)

        signal = Signal(
            direction=yakuniy_yonalish,
            strength=kuch,
            confirmations=jami_tasdiqlashlar,
            reasons=barcha_sabablar,
            entry_price=round(kirish_narxi, 3),
            sl_price=round(sl_narxi, 3),
            tp1=round(tp1, 3),
            tp2=round(tp2, 3),
            tp3=round(tp3, 3),
        )

        kuch_nomi = {
            "WEAK": "ZAIF",
            "MEDIUM": "O'RTA",
            "STRONG": "KUCHLI",
            "VERY_STRONG": "JUDA KUCHLI",
        }.get(kuch.name, kuch.name)

        logger.info(
            "Signal yaratildi: %s | Kuch: %s | Tasdiqlashlar: %d",
            yakuniy_yonalish,
            kuch_nomi,
            jami_tasdiqlashlar,
        )
        return signal
