"""
Yangiliklar filtri moduli.
Muhim iqtisodiy voqealar oldida trading ni to'xtatadi.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import aiohttp

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class YangilikVoqeasi:
    sarlavha: str
    voqea_vaqti: datetime
    valyuta: str
    ta_sir: str  # 'PAST', 'O\'RTA', 'YUQORI'

    def yuqori_tasirli_mi(self) -> bool:
        return self.ta_sir.upper() == "YUQORI"

    def __repr__(self) -> str:
        return (
            f"Yangilik({self.ta_sir} | {self.valyuta} | "
            f"{self.voqea_vaqti.strftime('%Y-%m-%d %H:%M')} | {self.sarlavha})"
        )


# Eski nom bilan ham ishlashi uchun
NewsEvent = YangilikVoqeasi


class NewsFilter:
    """
    Forex Factory iqtisodiy kalendarini yuklab oladi va
    muhim voqealar atrofida trading ni to'xtatish uchun
    yordamchi metodlar beradi.
    """

    FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    KESH_TTL_SEKUND = 3600  # 1 soat

    def __init__(self, config: Config) -> None:
        self.config = config
        self._kesh: List[YangilikVoqeasi] = []
        self._kesh_vaqti: Optional[datetime] = None

    async def fetch_news_calendar(self) -> List[YangilikVoqeasi]:
        """
        Shu haftaning iqtisodiy kalendarini Forex Factory dan olish.
        Natijalar 1 soat keshlanadi. Xato bo'lsa bo'sh ro'yxat qaytaradi.
        """
        if self._kesh_yaroqli_mi():
            return self._kesh

        voqealar: List[YangilikVoqeasi] = []
        try:
            async with aiohttp.ClientSession() as sessiya:
                sarlavhalar = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
                    )
                }
                async with sessiya.get(
                    self.FF_URL,
                    headers=sarlavhalar,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as javob:
                    if javob.status != 200:
                        logger.warning(
                            "Forex Factory HTTP %d qaytardi – bo'sh kalendar ishlatiladi",
                            javob.status,
                        )
                        return voqealar
                    malumot = await javob.json(content_type=None)

            for element in malumot:
                try:
                    voqea = self._ff_voqeasini_tahlil_qilish(element)
                    if voqea:
                        voqealar.append(voqea)
                except Exception as tahlil_xato:
                    logger.debug("Noto'g'ri voqea o'tkazib yuborildi: %s", tahlil_xato)

            logger.info("Forex Factory dan %d ta yangilik voqeasi olindi", len(voqealar))
        except asyncio.TimeoutError:
            logger.warning("Yangilik kalendari olish vaqt tugadi – trading davom etadi")
        except Exception as xato:
            logger.warning("Yangilik kalendari olishda xato: %s – trading davom etadi", xato)

        self._kesh = voqealar
        self._kesh_vaqti = datetime.now(timezone.utc)
        return voqealar

    @staticmethod
    def _ff_voqeasini_tahlil_qilish(element: dict) -> Optional[YangilikVoqeasi]:
        """Forex Factory JSON elementini YangilikVoqeasi ga aylantirish."""
        ta_sir_xom = str(element.get("impact", "")).upper()
        ta_sir_xarita = {
            "HIGH": "YUQORI",
            "MEDIUM": "O'RTA",
            "MED": "O'RTA",
            "LOW": "PAST",
            "HOLIDAY": "PAST",
            "NON-ECONOMIC": "PAST",
        }
        ta_sir = ta_sir_xarita.get(ta_sir_xom, "PAST")

        sarlavha = element.get("title", "Noma'lum voqea")
        valyuta = element.get("country", "USD").upper()

        sana_str = element.get("date", "")
        vaqt_str = element.get("time", "")

        try:
            if "all day" in vaqt_str.lower() or not vaqt_str:
                dt = datetime.strptime(sana_str, "%m-%d-%Y").replace(tzinfo=timezone.utc)
            else:
                dt_str = f"{sana_str} {vaqt_str}"
                for format_ in ("%m-%d-%Y %I:%M%p", "%m-%d-%Y %I%p"):
                    try:
                        dt = datetime.strptime(dt_str.upper(), format_).replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        continue
                else:
                    dt = datetime.strptime(sana_str, "%m-%d-%Y").replace(tzinfo=timezone.utc)
        except Exception:
            return None

        return YangilikVoqeasi(
            sarlavha=sarlavha,
            voqea_vaqti=dt,
            valyuta=valyuta,
            ta_sir=ta_sir,
        )

    def _kesh_yaroqli_mi(self) -> bool:
        if self._kesh_vaqti is None:
            return False
        yosh = (datetime.now(timezone.utc) - self._kesh_vaqti).total_seconds()
        return yosh < self.KESH_TTL_SEKUND

    def kelayotgan_voqealar(self, soat_oldin: float = 4.0) -> List[YangilikVoqeasi]:
        """Keyingi N soat ichidagi voqealarni qaytarish."""
        hozir = datetime.now(timezone.utc)
        chegara = hozir + timedelta(hours=soat_oldin)
        return [
            v for v in self._kesh
            if hozir <= v.voqea_vaqti.replace(tzinfo=timezone.utc) <= chegara
        ]

    # Eski nom bilan ham ishlashi uchun
    def get_upcoming_events(self, hours_ahead: float = 4.0) -> List[YangilikVoqeasi]:
        return self.kelayotgan_voqealar(hours_ahead)

    def yuqori_tasirli_yangilik_yaqinmi(
        self,
        soat_oldin: Optional[int] = None,
        soat_keyin: Optional[int] = None,
    ) -> bool:
        """
        USD yoki OLTIN ga tegishli yuqori ta'sirli yangilik
        yaqin vaqtda bo'lsa True qaytaradi – trading to'xtatiladi.
        """
        if not self.config.news_filter_enabled:
            return False

        sb = soat_oldin if soat_oldin is not None else self.config.news_filter_hours_before
        sk = soat_keyin if soat_keyin is not None else self.config.news_filter_hours_after

        hozir = datetime.now(timezone.utc)
        for voqea in self._kesh:
            if not voqea.yuqori_tasirli_mi():
                continue
            if voqea.valyuta not in ("USD", "XAU", "GOLD"):
                continue

            voqea_vaqti = voqea.voqea_vaqti.replace(tzinfo=timezone.utc)
            oyna_boshi = voqea_vaqti - timedelta(hours=sb)
            oyna_oxiri = voqea_vaqti + timedelta(hours=sk)

            if oyna_boshi <= hozir <= oyna_oxiri:
                logger.info(
                    "Yangilik bloklanishi faol: %s – %s da",
                    voqea.sarlavha,
                    voqea_vaqti.strftime("%H:%M UTC"),
                )
                return True
        return False

    # Eski nom bilan ham ishlashi uchun
    def is_high_impact_news_soon(self, hours_before=None, hours_after=None) -> bool:
        return self.yuqori_tasirli_yangilik_yaqinmi(hours_before, hours_after)

    def is_news_time(self, symbol: Optional[str] = None) -> bool:
        """Trading to'xtatilishi kerakmi – agar ha bo'lsa True qaytaradi."""
        return self.yuqori_tasirli_yangilik_yaqinmi()
