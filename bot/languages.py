"""Языки, доступные в /settings."""

from __future__ import annotations

# код: (название для меню, название для промпта)
LANGUAGES: dict[str, tuple[str, str]] = {
    "ru": ("🇷🇺 Русский", "Russian"),
    "en": ("🇬🇧 English", "English"),
    "uk": ("🇺🇦 Українська", "Ukrainian"),
    "be": ("🇧🇾 Беларуская", "Belarusian"),
    "kk": ("🇰🇿 Қазақша", "Kazakh"),
    "de": ("🇩🇪 Deutsch", "German"),
    "fr": ("🇫🇷 Français", "French"),
    "es": ("🇪🇸 Español", "Spanish"),
    "it": ("🇮🇹 Italiano", "Italian"),
    "pt": ("🇵🇹 Português", "Portuguese"),
    "pl": ("🇵🇱 Polski", "Polish"),
    "cs": ("🇨🇿 Čeština", "Czech"),
    "sr": ("🇷🇸 Srpski", "Serbian"),
    "tr": ("🇹🇷 Türkçe", "Turkish"),
    "he": ("🇮🇱 עברית", "Hebrew"),
    "zh": ("🇨🇳 中文", "Simplified Chinese"),
    "ja": ("🇯🇵 日本語", "Japanese"),
    "ko": ("🇰🇷 한국어", "Korean"),
}


def prompt_name(code: str) -> str:
    return LANGUAGES.get(code, (code, code))[1]


def menu_name(code: str) -> str:
    return LANGUAGES.get(code, (code, code))[0]
