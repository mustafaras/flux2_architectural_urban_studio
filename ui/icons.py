"""Emoji icon helpers for Streamlit UI."""

from __future__ import annotations

from html import escape

import streamlit as st


_EMOJI_MAP: dict[str, str] = {
    "image": "🖼️",
    "settings": "⚙️",
    "wand": "🪄",
    "edit": "✏️",
    "history": "🕘",
    "book": "📚",
    "sparkles": "✨",
    "activity": "📈",
    "bolt": "⚡",
    "queue": "⏳",
    "play": "▶️",
    "pause": "⏸️",
    "resume": "⏯️",
    "boost": "↥",
    "trash": "🗑️",
    "check": "✅",
    "cross": "❌",
    "cancel": "⊘",
    "clock": "🕐",
    "zip": "📦",
    "template": "📋",
    "download": "⬇️",
    "refresh": "🔄",
    "upload": "📤",
    "phone": "📱",
    "cloud": "☁️",
    "folder": "📁",
    "link": "🔗",
    "plus": "➕",
    "target": "🎯",
    "fire": "🔥",
    "report": "📄",
    "shield": "🛡️",
    "warning": "⚠️",
    "robot": "🤖",
    "rocket": "🚀",
    "note": "📝",
    "save": "💾",
    "lab": "🔬",
    "recycle": "♻️",
    "scales": "⚖️",
    "violation": "🚨",
    "film": "📽️",
    "idea": "💡",
}


def _emoji(icon: str) -> str:
    if icon in _EMOJI_MAP:
        return _EMOJI_MAP[icon]
    # If already an emoji/text icon (e.g. "⏳"), preserve it.
    if any(ord(ch) > 127 for ch in icon):
        return icon
    return _EMOJI_MAP["sparkles"]


def heading(text: str, icon: str = "sparkles", level: int = 2) -> None:
    safe_text = escape(text)
    emoji = _emoji(icon)
    prefix = "##" if level == 2 else "###"
    st.markdown(f"{prefix} {emoji} {safe_text}")


def title(text: str, icon: str = "sparkles") -> None:
    safe_text = escape(text)
    emoji = _emoji(icon)
    st.markdown(f"# {emoji} {safe_text}")


def page_intro(text: str, description: str, icon: str = "sparkles", level: int = 2) -> None:
    """Render a standardized page heading with one-line description."""
    heading(text, icon=icon, level=level)
    st.caption(description)


def tab(text: str, icon: str = "sparkles") -> str:
    """Return a standardized emoji-prefixed tab label."""
    safe_text = escape(text)
    emoji = _emoji(icon)
    return f"{emoji} {safe_text}"


def label(text: str, icon: str = "sparkles") -> str:
    """Return a standardized emoji-prefixed label for buttons/messages."""
    emoji = _emoji(icon)
    return f"{emoji} {text}"
