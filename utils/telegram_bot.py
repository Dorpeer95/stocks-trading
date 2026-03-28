"""
Telegram bot — sends alerts and formatted messages to the user.

Handles message formatting, splitting (>4096 chars), retry logic,
and all alert templates for the stocks-agent.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_MESSAGE_LENGTH = 4096

# Emoji conventions
EMOJI = {
    "bullish": "🟢",
    "bearish": "🔴",
    "neutral": "🟡",
    "action": "⚠️",
    "chart": "📊",
    "money": "💰",
    "target": "🎯",
    "stop": "🛑",
    "morning": "☀️",
    "evening": "🌙",
    "rocket": "🚀",
    "fire": "🔥",
    "warning": "⚠️",
    "check": "✅",
    "cross": "❌",
    "clock": "🕐",
    "bot_start": "🟢",
    "bot_stop": "🔴",
}

# ---------------------------------------------------------------------------
# Bot singleton (Using standard HTTP)
# ---------------------------------------------------------------------------

_chat_id: Optional[str] = None
_token: Optional[str] = None

def init_bot() -> None:
    """Initialise and validate the Telegram credentials.

    Reads ``STOCKS_TELEGRAM_TOKEN`` and ``STOCKS_TELEGRAM_CHAT_ID``
    from the environment.

    Raises
    ------
    ValueError
        If required environment variables are missing.
    """
    global _token, _chat_id

    token = os.getenv("STOCKS_TELEGRAM_TOKEN")
    chat_id = os.getenv("STOCKS_TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise ValueError(
            "STOCKS_TELEGRAM_TOKEN and STOCKS_TELEGRAM_CHAT_ID must be set"
        )

    _token = token
    _chat_id = chat_id
    logger.info("Telegram configuration validated")


def _get_chat_id() -> str:
    if _chat_id is None:
        init_bot()
    return _chat_id  # type: ignore


# ---------------------------------------------------------------------------
# Core send function
# ---------------------------------------------------------------------------

def _split_message(text: str) -> List[str]:
    """Split a long message into chunks ≤ MAX_MESSAGE_LENGTH.

    Tries to split at newlines to avoid breaking sentences.
    """
    if len(text) <= MAX_MESSAGE_LENGTH:
        return [text]

    chunks: List[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= MAX_MESSAGE_LENGTH:
            chunks.append(remaining)
            break

        # Find the last newline within the limit
        split_at = remaining[:MAX_MESSAGE_LENGTH].rfind("\n")
        if split_at == -1 or split_at < MAX_MESSAGE_LENGTH // 2:
            # No good newline; force split at limit
            split_at = MAX_MESSAGE_LENGTH

        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")

    return chunks


def send_message(text: str, parse_mode: Optional[str] = None) -> bool:
    """Send a Telegram message (synchronous HTTP dispatch).

    Parameters
    ----------
    text : Message text.  Will be split if >4096 chars.
    parse_mode : ``'HTML'``, ``'Markdown'``, or ``None`` for plain text.

    Returns
    -------
    ``True`` on success, ``False`` on failure.
    """
    import httpx
    import time
    
    token = os.getenv("STOCKS_TELEGRAM_TOKEN")
    chat_id = os.getenv("STOCKS_TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        logger.warning("Telegram token or chat ID missing.")
        return False

    chunks = _split_message(text)

    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            chunk = f"({i + 1}/{len(chunks)})\n{chunk}"

        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "disable_web_page_preview": True
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        for attempt in range(1, 4):  # 3 retries
            try:
                resp = httpx.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json=payload,
                    timeout=10.0
                )
                
                if resp.status_code == 200:
                    break
                elif resp.status_code == 429:
                    retry_after = int(resp.json().get("parameters", {}).get("retry_after", 3))
                    logger.warning(f"Telegram rate limited, waiting {retry_after}s")
                    time.sleep(retry_after)
                else:
                    if attempt < 3:
                        logger.warning(f"Telegram send attempt {attempt}/3 failed: {resp.text}")
                        time.sleep(2 ** attempt)
                    else:
                        logger.error(f"Telegram send failed after 3 attempts: {resp.text}")
                        return False

            except Exception as e:
                if attempt < 3:
                    logger.warning(f"Telegram send attempt {attempt}/3 failed: {e}")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Telegram send failed after 3 attempts: {e}")
                    return False

    return True


# ---------------------------------------------------------------------------
# Alert convenience function
# ---------------------------------------------------------------------------

def send_alert(alert_type: str, data: Dict[str, Any]) -> bool:
    """Send a pre-formatted alert based on type.

    Parameters
    ----------
    alert_type : One of ``'opportunity'``, ``'position_update'``,
                 ``'weekly_summary'``, ``'morning_briefing'``,
                 ``'eod_summary'``, ``'action_needed'``,
                 ``'bot_start'``, ``'bot_stop'``.
    data : Context dict for formatting.

    Returns
    -------
    ``True`` on success, ``False`` on failure.
    """
    formatters = {
        "opportunity": lambda d: format_opportunity(d),
        "position_update": lambda d: format_position_update(d),
        "portfolio_update": lambda d: format_portfolio_update(
            d.get("portfolio_diff", {}),
            d.get("market_mood", "Neutral"),
            d.get("hot_sectors", []),
            d.get("regime"),
            gpt_briefing=d.get("gpt_briefing"),
            cross_asset_regime=d.get("cross_asset_regime"),
            safety=d.get("safety"),
            combined_size_mod=d.get("combined_size_mod"),
        ),
        # legacy — kept for backward compat
        "weekly_summary": lambda d: format_weekly_summary(
            d.get("opportunities", []),
            d.get("market_mood", "Neutral"),
            d.get("hot_sectors", []),
            gpt_briefing=d.get("gpt_briefing"),
        ),
        "morning_briefing": lambda d: format_morning_briefing(
            d.get("events", []),
            d.get("positions", []),
            d.get("macro", {}),
            enter_now=d.get("enter_now", []),
            watchlist=d.get("watchlist", []),
            safety=d.get("safety"),
        ),
        "eod_summary": lambda d: format_eod_summary(d),
        "action_needed": lambda d: format_action_needed(d),
        "entry_signal": lambda d: format_entry_signal(d),
        "bot_start": lambda d: format_bot_start(d),
        "bot_stop": lambda d: f"{EMOJI['bot_stop']} Bot stopped.",
    }

    formatter = formatters.get(alert_type)
    if formatter is None:
        logger.warning(f"Unknown alert type: {alert_type}")
        return send_message(f"[{alert_type}] {data}")

    text = formatter(data)
    return send_message(text)


# ---------------------------------------------------------------------------
# Message formatters
# ---------------------------------------------------------------------------

def format_opportunity(opp: Dict[str, Any]) -> str:
    """Format a single trade opportunity for Telegram."""
    ticker = opp.get("ticker", "???")
    confidence = opp.get("confidence", 0)
    entry_low = opp.get("entry_price_low", 0)
    entry_high = opp.get("entry_price_high", 0)
    stop = opp.get("stop_loss", 0)
    target = opp.get("target_price", 0)
    size = opp.get("position_size_usd", 0)
    risk = opp.get("risk_usd", 0)
    reward = opp.get("reward_usd", 0)
    reasons = opp.get("reasons", [])
    setup = opp.get("setup_type", "")

    mood = EMOJI["bullish"] if confidence >= 70 else EMOJI["neutral"]

    lines = [
        f"{mood} {ticker} — {confidence}/100 confidence",
        f"",
        f"  Buy: ${size:,.0f} worth at ${entry_low:.2f}-${entry_high:.2f}",
        f"  {EMOJI['stop']} Stop: ${stop:.2f} (max loss ${risk:,.0f})",
        f"  {EMOJI['target']} Target: ${target:.2f} (gain ${reward:,.0f})",
    ]

    if setup:
        lines.append(f"  Setup: {setup}")

    if reasons:
        lines.append(f"  Why: {' + '.join(reasons)}")

    return "\n".join(lines)


def format_position_update(pos: Dict[str, Any]) -> str:
    """Format a position status update for Telegram."""
    ticker = pos.get("ticker", "???")
    pnl = pos.get("unrealized_pnl", 0)
    pnl_pct = pos.get("unrealized_pnl_pct", 0)
    current = pos.get("current_price", 0)
    target = pos.get("target_price", 0)
    stop = pos.get("stop_loss", 0)
    days = pos.get("days_held", 0)

    emoji = EMOJI["bullish"] if pnl >= 0 else EMOJI["bearish"]
    sign = "+" if pnl >= 0 else ""

    lines = [
        f"{emoji} {ticker} — Day {days}",
        f"  Price: ${current:.2f}",
        f"  P&L: {sign}${pnl:,.2f} ({sign}{pnl_pct:.1f}%)",
        f"  {EMOJI['target']} Target: ${target:.2f}",
        f"  {EMOJI['stop']} Stop: ${stop:.2f}",
    ]

    return "\n".join(lines)


def format_portfolio_update(
    diff: Dict[str, Any],
    market_mood: str = "Neutral",
    hot_sectors: Optional[List[str]] = None,
    regime: Optional[Dict[str, Any]] = None,
    gpt_briefing: Optional[str] = None,
    cross_asset_regime: Optional[Dict[str, Any]] = None,
    safety: Optional[Dict[str, Any]] = None,
    combined_size_mod: Optional[float] = None,
) -> str:
    """Format the weekly portfolio state-machine update.

    Replaces the old 'fresh picks' weekly summary with a clear
    HOLD / WATCH / BUY / SELL / SWAP diff view.
    """
    from datetime import date as dt_date

    mood_emoji = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}.get(
        market_mood, "🟡"
    )
    vix_str = ""
    if regime:
        vix = regime.get("vix", {})
        if isinstance(vix, dict) and vix.get("current"):
            vix_str = f"  |  VIX {vix['current']:.1f}"

    lines = [
        f"📊 PORTFOLIO — {dt_date.today().strftime('%b %d, %Y')}",
        f"Market: {market_mood} {mood_emoji}{vix_str}",
        "",
    ]

    if hot_sectors:
        lines.append(f"Leading sectors: {', '.join(hot_sectors[:3])}")
        lines.append("")

    # Cross-asset regime + breadth gate context
    if cross_asset_regime:
        cr = cross_asset_regime.get("regime", "")
        cr_score = cross_asset_regime.get("score", 0)
        cr_mod = cross_asset_regime.get("position_size_modifier", 1.0)
        regime_emojis = {
            "broadening": "🟢", "concentration": "🟡", "transitional": "🟡",
            "inflationary": "🟠", "contraction": "🔴",
        }
        re = regime_emojis.get(cr, "⚪")
        lines.append(f"Regime: {re} {cr.title()} (score {cr_score:.0f}  size×{cr_mod:.2f})")

    if safety:
        breadth_pct = safety.get("breadth", {}).get("pct_above_200", 0)
        top_risk = safety.get("top", {}).get("top_risk", "low")
        gate_ok = safety.get("safe_to_enter", True)
        gate_str = "✅ Gate open" if gate_ok else "🔴 Gate CLOSED"
        dist_str = f"{safety.get('top', {}).get('distribution_days', 0)}d dist"
        lines.append(
            f"Breadth: {breadth_pct:.0f}% above 200MA  |  {dist_str}  |  {gate_str}"
        )

    if combined_size_mod is not None and combined_size_mod < 1.0:
        lines.append(f"Position sizing: {combined_size_mod:.0%} of normal (regime+breadth penalty)")

    if cross_asset_regime or safety:
        lines.append("")

    if gpt_briefing:
        lines.append("━━━ AI Briefing ━━━")
        lines.append(gpt_briefing.strip())
        lines.append("━━━━━━━━━━━━━━━━━━")
        lines.append("")

    holdings = diff.get("holdings", [])
    new_entries = diff.get("new_entries", [])
    exits = diff.get("exits", [])
    watch_cleared = diff.get("watch_cleared", [])
    displacements = diff.get("displacements", [])
    open_slots = diff.get("open_slots", 0)

    entry_tickers = {c.get("ticker") for c in new_entries}

    # ── HOLDING STRONG ────────────────────────────────────────────────────
    strong_holds = [
        h for h in holdings
        if h["status"] == "active" and h["ticker"] not in entry_tickers
    ]
    if strong_holds:
        lines.append(f"✅ HOLDING STRONG ({len(strong_holds)})")
        for h in strong_holds:
            ticker = h["ticker"]
            conf = h.get("current_confidence") or 0
            prev = h.get("prev_confidence") or conf
            weeks = h.get("weeks_held") or 0
            arrow = "↑" if conf >= prev else "→"
            conf_delta = f"{conf:.0f}{arrow}"
            lines.append(f"  {ticker:<6} conf {conf_delta}  wk{weeks}")
        lines.append("")

    # ── WATCH — WEAKENING ─────────────────────────────────────────────────
    watch_holds = [h for h in holdings if h["status"] == "watch"]
    if watch_holds:
        lines.append("⚠️ WATCH — signal weakening")
        for h in watch_holds:
            ticker = h["ticker"]
            conf = h.get("current_confidence") or 0
            prev = h.get("prev_confidence") or conf
            weak_weeks = h.get("consecutive_weak_weeks") or 1
            lines.append(
                f"  {ticker:<6} conf {conf:.0f}↓ was {prev:.0f}  "
                f"({weak_weeks}/2 weak weeks)"
            )
        lines.append("")

    # ── WATCH CLEARED ─────────────────────────────────────────────────────
    if watch_cleared:
        lines.append("💪 SIGNAL RECOVERED")
        for ticker in watch_cleared:
            lines.append(f"  {ticker} — back to active")
        lines.append("")

    # ── NEW ENTRIES ───────────────────────────────────────────────────────
    if new_entries:
        lines.append(f"🟢 NEW ENTRY — add to portfolio")
        for c in new_entries:
            ticker = c.get("ticker", "???")
            conf = c.get("confidence", 0)
            entry_low = c.get("entry_price_low", 0)
            entry_high = c.get("entry_price_high", 0)
            stop = c.get("stop_loss", 0)
            target = c.get("target_price", 0)
            setup = c.get("setup_type", "")
            lines.append(
                f"  {ticker}  conf {conf:.0f}  |  "
                f"entry ${entry_low:.2f}–${entry_high:.2f}  "
                f"stop ${stop:.2f}  tgt ${target:.2f}"
            )
            if setup:
                lines.append(f"    Setup: {setup}")
        lines.append("")

    # ── EXITS ─────────────────────────────────────────────────────────────
    if exits:
        lines.append("🔴 EXIT — sell these positions")
        for ticker in exits:
            # Find exit holding detail if still in diff
            exiting = next(
                (h for h in holdings if h["ticker"] == ticker), None
            )
            reason = exiting.get("notes", "Signal broken") if exiting else "Signal broken"
            lines.append(f"  {ticker}  — {reason}")
        lines.append("")

    # ── DISPLACEMENT SUGGESTION ───────────────────────────────────────────
    if displacements:
        lines.append("🔄 CONSIDER SWAP (optional)")
        for d in displacements:
            lines.append(
                f"  Sell {d['weak_ticker']} (conf {d['weak_conf']:.0f})  →  "
                f"Buy {d['new_ticker']} (conf {d['new_conf']:.0f}, "
                f"+{d['gap']:.0f} pts)"
            )
        lines.append("")

    # ── PORTFOLIO SLOTS ───────────────────────────────────────────────────
    total_active = len([h for h in holdings if h["status"] in ("active", "watch")])
    max_slots = total_active + open_slots
    lines.append(f"📭 Slots: {total_active}/{max_slots} filled")

    if not strong_holds and not watch_holds and not new_entries:
        lines.append("")
        lines.append("No high-conviction setups this week. Stay patient.")

    return "\n".join(lines)


def format_weekly_summary(
    opportunities: List[Dict[str, Any]],
    market_mood: str,
    hot_sectors: Optional[List[str]] = None,
    gpt_briefing: Optional[str] = None,
) -> str:
    """Format the weekly summary with top picks.

    Top 3 are flagged as priority plays. Rest go on the watch list.
    """
    mood_emoji = {
        "Bullish": EMOJI["bullish"],
        "Bearish": EMOJI["bearish"],
        "Neutral": EMOJI["neutral"],
    }.get(market_mood, EMOJI["neutral"])

    lines = [
        f"{EMOJI['chart']} WEEKLY SWING TRADING PLAYBOOK",
        f"",
        f"Market: {market_mood} {mood_emoji}",
    ]

    if hot_sectors:
        lines.append(f"Hot sectors: {', '.join(hot_sectors[:3])}")

    lines.append("")

    if gpt_briefing:
        lines.append("━━━ AI Briefing ━━━")
        lines.append(gpt_briefing.strip())
        lines.append("━━━━━━━━━━━━━━━━━━")
        lines.append("")

    if not opportunities:
        lines.append("No high-conviction setups this week. Stay in cash.")
    else:
        priority = opportunities[:3]
        watchlist = opportunities[3:]

        lines.append("🎯 PRIORITY — enter if price hits trigger:")
        lines.append("")
        for opp in priority:
            lines.append(format_opportunity(opp))
            lines.append("")

        if watchlist:
            lines.append("👀 WATCHING — not ready yet:")
            for opp in watchlist:
                ticker = opp.get("ticker", "???")
                entry_low = opp.get("entry_price_low", 0)
                entry_high = opp.get("entry_price_high", 0)
                conf = opp.get("confidence", 0)
                lines.append(
                    f"  • {ticker}  trigger ${entry_low:.2f}–${entry_high:.2f}  ({conf}/100)"
                )
            lines.append("")

    lines.append("Morning briefing will show exact entry signals each day.")
    return "\n".join(lines)


def format_morning_briefing(
    events: List[Dict[str, Any]],
    positions: List[Dict[str, Any]],
    macro: Optional[Dict[str, Any]] = None,
    enter_now: Optional[List[Dict[str, Any]]] = None,
    watchlist: Optional[List[Dict[str, Any]]] = None,
    safety: Optional[Dict[str, Any]] = None,
) -> str:
    """Format the daily morning briefing with a clear action plan.

    Sections (in order):
    1. TODAY'S ACTION PLAN — enter_now stocks at entry trigger
    2. HOLD — open positions with P&L and levels
    3. WATCHING — pending setups not yet at entry
    4. Market indicators / macro events
    """
    from datetime import date as dt_date

    lines = [
        f"{EMOJI['morning']} MORNING BRIEFING — {dt_date.today().strftime('%A %b %d')}",
        "",
    ]

    # ── 1. ENTER NOW ──────────────────────────────────────────────────────────
    if enter_now:
        lines.append("🚀 ENTER NOW — price at trigger zone:")
        lines.append("")
        for item in enter_now:
            ticker = item.get("ticker", "???")
            current = item.get("current_price", 0)
            entry = item.get("entry_price", 0)
            stop = item.get("stop_loss", 0)
            target = item.get("target_price", 0)
            conf = item.get("confidence", 0)
            dist = item.get("distance_pct", 0)
            dist_str = f"{dist:+.1f}% from trigger" if dist else ""
            lines.append(f"  ✅ {ticker}  now ${current:.2f}  {dist_str}")
            lines.append(f"     Entry: ${entry:.2f}  Stop: ${stop:.2f}  Target: ${target:.2f}  ({conf}/100)")
        lines.append("")
    else:
        lines.append("— No new entries today.")
        lines.append("")

    # ── 2. OPEN POSITIONS (HOLD) ──────────────────────────────────────────────
    if positions:
        lines.append(f"{EMOJI['money']} HOLD — your open positions:")
        lines.append("")
        for pos in positions:
            ticker = pos.get("ticker", "???")
            current = pos.get("current_price", 0)
            pnl = pos.get("unrealized_pnl", 0)
            pnl_pct = pos.get("unrealized_pnl_pct", 0)
            stop = pos.get("stop_loss", 0)
            target = pos.get("target_price", 0)
            days = pos.get("days_held", 0)
            sign = "+" if pnl >= 0 else ""
            emoji = EMOJI["bullish"] if pnl >= 0 else EMOJI["bearish"]
            lines.append(
                f"  {emoji} {ticker}  ${current:.2f}  {sign}${pnl:,.0f} ({sign}{pnl_pct:.1f}%)  Day {days}"
            )
            lines.append(f"     Stop: ${stop:.2f}  Target: ${target:.2f}")
        lines.append("")

    # ── 3. WATCHING ───────────────────────────────────────────────────────────
    if watchlist:
        lines.append("👀 WATCHING — waiting for price to hit trigger:")
        for item in watchlist:
            ticker = item.get("ticker", "???")
            current = item.get("current_price", 0)
            entry = item.get("entry_price", 0)
            dist = item.get("distance_pct", 0)
            conf = item.get("confidence", 0)
            lines.append(
                f"  • {ticker}  now ${current:.2f}  trigger ${entry:.2f}  ({dist:.1f}% away)  {conf}/100"
            )
        lines.append("")

    # ── 4. BREADTH / SAFETY ──────────────────────────────────────────────────
    if safety:
        breadth_pct = safety.get("breadth", {}).get("pct_above_200", 0)
        top_risk = safety.get("top", {}).get("top_risk", "low")
        dist_days = safety.get("top", {}).get("distribution_days", 0)
        gate_ok = safety.get("safe_to_enter", True)
        risk_emojis = {"low": "🟢", "medium": "🟡", "high": "🔴"}
        re = risk_emojis.get(top_risk, "⚪")
        gate_str = "✅ entries open" if gate_ok else "🔴 entries PAUSED"
        lines.append(
            f"Market: {breadth_pct:.0f}% stocks above 200MA  |  {dist_days} dist days  |  {gate_str} {re}"
        )
        if not gate_ok:
            lines.append(f"  ⚠️ {safety.get('summary', '')}")
        lines.append("")

    # ── 5. MACRO / EVENTS ────────────────────────────────────────────────────
    if events:
        lines.append(f"{EMOJI['warning']} Market events:")
        for ev in events:
            if isinstance(ev, dict):
                severity = ev.get("severity", "low")
                detail = ev.get("event_detail", ev.get("event_type", str(ev)))
            else:
                severity = "low"
                detail = str(ev)
            sev_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(severity, "⚪")
            lines.append(f"  {sev_emoji} {detail}")
        lines.append("")

    if macro:
        spy = macro.get("spy", {})
        vix = macro.get("vix", {})
        if spy or vix:
            parts = []
            if isinstance(spy, dict):
                parts.append(f"SPY {spy.get('current', 0):.0f} ({spy.get('change_pct', 0):+.1f}%)")
            if isinstance(vix, dict):
                parts.append(f"VIX {vix.get('current', 0):.1f}")
            if parts:
                lines.append("  " + "  |  ".join(parts))

    return "\n".join(lines)


def format_eod_summary(data: Dict[str, Any]) -> str:
    """Format the end-of-day summary."""
    from datetime import date as dt_date

    lines = [
        f"{EMOJI['evening']} END OF DAY — {dt_date.today().strftime('%A %b %d')}",
        "",
    ]

    positions = data.get("positions", [])
    total_pnl = data.get("total_pnl", 0)
    total_pnl_pct = data.get("total_pnl_pct", 0)

    sign = "+" if total_pnl >= 0 else ""
    emoji = EMOJI["bullish"] if total_pnl >= 0 else EMOJI["bearish"]

    lines.append(
        f"Portfolio P&L: {emoji} {sign}${total_pnl:,.2f} ({sign}{total_pnl_pct:.1f}%)"
    )
    lines.append("")

    if positions:
        for pos in positions:
            lines.append(format_position_update(pos))
            lines.append("")

    # Insider activity
    insider = data.get("insider_activity", [])
    if insider:
        lines.append("Insider activity detected:")
        for act in insider:
            lines.append(f"  • {act}")
        lines.append("")

    # Recommendations
    recommendations = data.get("recommendations", [])
    if recommendations:
        lines.append("Recommendations:")
        for rec in recommendations:
            lines.append(f"  → {rec}")

    return "\n".join(lines)


def format_action_needed(data: Dict[str, Any]) -> str:
    """Format an urgent action-needed alert."""
    ticker = data.get("ticker", "???")
    action = data.get("action", "")
    reason = data.get("reason", "")
    current_price = data.get("current_price", 0)
    pnl = data.get("pnl", 0)
    pnl_pct = data.get("pnl_pct", 0)

    sign = "+" if pnl >= 0 else ""
    pnl_emoji = EMOJI["bullish"] if pnl >= 0 else EMOJI["bearish"]

    lines = [
        f"{EMOJI['action']} ACTION NEEDED — {ticker}",
        "",
    ]

    if action:
        lines.append(f"{action}")
        lines.append("")

    lines.append(reason)
    lines.append("")

    if current_price:
        lines.append(f"  Price: ${current_price:.2f}")
    lines.append(f"  P&L: {pnl_emoji} {sign}${pnl:,.2f} ({sign}{pnl_pct:.1f}%)")

    return "\n".join(lines)


def format_entry_signal(data: Dict[str, Any]) -> str:
    """Format an intraday entry trigger alert — sent when a watchlist stock hits its entry price."""
    ticker = data.get("ticker", "???")
    current = data.get("current_price", 0)
    entry = data.get("entry_price", 0)
    stop = data.get("stop_loss", 0)
    target = data.get("target_price", 0)
    conf = data.get("confidence", 0)
    risk = abs(current - stop) if stop else 0
    reward = abs(target - current) if target else 0
    rr = round(reward / risk, 1) if risk > 0 else 0

    lines = [
        f"🚀 ENTRY SIGNAL — {ticker}",
        "",
        f"Price ${current:.2f} hit trigger ${entry:.2f}",
        "",
        f"  BUY {ticker} at market now",
        f"  {EMOJI['stop']} Stop loss: ${stop:.2f}  (risk ${risk:.2f}/share)",
        f"  {EMOJI['target']} Target:    ${target:.2f}  (reward ${reward:.2f}/share)",
        f"  Risk/Reward: 1:{rr}  |  Confidence: {conf}/100",
    ]

    notes = data.get("notes", "")
    if notes:
        lines.append(f"  Why: {notes}")

    return "\n".join(lines)


def format_bot_start(data: Dict[str, Any]) -> str:
    """Format the startup/redeploy confirmation alert."""
    from datetime import datetime as dt

    env = data.get("env", "unknown")
    dry_run = data.get("dry_run", False)
    init_ok = data.get("init_ok", {})
    next_scan = data.get("next_weekly_scan", "unknown")
    next_briefing = data.get("next_morning_briefing", "unknown")

    mode = "DRY RUN" if dry_run else "LIVE"
    now = dt.now().strftime("%b %d %H:%M")

    lines = [
        f"🟢 Bot redeployed successfully — {now}",
        f"Mode: {mode}  |  Env: {env}",
        "",
        "Components:",
        f"  {'✅' if init_ok.get('supabase') else '❌'} Supabase",
        f"  {'✅' if init_ok.get('telegram') else '❌'} Telegram",
        "",
        "Next scheduled:",
        f"  📊 Weekly scan:      {next_scan}",
        f"  ☀️  Morning briefing: {next_briefing}",
    ]

    return "\n".join(lines)


def start_command_listener() -> None:
    """Start a background polling loop for Telegram commands."""
    token = os.getenv("STOCKS_TELEGRAM_TOKEN")
    if not token:
        logger.warning("No STOCKS_TELEGRAM_TOKEN, skipping command listener.")
        return

    import asyncio
    import threading
    from telegram.ext import Application, CommandHandler
    
    def _run_polling():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        app = Application.builder().token(token).build()
        
        async def cmd_performance(update, context):
            from agent.portfolio import get_portfolio_summary
            summary = get_portfolio_summary()
            
            pnl = summary.get("total_unrealized_pnl", 0)
            sign = "+" if pnl >= 0 else ""
            emoji = EMOJI["bullish"] if pnl >= 0 else EMOJI["bearish"]
            
            lines = [
                f"{EMOJI['chart']} PERFORMANCE SUMMARY",
                f"Portfolio Value: ${summary.get('portfolio_value', 0):,.2f}",
                f"Invested: ${summary.get('total_invested', 0):,.2f}",
                f"Cash: ${summary.get('cash_available', 0):,.2f}",
                f"Unrealized P&L: {emoji} {sign}${pnl:,.2f}",
                f"Open Positions: {summary.get('open_positions', 0)}",
                f"Win Rate: {summary.get('win_rate', 0)}%",
            ]
            await update.message.reply_text("\n".join(lines))

        app.add_handler(CommandHandler("performance", cmd_performance))
        
        logger.info("Starting Telegram command listener (/performance)")
        app.run_polling(drop_pending_updates=True)

    t = threading.Thread(target=_run_polling, daemon=True, name="telegram-listener")
    t.start()
