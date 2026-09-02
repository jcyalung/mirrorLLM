"""Tiny in-process event log the mirror front end polls for voice-turn state.

The GPIO button flow runs inside this same API process (see
`api/gpio_button.py`) but the MagicMirror module lives in a separate
Electron/Node process, so it has no way to observe a Python thread directly.
Rather than push state over a socket, the button flow appends discrete events
here and `GET /voice/events?since=<id>` hands back everything new -- plain
polling, same as the rest of this codebase (see the spotify module).

Events are kept in order and never mutated, so a client that remembers the
last id it saw can never miss or replay one, regardless of its poll cadence.
"""

import itertools
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional

# Generous cap: a burst of events from one turn is a handful, so this covers
# many turns' worth of history for a client that hasn't polled in a while.
_MAX_EVENTS = 200

_lock = threading.Lock()
_events: Deque[Dict[str, Any]] = deque(maxlen=_MAX_EVENTS)
_next_id = itertools.count(1)


def emit(event_type: str, payload: Optional[dict] = None) -> None:
    """Record a new event. Types: listening_start, listening_stop,
    user_message ({text}), assistant_thinking (no payload -- emitted right
    before the agent call so the front end can show a placeholder while it
    waits), assistant_message ({text, display_card}), turn_idle (no payload --
    emitted instead of assistant_thinking/assistant_message when the mic timed
    out with nobody speaking, so the front end can drop its loading state),
    calendar_updated ({html_link} -- emitted right after create_calendar_event
    succeeds, so the gcalendar module can refetch instead of waiting out its
    own multi-minute poll interval)."""
    with _lock:
        _events.append({"id": next(_next_id), "type": event_type, "payload": payload or {}})


def since(last_id: int) -> List[Dict[str, Any]]:
    """Every event with id greater than `last_id`, oldest first."""
    with _lock:
        return [event for event in _events if event["id"] > last_id]
