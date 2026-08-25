"""Spotify now-playing lookups for the mirror.

Reading playback state is a user-scoped call, so this needs the authorization
code flow rather than client credentials: run `python -m flow.tools.spotify`
once to sign in, after which the cached refresh token keeps working headlessly.
"""

import os
import random

from dotenv import load_dotenv
from spotipy import Spotify
from spotipy.cache_handler import CacheFileHandler
from spotipy.oauth2 import SpotifyOAuth

load_dotenv(".env.local")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOKEN_FILE = os.path.join(_ROOT, "spotify-token.json")

# Reading state needs the two read scopes; the agent tools additionally need to
# list the user's own playlists and drive playback. Widening this list
# invalidates an existing token, so the sign-in has to be re-run after a change.
SCOPE = " ".join(
    [
        "user-read-currently-playing",
        "user-read-playback-state",
        "user-modify-playback-state",
        "playlist-read-private",
        "playlist-read-collaborative",
    ]
)

# Spotify dropped `localhost` from the allowed redirect URIs in November 2025;
# loopback IP literals over plain HTTP are still accepted. This must match the
# URI registered in the Spotify developer dashboard exactly.
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8888/callback"


class SpotifyNotConfigured(RuntimeError):
    """Raised when the Spotify client credentials are missing."""


class SpotifyNotAuthorized(RuntimeError):
    """Raised when the mirror reads playback before the one-time sign-in."""


class SpotifyNoDevice(RuntimeError):
    """Raised when playback is requested with no Spotify device available."""


def _redirect_uri() -> str:
    return os.environ.get("SPOTIFY_REDIRECT_URI", DEFAULT_REDIRECT_URI)


def _auth_manager(open_browser: bool = False) -> SpotifyOAuth:
    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise SpotifyNotConfigured(
            "Spotify is not configured. Set SPOTIFY_CLIENT_ID and "
            "SPOTIFY_CLIENT_SECRET in .env.local."
        )

    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=_redirect_uri(),
        scope=SCOPE,
        cache_handler=CacheFileHandler(cache_path=TOKEN_FILE),
        open_browser=open_browser,
    )


def _client(interactive: bool = False) -> Spotify:
    auth_manager = _auth_manager(open_browser=interactive)
    token = auth_manager.validate_token(auth_manager.cache_handler.get_cached_token())

    if not token:
        if not interactive:
            raise SpotifyNotAuthorized(
                "Spotify is not authorized, or the cached token predates the "
                "current permissions. Run `python -m flow.tools.spotify` and "
                "approve access."
            )
        auth_manager.get_access_token(as_dict=False)

    return Spotify(auth_manager=auth_manager)


def _artwork(images: list) -> str:
    """Largest available cover image, or an empty string."""
    if not images:
        return ""
    best = max(images, key=lambda image: image.get("width") or 0)
    return best.get("url") or ""


def _track_payload(item: dict) -> dict:
    """Flatten a track or podcast episode into one shape the mirror can render."""
    if item.get("type") == "episode":
        show = item.get("show") or {}
        return {
            "title": item.get("name") or "",
            "artist": show.get("name") or "",
            "album": show.get("name") or "",
            "albumArt": _artwork(item.get("images") or show.get("images") or []),
        }

    album = item.get("album") or {}
    artists = [a.get("name") for a in item.get("artists") or [] if a.get("name")]
    return {
        "title": item.get("name") or "",
        "artist": ", ".join(artists),
        "album": album.get("name") or "",
        "albumArt": _artwork(album.get("images") or []),
    }


def get_now_playing() -> dict:
    """Current playback state, shaped for the mirror module."""
    playback = _client().current_playback(additional_types="track,episode")

    item = (playback or {}).get("item")
    if not playback or not item:
        # Nothing playing, an ad, or no active device.
        return {"isPlaying": False}

    payload = _track_payload(item)
    payload.update(
        {
            "id": item.get("id") or "",
            "isPlaying": bool(playback.get("is_playing")),
            "progressMs": playback.get("progress_ms") or 0,
            "durationMs": item.get("duration_ms") or 0,
        }
    )
    return payload


def _playback_device(client: Spotify) -> str:
    """Pick where to play: whatever is already active, else any known device."""
    devices = (client.devices() or {}).get("devices") or []
    if not devices:
        raise SpotifyNoDevice(
            "No Spotify device is available. Open Spotify on a phone, desktop, "
            "or speaker and try again."
        )

    active = next((d for d in devices if d.get("is_active")), None)
    return (active or devices[0])["id"]


def list_playlists(limit: int = 20) -> list[dict]:
    """The user's own playlists, newest-followed first."""
    client = _client()
    # The model sometimes sends numbers as strings.
    page = client.current_user_playlists(limit=min(max(int(limit), 1), 50))

    return [
        {
            "name": item.get("name") or "",
            "owner": (item.get("owner") or {}).get("display_name") or "",
            "tracks": (item.get("tracks") or {}).get("total") or 0,
        }
        for item in (page or {}).get("items") or []
        if item
    ]


def play_song(query: str) -> dict:
    """Search for a track and start playing the best match."""
    client = _client()
    results = client.search(q=query, type="track", limit=1)
    items = ((results or {}).get("tracks") or {}).get("items") or []

    if not items:
        return {"status": "not_found", "query": query}

    track = items[0]
    artists = ", ".join(a["name"] for a in track.get("artists") or [] if a.get("name"))

    device_id = _playback_device(client)
    client.start_playback(device_id=device_id, uris=[track["uri"]])

    return {
        "status": "playing",
        "title": track.get("name") or "",
        "artist": artists,
        "album": (track.get("album") or {}).get("name") or "",
    }


def _find_playlist(client: Spotify, name: str) -> dict:
    """Match a playlist by name, preferring an exact hit over a partial one."""
    wanted = name.strip().lower()
    playlists = []

    # The library can span several pages; 100 covers any realistic library.
    for offset in (0, 50):
        page = client.current_user_playlists(limit=50, offset=offset)
        items = [p for p in (page or {}).get("items") or [] if p]
        playlists.extend(items)
        if len(items) < 50:
            break

    for playlist in playlists:
        if (playlist.get("name") or "").strip().lower() == wanted:
            return playlist
    for playlist in playlists:
        if wanted in (playlist.get("name") or "").strip().lower():
            return playlist

    available = [p.get("name") for p in playlists if p.get("name")]
    raise LookupError(
        f"No playlist matching '{name}'. Available: {', '.join(available) or 'none'}"
    )


def shuffle_playlist(name: str) -> dict:
    """Turn on shuffle and start the named playlist at a random track."""
    client = _client()
    playlist = _find_playlist(client, name)
    total = (playlist.get("tracks") or {}).get("total") or 0

    if not total:
        return {"status": "empty", "playlist": playlist.get("name") or name}

    device_id = _playback_device(client)
    client.shuffle(True, device_id=device_id)

    # Shuffle alone always begins on the first track, so pick the entry point.
    client.start_playback(
        device_id=device_id,
        context_uri=playlist["uri"],
        offset={"position": random.randrange(total)},
    )

    return {
        "status": "shuffling",
        "playlist": playlist.get("name") or name,
        "tracks": total,
    }


def authorize() -> str:
    """Run the one-time browser sign-in and cache the refresh token."""
    client = _client(interactive=True)
    me = client.me()
    return me.get("display_name") or me.get("id") or "unknown"


LIST_PLAYLISTS_TOOL = {
    "type": "function",
    "function": {
        "name": "list_spotify_playlists",
        "description": "List the playlists saved in the user's Spotify library.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "How many playlists to return (1-50). Defaults to 20.",
                },
            },
            "required": [],
        },
    },
}

PLAY_SONG_TOOL = {
    "type": "function",
    "function": {
        "name": "play_spotify_song",
        "description": (
            "Search Spotify and immediately play the best matching track on the "
            "user's active device."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Song to play, ideally 'title artist' e.g. 'Redbone Childish Gambino'.",
                },
            },
            "required": ["query"],
        },
    },
}

SHUFFLE_PLAYLIST_TOOL = {
    "type": "function",
    "function": {
        "name": "shuffle_spotify_playlist",
        "description": (
            "Turn on shuffle and start playing one of the user's playlists by name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Playlist name as the user said it; partial names are matched.",
                },
            },
            "required": ["name"],
        },
    },
}


if __name__ == "__main__":
    print(f"Redirect URI: {_redirect_uri()}")
    print("Opening browser for Spotify sign-in...")
    print(f"\nAuthorized as: {authorize()}")
    print(f"Token cached at: {TOKEN_FILE}")
