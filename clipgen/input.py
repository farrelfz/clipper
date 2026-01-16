from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

import requests

from clipgen.utils import sha256_text


class InputResolutionError(ValueError):
    pass


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def validate_direct_video_url(url: str, headers: Dict[str, str]) -> None:
    response = requests.head(url, allow_redirects=True, headers=headers, timeout=15)
    content_type = response.headers.get("Content-Type", "")
    if content_type.startswith("video/"):
        return
    if response.status_code in {403, 401}:
        raise InputResolutionError(
            "URL requires authorization. Provide a direct video URL and include --header values."
        )
    range_headers = dict(headers)
    range_headers["Range"] = "bytes=0-1023"
    range_response = requests.get(url, headers=range_headers, stream=True, timeout=15)
    if range_response.status_code not in {200, 206}:
        raise InputResolutionError(
            "URL does not appear to be a direct-download video. Provide a direct file URL."
        )
    sniff = range_response.headers.get("Content-Type", "")
    if sniff.startswith("video/"):
        return
    guessed, _ = mimetypes.guess_type(url)
    if not (guessed or "video" in sniff):
        raise InputResolutionError(
            "URL is not a direct video file. Provide a local file path or direct-download URL."
        )


def download_with_resume(url: str, headers: Dict[str, str], cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(urlparse(url).path).suffix or ".mp4"
    target = cache_dir / f"{sha256_text(url)}{suffix}"
    temp = target.with_suffix(target.suffix + ".part")
    downloaded = temp.stat().st_size if temp.exists() else 0
    request_headers = dict(headers)
    if downloaded:
        request_headers["Range"] = f"bytes={downloaded}-"
    with requests.get(url, headers=request_headers, stream=True, timeout=30) as response:
        if response.status_code in {401, 403}:
            raise InputResolutionError(
                "URL requires authorization. Provide valid --header values for download."
            )
        response.raise_for_status()
        mode = "ab" if downloaded else "wb"
        with temp.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temp.rename(target)
    return target


def resolve_input(source: str, headers: Iterable[str], cache_dir: Path) -> Path:
    header_map: Dict[str, str] = {}
    for header in headers:
        if ":" not in header:
            raise InputResolutionError("Header must be in 'Key: Value' format.")
        key, value = header.split(":", 1)
        header_map[key.strip()] = value.strip()
    if is_url(source):
        validate_direct_video_url(source, header_map)
        return download_with_resume(source, header_map, cache_dir)
    path = Path(source)
    if not path.exists():
        raise InputResolutionError("Input file does not exist.")
    return path
