from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .logging import JsonlLogger, redact_headers


class InputError(RuntimeError):
    pass


def is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in {"http", "https"}


def _head_request(url: str, headers: Dict[str, str]) -> Optional[Dict[str, str]]:
    req = Request(url, method="HEAD", headers=headers)
    try:
        with urlopen(req) as resp:
            return dict(resp.headers)
    except HTTPError:
        return None


def _range_request(url: str, headers: Dict[str, str]) -> Optional[Dict[str, str]]:
    range_headers = dict(headers)
    range_headers["Range"] = "bytes=0-1023"
    req = Request(url, method="GET", headers=range_headers)
    try:
        with urlopen(req) as resp:
            if resp.status in {200, 206}:
                return dict(resp.headers)
    except HTTPError:
        return None
    return None


def validate_direct_video_url(url: str, headers: Dict[str, str]) -> None:
    header_info = _head_request(url, headers)
    content_type = ""
    if header_info:
        content_type = header_info.get("Content-Type", "")
    if content_type.startswith("video/"):
        return
    range_info = _range_request(url, headers)
    if range_info is None:
        raise InputError(
            "URL does not appear to be a direct-download video. Provide a direct file URL or local file."
        )


def _cache_path(url: str, headers: Dict[str, str]) -> Path:
    cache_root = Path.home() / ".cache" / "clipgen"
    cache_root.mkdir(parents=True, exist_ok=True)
    header_blob = json.dumps(sorted(headers.items())).encode("utf-8")
    hasher = hashlib.sha256()
    hasher.update(url.encode("utf-8"))
    hasher.update(header_blob)
    return cache_root / f"{hasher.hexdigest()}.mp4"


def download_with_resume(url: str, headers: Dict[str, str], logger: JsonlLogger) -> Path:
    validate_direct_video_url(url, headers)
    target = _cache_path(url, headers)
    tmp_path = target.with_suffix(".part")
    existing = tmp_path.stat().st_size if tmp_path.exists() else 0

    req_headers = dict(headers)
    if existing:
        req_headers["Range"] = f"bytes={existing}-"
    logger.log(
        "download",
        "start",
        url=url,
        headers=redact_headers(headers),
        existing_bytes=existing,
    )
    req = Request(url, headers=req_headers)
    try:
        with urlopen(req) as resp, tmp_path.open("ab") as handle:
            chunk = resp.read(1024 * 1024)
            while chunk:
                handle.write(chunk)
                chunk = resp.read(1024 * 1024)
    except (HTTPError, URLError) as exc:
        raise InputError(f"Failed to download URL: {exc}") from exc

    os.replace(tmp_path, target)
    logger.log("download", "complete", path=str(target))
    return target


def resolve_input(path_or_url: str, headers: Optional[Dict[str, str]], logger: JsonlLogger) -> Path:
    if is_url(path_or_url):
        return download_with_resume(path_or_url, headers or {}, logger)
    path = Path(path_or_url)
    if not path.exists():
        raise InputError(f"Input file not found: {path}")
    return path
