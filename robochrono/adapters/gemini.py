#!/usr/bin/env python3
# coding: utf-8
"""Adapter for the Gemini generateContent API.

The ``api`` block supplies ``url`` (with an optional ``{model}`` placeholder),
``model``, and ``key_env``; media goes inline as base64. The generation
budget is sent as ``maxOutputTokens`` — the same must-actually-send rule as
the OpenAI-compatible adapter, which historically missed this exact field on
this exact dialect. ``generation_config`` from the configuration is merged
into ``generationConfig`` key by key: replacing the whole object would silently
drop the temperature.

Three rules this adapter enforces, each learned the hard way:

* The key travels in a request header (``key_header``, default
  ``x-goog-api-key``; ``key_prefix`` for ``Authorization: Bearer`` dialects),
  never in the URL. A key in the query string ends up in every connection
  error message, and from there in the ``error`` field of result rows.
* The video frame rate sent as ``videoMetadata.fps`` comes from the
  dimension's frame spec that the engine passes to ``call`` — the same
  protocol value the local adapters sample at — so action_time (1 fps) and
  the clip dimensions (2 fps) are not silently collapsed to one rate.
  ``api.video_fps`` may pin a number instead, or ``0`` to send no rate; both
  are recorded in ``media_transforms`` so the run keeps what was sent.
* Field names are the canonical lowerCamelCase of the REST API
  (``inlineData``, ``mimeType``, ``videoMetadata``): Google accepts both
  spellings, relays that re-serialise the body do not always.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import Adapter, AdapterResult, encode_file, media_mime

DEFAULT_KEY_HEADER = "x-goog-api-key"


def video_fps_from_spec(frames: dict[str, Any] | None) -> float:
    """The frame rate a dimension's protocol spec asks for, or 0 when it is
    not expressed as a rate (``uniform`` counts have no Gemini equivalent)."""
    frames = frames or {}
    if str(frames.get("mode") or "") == "fps" and frames.get("value"):
        return float(frames["value"])
    return 0.0


def gemini_parts(parts: list[dict[str, Any]], *,
                 video_fps: float = 0.0) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for part in parts:
        kind = part.get("type")
        if kind == "text":
            output.append({"text": str(part.get("text", ""))})
        elif kind in {"image", "video"}:
            path = Path(str(part["path"]))
            media_part: dict[str, Any] = {
                "inlineData": {"mimeType": media_mime(path, kind),
                               "data": encode_file(path)}
            }
            if kind == "video" and video_fps > 0:
                media_part["videoMetadata"] = {"fps": video_fps}
            output.append(media_part)
        else:
            raise ValueError(f"unsupported content part type: {kind}")
    return output


def gemini_text(response: dict[str, Any]) -> str:
    try:
        parts = response["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected Gemini response: {response}") from exc
    return "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))


class GeminiAdapter(Adapter):
    def __init__(self, model, protocol, runtime=None) -> None:
        super().__init__(model, protocol, runtime)
        api = model.api
        for required in ("url", "model"):
            if not api.get(required):
                raise ValueError(f"model {model.slug!r}: api.{required} is required "
                                 f"for the gemini adapter")
        self.served_model = str(api["model"])
        self.url = str(api["url"]).format(model=self.served_model)
        if "key=" in self.url:
            raise ValueError(f"model {model.slug!r}: api.url carries a key= query "
                             f"parameter; keys go in key_env and are sent as a header")
        self.min_video_seconds = float(api.get("min_video_seconds") or 0.0)
        self.max_request_bytes = int(api.get("max_request_bytes",
                                             protocol.max_request_bytes))
        # None: follow the dimension's protocol spec. A number pins the rate
        # for every dimension; 0 sends no videoMetadata at all.
        self.video_fps_override: float | None = (
            None if api.get("video_fps") is None else float(api["video_fps"]))
        self.key_header = str(api.get("key_header") or DEFAULT_KEY_HEADER)
        self.key_prefix = str(api.get("key_prefix") or "")
        self.proxy = api.get("proxy")
        # The key is looked up at construction but only *required* at call
        # time: a missing credential should stop a call, not a dry run or a
        # configuration check. Preflight reports it as its own line item.
        self.key_env = api.get("key_env")
        self.api_key = os.environ.get(str(self.key_env), "") if self.key_env else ""

    def video_fps(self, frames: dict[str, Any] | None) -> float:
        if self.video_fps_override is not None:
            return self.video_fps_override
        return video_fps_from_spec(frames)

    def auth_headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {self.key_header: f"{self.key_prefix}{self.api_key}"}

    def payload(self, parts: list[dict[str, Any]], *,
                video_fps: float = 0.0) -> dict[str, Any]:
        api = self.model.api
        gen_cfg: dict[str, Any] = {"temperature": self.temperature,
                                   "maxOutputTokens": self.max_new_tokens}
        # key-by-key, never a whole-object replace
        gen_cfg.update(api.get("generation_config") or {})
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": self.system_prompt}]},
            "contents": [{"role": "user",
                          "parts": gemini_parts(parts, video_fps=video_fps)}],
            "generationConfig": gen_cfg,
        }
        for k, v in (api.get("extra_payload") or {}).items():
            if k == "generationConfig" and isinstance(v, dict):
                payload["generationConfig"].update(v)
            else:
                payload[k] = v
        return payload

    def call(self, parts: list[dict[str, Any]], *, frames: dict[str, Any],
             key: str = "") -> AdapterResult:
        if self.key_env and not self.api_key:
            raise RuntimeError(f"{self.model.slug}: environment variable "
                               f"{self.key_env} is not set — see configs/README.md "
                               f"on keys, or run `robochrono preflight`")
        from .base import post_with_retries
        from ..media_prep import prepare_parts

        transforms: list[dict[str, Any]] = []
        budget = self.max_request_bytes
        if budget > 0 or self.min_video_seconds > 0:
            cache_dir = Path(self.runtime.get("media_cache_dir") or ".cache/media")
            parts, transforms = prepare_parts(
                parts, budget or (1 << 62), cache_dir,
                min_video_seconds=self.min_video_seconds)

        fps = self.video_fps(frames)
        if any(p.get("type") == "video" for p in parts):
            transforms.append({"op": "video_metadata",
                               "fps": fps if fps > 0 else None,
                               "source": ("api.video_fps"
                                          if self.video_fps_override is not None
                                          else "protocol.frames")})

        raw = post_with_retries(
            self.url,
            headers={"Content-Type": "application/json",
                     **self.auth_headers(),
                     **(self.model.api.get("extra_headers") or {})},
            payload=self.payload(parts, video_fps=fps),
            timeout=self.protocol.timeout, max_retries=self.protocol.max_retries,
            proxy=self.proxy)
        usage = raw.get("usageMetadata") if isinstance(raw.get("usageMetadata"), dict) else {}
        return AdapterResult(text=gemini_text(raw), raw=raw,
                             usage=usage, media_transforms=transforms)


def build(model, protocol, runtime=None) -> GeminiAdapter:
    return GeminiAdapter(model, protocol, runtime)
