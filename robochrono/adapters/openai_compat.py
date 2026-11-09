#!/usr/bin/env python3
# coding: utf-8
"""Adapter for OpenAI-compatible chat/completions endpoints.

The model's ``api`` configuration block supplies what varies by endpoint:

    url               chat/completions endpoint (required)
    model             the served model id (required)
    key_env           name of the environment variable holding the key;
                      keys themselves never appear in configuration files
    media_url_format  "data_url" (default) or bare "base64"
    extra_payload     endpoint dialect, merged last so it can override anything
    extra_headers     additional HTTP headers
    thinking_param    name of a boolean thinking toggle (DashScope-style
                      `enable_thinking` and friends)
    send_thinking     send `thinking: {"type": ...}` (GLM-style dialect)
    omit_temperature  omit temperature for models that reject it
    max_tokens_param  output-token budget field (default: max_tokens)
    video_input       "native" (default) or protocol-sampled "frames"
    frame_jpeg_quality JPEG quality for extracted frames (default: 90)
    min_video_seconds server-side minimum video duration, if the endpoint
                      enforces one

Two hard-won rules are load-bearing here. ``max_tokens`` must actually be
sent — a budget that lives only in result metadata leaves the server free to
apply its own. And the thinking toggle must actually be sent, in whichever
dialect the endpoint speaks: a configuration that *says* thinking is disabled
while the server happily reasons at length is worse than no setting, because
the results claim a uniformity that does not exist. ``max_tokens`` does not
cap reasoning tokens on the measured endpoints, so cost control comes only
from the toggle itself.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from .base import Adapter, AdapterResult, chat_completion_text, data_url, encode_file


def openai_content(parts: list[dict[str, Any]],
                   media_url_format: str = "data_url") -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for part in parts:
        kind = part.get("type")
        if kind == "text":
            content.append({"type": "text", "text": str(part.get("text", ""))})
        elif kind in {"image", "video"}:
            path = Path(str(part["path"]))
            if media_url_format == "data_url":
                url = data_url(path, kind)
            elif media_url_format == "base64":
                url = encode_file(path)
            else:
                raise ValueError(f"unsupported media_url_format: {media_url_format!r}")
            url_key = "image_url" if kind == "image" else "video_url"
            content.append({"type": url_key, url_key: {"url": url}})
        else:
            raise ValueError(f"unsupported content part type: {kind}")
    return content


def video_frame_count(protocol, *, total_frames: int, source_fps: float,
                      frames: dict[str, Any]) -> int:
    """Resolve one video's sample count using the shared protocol limits."""
    mode = str(frames.get("mode") or "")
    value = frames.get("value")
    if mode == "fps" and value:
        duration = total_frames / source_fps if source_fps > 0 else 0.0
        return protocol.frames_from_fps(duration, float(value))
    if mode == "uniform" and value:
        factor = max(1, int(protocol.frame_factor))
        count = round(int(value) / factor) * factor
        return int(max(protocol.min_frames, min(protocol.max_frames, count)))
    raise ValueError(f"no usable frame spec {frames!r}")


def midpoint_indices(frame_count: int, total_frames: int) -> list[int]:
    """Choose one frame at the midpoint of each equal temporal segment."""
    if total_frames <= 0:
        raise ValueError("video contains no frames")
    if frame_count <= 1:
        return [(total_frames - 1) // 2]
    segment = total_frames / frame_count
    return [min(total_frames - 1, int((index + 0.5) * segment))
            for index in range(frame_count)]


def extract_video_frames(
    path: Path,
    frames: dict[str, Any],
    protocol,
    *,
    media_url_format: str = "data_url",
    jpeg_quality: int = 90,
) -> tuple[list[tuple[str, float]], dict[str, Any]]:
    """Uniformly sample a video and encode its frames as JPEG image URLs."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "video_input='frames' requires OpenCV; install "
            "opencv-python-headless in the API environment") from exc

    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(video.get(cv2.CAP_PROP_FPS))
        if total_frames <= 0 or source_fps <= 0:
            raise ValueError(
                f"cannot read video metadata: {path} "
                f"(frames={total_frames}, fps={source_fps})")
        count = video_frame_count(
            protocol, total_frames=total_frames, source_fps=source_fps,
            frames=frames)
        indices = midpoint_indices(count, total_frames)
        encoded_frames: list[tuple[str, float]] = []
        for index in indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = video.read()
            if not ok:
                raise ValueError(f"cannot decode frame {index} from {path}")
            encoded, jpeg = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not encoded:
                raise ValueError(f"cannot encode frame {index} from {path}")
            raw = base64.b64encode(jpeg.tobytes()).decode("ascii")
            url = (f"data:image/jpeg;base64,{raw}"
                   if media_url_format == "data_url" else raw)
            encoded_frames.append((url, index / source_fps))
    finally:
        video.release()

    return encoded_frames, {
        "source": str(path),
        "op": "extract_frames",
        "source_frames": total_frames,
        "source_fps": source_fps,
        "sampled_frames": len(encoded_frames),
        "sampled_indices": indices,
        "timestamps_seconds": [round(index / source_fps, 6) for index in indices],
    }


class OpenAICompatAdapter(Adapter):
    def __init__(self, model, protocol, runtime=None) -> None:
        super().__init__(model, protocol, runtime)
        api = model.api
        for required in ("url", "model"):
            if not api.get(required):
                raise ValueError(f"model {model.slug!r}: api.{required} is required "
                                 f"for the openai_compat adapter")
        self.url = str(api["url"])
        self.served_model = str(api["model"])
        self.media_url_format = str(api.get("media_url_format", "data_url"))
        self.extra_payload = dict(api.get("extra_payload") or {})
        self.extra_headers = dict(api.get("extra_headers") or {})
        self.omit_temperature = bool(api.get("omit_temperature", False))
        self.max_tokens_param = str(api.get("max_tokens_param", "max_tokens"))
        self.video_input = str(api.get("video_input", "native"))
        if self.video_input not in {"native", "frames"}:
            raise ValueError(
                f"model {model.slug!r}: api.video_input must be 'native' or "
                f"'frames', got {self.video_input!r}")
        self.frame_jpeg_quality = int(api.get("frame_jpeg_quality", 90))
        if not 1 <= self.frame_jpeg_quality <= 100:
            raise ValueError(
                f"model {model.slug!r}: api.frame_jpeg_quality must be 1..100")
        self.min_video_seconds = float(api.get("min_video_seconds") or 0.0)
        # The request-body limit is a property of the serving endpoint, not of
        # the protocol; the protocol value is only the fallback.
        self.max_request_bytes = int(api.get("max_request_bytes",
                                             protocol.max_request_bytes))
        self.proxy = api.get("proxy")
        # The key is looked up at construction but only *required* at call
        # time: a missing credential should stop a call, not a dry run or a
        # configuration check. Preflight reports it as its own line item.
        self.key_env = api.get("key_env")
        self.api_key = os.environ.get(str(self.key_env), "") if self.key_env else ""

    # -- payload -----------------------------------------------------------

    def request_content(
        self, parts: list[dict[str, Any]], frames: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
        """Convert parts and report the exact frame sampling sent to the API."""
        if self.video_input == "native":
            return openai_content(parts, self.media_url_format), {}, []

        content: list[dict[str, Any]] = []
        frames_used: dict[str, int] = {}
        transforms: list[dict[str, Any]] = []
        video_index = 0
        for part in parts:
            if part.get("type") != "video":
                content.extend(openai_content([part], self.media_url_format))
                continue

            sampled, record = extract_video_frames(
                Path(str(part["path"])), frames, self.protocol,
                media_url_format=self.media_url_format,
                jpeg_quality=self.frame_jpeg_quality)
            for frame_index, (url, timestamp) in enumerate(sampled, 1):
                content.append({
                    "type": "text",
                    "text": (f"Video{video_index + 1} Frame{frame_index} "
                             f"| t={timestamp:.3f}s"),
                })
                content.append({"type": "image_url", "image_url": {"url": url}})
            frames_used[f"video{video_index}"] = len(sampled)
            transforms.append(record)
            video_index += 1
        return content, frames_used, transforms

    def _payload_from_content(self, content: list[dict[str, Any]]) -> dict[str, Any]:
        api = self.model.api
        payload: dict[str, Any] = {
            "model": self.served_model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ],
            **({} if self.omit_temperature else {"temperature": self.temperature}),
            self.max_tokens_param: self.max_new_tokens,
            # dialect escape hatch: merged last, may override anything above
            **self.extra_payload,
        }
        if api.get("send_thinking"):
            payload.setdefault("thinking", {"type": self.thinking})
        param = api.get("thinking_param")
        if param:
            payload.setdefault(str(param), self.thinking == "enabled")
        return payload

    def payload(self, parts: list[dict[str, Any]],
                frames: dict[str, Any] | None = None) -> dict[str, Any]:
        """The request body. Separate from ``call`` so tests can pin it."""
        content, _, _ = self.request_content(parts, frames or {})
        return self._payload_from_content(content)

    def generation_settings(self) -> dict[str, Any]:
        settings = super().generation_settings()
        if self.omit_temperature:
            settings["temperature"] = None
        if "reasoning_effort" in self.extra_payload:
            settings["reasoning_effort"] = self.extra_payload["reasoning_effort"]
        return settings

    # -- call --------------------------------------------------------------

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
        # Frame mode never uploads the MP4, so resizing or padding the MP4
        # would waste work and change the sampled images.
        if self.video_input == "native" and (budget > 0 or self.min_video_seconds > 0):
            cache_dir = Path(self.runtime.get("media_cache_dir") or ".cache/media")
            parts, transforms = prepare_parts(
                parts, budget or (1 << 62), cache_dir,
                min_video_seconds=self.min_video_seconds)

        content, frames_used, frame_transforms = self.request_content(parts, frames)
        transforms.extend(frame_transforms)

        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        raw = post_with_retries(self.url, headers=headers,
                                payload=self._payload_from_content(content),
                                timeout=self.protocol.timeout,
                                max_retries=self.protocol.max_retries,
                                proxy=self.proxy)
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        return AdapterResult(text=chat_completion_text(raw), raw=raw,
                             frames_used=frames_used, usage=usage,
                             media_transforms=transforms)


def build(model, protocol, runtime=None) -> OpenAICompatAdapter:
    return OpenAICompatAdapter(model, protocol, runtime)
