"""VisionPlugin — image analysis via the local multimodal LLM.

Exposes a `view_image` tool that reads an image from disk, sends it to
the LLM as a multimodal content block, and returns the model's text
analysis. The model is the main vision-capable LLM (swift-1.5 with
mmproj), running locally on the llama.cpp server. No external service,
no relay: the image bytes go from disk to the local API and back.

Uses the "snap" LLM stage (no reasoning overhead for a single-image
description).

Tool:
  view_image(path, prompt?) — analyze an image file
    path:   absolute or workspace-relative path to an image file
    prompt: optional instruction (default: "Describe this image in detail.")
    Returns the model's text response.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.tool import Tool, tool_to_schema

logger = logging.getLogger("corvidae.vision")

# Maximum image size to send to the LLM (10 MB, matching wiki attachments).
MAX_IMAGE_BYTES = 10 * 1024 * 1024

# Extensions we accept. Anything else is refused at the tool boundary.
IMAGE_EXTENSIONS: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

DEFAULT_PROMPT = "Describe this image in detail."


class VisionPlugin(CorvidaePlugin):
    """Image analysis tool backed by the local multimodal LLM."""

    name = "vision"
    depends_on = frozenset({"llm"})

    def __init__(self) -> None:
        super().__init__()
        self._workspace = Path(os.environ.get(
            "LEX_WORKSPACE", "/home/lex/workspace"
        ))

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        tool_registry.append(Tool(
            name="view_image",
            fn=self.view_image,
            schema=tool_to_schema(self.view_image),
        ))

    @hookimpl
    def register_llm_stages(self, stages: dict) -> None:
        stages["vision"] = "chat"

    @hookimpl
    async def view_image(self, path: str, prompt: str = DEFAULT_PROMPT) -> str:
        """Analyze an image file using the local vision LLM.

        Reads the image from disk, sends it to the model as a multimodal
        content block, and returns the model's text description or analysis.
        Accepts PNG, JPEG, WebP, GIF, and BMP files up to 10 MB.

        Args:
            path: Path to the image file. Absolute paths are used as-is;
                relative paths are resolved against the workspace directory.
            prompt: Instruction for the analysis.
        """
        if not path:
            return "Error: 'path' is required."

        # Resolve the path.
        p = Path(path)
        if not p.is_absolute():
            p = self._workspace / p

        if not p.exists():
            return f"Error: file not found: {p}"
        if not p.is_file():
            return f"Error: not a file: {p}"

        # Check the extension.
        suffix = p.suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            supported = ", ".join(sorted(IMAGE_EXTENSIONS))
            return (
                f"Error: unsupported image format '{suffix}'. "
                f"Supported: {supported}"
            )

        # Check the size.
        size = p.stat().st_size
        if size > MAX_IMAGE_BYTES:
            return (
                f"Error: file is {size / 1024 / 1024:.1f} MB, "
                f"exceeds the {MAX_IMAGE_BYTES // 1024 // 1024} MB limit."
            )
        if size == 0:
            return f"Error: file is empty: {p}"

        # Read and encode.
        try:
            data = p.read_bytes()
        except OSError as exc:
            return f"Error: cannot read file: {exc}"

        b64 = base64.b64encode(data).decode("ascii")
        mime = IMAGE_EXTENSIONS[suffix]
        data_url = f"data:{mime};base64,{b64}"

        # Send to the LLM.
        from corvidae.llm_plugin import LLMPlugin

        try:
            llm_plugin = get_dependency(self.pm, "llm", LLMPlugin)
            client = llm_plugin.get_client("vision")
        except Exception as exc:
            return f"Error: cannot access LLM client: {exc}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ]

        try:
            response = await client.chat(messages=messages)
        except Exception as exc:
            logger.exception("vision: LLM call failed")
            return f"Error: LLM call failed: {exc}"

        text = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
            or ""
        )
        if not text:
            return "Error: model returned empty response."

        return text
