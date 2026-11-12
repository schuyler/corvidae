"""Tests for VisionPlugin — image analysis via the local multimodal LLM."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corvidae_vision import (
    DEFAULT_PROMPT,
    IMAGE_EXTENSIONS,
    MAX_IMAGE_BYTES,
    VisionPlugin,
)

# Minimal 1x1 transparent PNG
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFc"
    "SJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


@pytest.fixture
def plugin() -> VisionPlugin:
    p = VisionPlugin()
    p.pm = MagicMock()
    return p


def _mock_llm(response_text: str = "A tiny image."):
    """Build a mocked LLM plugin + client returning response_text."""
    llm_plugin = MagicMock()
    client = MagicMock()
    client.chat = AsyncMock(return_value={
        "choices": [{"message": {"content": response_text}}]
    })
    llm_plugin.get_client.return_value = client
    return llm_plugin, client


class TestToolRegistration:
    def test_registers_view_image(self, plugin: VisionPlugin):
        registry: list = []
        plugin.register_tools(registry)
        assert len(registry) == 1
        assert registry[0].name == "view_image"

    def test_schema_has_path_required(self, plugin: VisionPlugin):
        registry: list = []
        plugin.register_tools(registry)
        schema = registry[0].schema
        params = schema["function"]["parameters"]
        assert "path" in params["required"]


class TestPathResolution:
    @pytest.mark.asyncio
    async def test_file_not_found(self, plugin: VisionPlugin):
        result = await plugin.view_image("/nonexistent/image.png")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_directory_rejected(self, plugin: VisionPlugin, tmp_path: Path):
        result = await plugin.view_image(str(tmp_path))
        assert "not a file" in result

    @pytest.mark.asyncio
    async def test_relative_path_resolves_against_workspace(
        self, plugin: VisionPlugin, tmp_path: Path
    ):
        png = tmp_path / "test.png"
        png.write_bytes(TINY_PNG)
        plugin._workspace = tmp_path

        with patch("corvidae_vision.get_dependency") as mock_dep:
            llm_plugin, client = _mock_llm()
            mock_dep.return_value = llm_plugin

            result = await plugin.view_image("test.png")

        assert "tiny image" in result
        call_args = client.chat.call_args
        messages = call_args[1]["messages"]
        content = messages[0]["content"]
        image_block = [b for b in content if b.get("type") == "image_url"][0]
        assert image_block["image_url"]["url"].startswith("data:image/png;base64,")


class TestFormatValidation:
    @pytest.mark.asyncio
    async def test_unsupported_extension_rejected(
        self, plugin: VisionPlugin, tmp_path: Path
    ):
        txt = tmp_path / "readme.txt"
        txt.write_text("not an image")
        result = await plugin.view_image(str(txt))
        assert "unsupported image format" in result
        assert ".txt" in result

    @pytest.mark.asyncio
    async def test_empty_file_rejected(
        self, plugin: VisionPlugin, tmp_path: Path
    ):
        empty = tmp_path / "empty.png"
        empty.write_bytes(b"")
        result = await plugin.view_image(str(empty))
        assert "empty" in result

    @pytest.mark.asyncio
    async def test_oversized_file_rejected(
        self, plugin: VisionPlugin, tmp_path: Path
    ):
        big = tmp_path / "big.png"
        big.write_bytes(b"\x00" * (MAX_IMAGE_BYTES + 1))
        result = await plugin.view_image(str(big))
        assert "exceeds" in result
        assert "MB" in result


class TestLLMCall:
    @pytest.mark.asyncio
    async def test_default_prompt_used(
        self, plugin: VisionPlugin, tmp_path: Path
    ):
        png = tmp_path / "test.png"
        png.write_bytes(TINY_PNG)

        with patch("corvidae_vision.get_dependency") as mock_dep:
            llm_plugin, client = _mock_llm()
            mock_dep.return_value = llm_plugin

            await plugin.view_image(str(png))

        call_args = client.chat.call_args
        messages = call_args[1]["messages"]
        text_block = [b for b in messages[0]["content"] if b.get("type") == "text"][0]
        assert text_block["text"] == DEFAULT_PROMPT

    @pytest.mark.asyncio
    async def test_custom_prompt_used(
        self, plugin: VisionPlugin, tmp_path: Path
    ):
        png = tmp_path / "test.png"
        png.write_bytes(TINY_PNG)

        with patch("corvidae_vision.get_dependency") as mock_dep:
            llm_plugin, client = _mock_llm()
            mock_dep.return_value = llm_plugin

            await plugin.view_image(str(png), prompt="What color is this?")

        call_args = client.chat.call_args
        messages = call_args[1]["messages"]
        text_block = [b for b in messages[0]["content"] if b.get("type") == "text"][0]
        assert text_block["text"] == "What color is this?"

    @pytest.mark.asyncio
    async def test_empty_llm_response(
        self, plugin: VisionPlugin, tmp_path: Path
    ):
        png = tmp_path / "test.png"
        png.write_bytes(TINY_PNG)

        with patch("corvidae_vision.get_dependency") as mock_dep:
            llm_plugin, _ = _mock_llm(response_text="")
            mock_dep.return_value = llm_plugin

            result = await plugin.view_image(str(png))

        assert "empty response" in result

    @pytest.mark.asyncio
    async def test_llm_call_failure(
        self, plugin: VisionPlugin, tmp_path: Path
    ):
        png = tmp_path / "test.png"
        png.write_bytes(TINY_PNG)

        with patch("corvidae_vision.get_dependency") as mock_dep:
            llm_plugin = MagicMock()
            client = MagicMock()
            client.chat = AsyncMock(side_effect=RuntimeError("connection refused"))
            llm_plugin.get_client.return_value = client
            mock_dep.return_value = llm_plugin

            result = await plugin.view_image(str(png))

        assert "LLM call failed" in result
        assert "connection refused" in result


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_missing_path(self, plugin: VisionPlugin):
        result = await plugin.view_image("")
        assert "required" in result

    def test_all_supported_extensions(self):
        """The extension map covers the formats we claim to support."""
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
            assert ext in IMAGE_EXTENSIONS
            assert IMAGE_EXTENSIONS[ext].startswith("image/")


class TestReviewFixes:
    """Tests for LLM review findings."""

    @pytest.mark.asyncio
    async def test_empty_choices_list_returns_error(self, plugin: VisionPlugin, tmp_path: Path):
        """Bug #1: {"choices": []} must not raise IndexError."""
        png = tmp_path / "img.png"
        png.write_bytes(TINY_PNG)

        llm_plugin, _ = _mock_llm()
        # Override with empty choices
        llm_plugin.get_client.return_value.chat = AsyncMock(
            return_value={"choices": []}
        )

        with patch("corvidae_vision.get_dependency") as mock_get:
            mock_get.return_value = llm_plugin
            result = await plugin.view_image(str(png))

        assert "empty response" in result

    @pytest.mark.asyncio
    async def test_get_client_failure_returns_error(self, plugin: VisionPlugin, tmp_path: Path):
        """get_client raising should return an error string, not crash."""
        png = tmp_path / "img.png"
        png.write_bytes(TINY_PNG)

        llm_plugin = MagicMock()
        llm_plugin.get_client.side_effect = RuntimeError("no client")

        with patch("corvidae_vision.get_dependency") as mock_get:
            mock_get.return_value = llm_plugin
            result = await plugin.view_image(str(png))

        assert "cannot access LLM client" in result

    def test_register_llm_stages(self, plugin: VisionPlugin):
        """register_llm_stages should map 'vision' to 'chat'."""
        stages = {}
        plugin.register_llm_stages(stages)
        assert stages["vision"] == "chat"

    @pytest.mark.asyncio
    async def test_correct_stage_name_passed(self, plugin: VisionPlugin, tmp_path: Path):
        """get_client should be called with 'vision', not 'snap'."""
        png = tmp_path / "img.png"
        png.write_bytes(TINY_PNG)

        llm_plugin, client = _mock_llm()

        with patch("corvidae_vision.get_dependency") as mock_get:
            mock_get.return_value = llm_plugin
            await plugin.view_image(str(png))

        llm_plugin.get_client.assert_called_with("vision")
