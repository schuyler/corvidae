# sherman/tools/__init__.py

from sherman.hooks import hookimpl
from sherman.tool import Tool

from sherman.tools.shell import shell
from sherman.tools.files import read_file, write_file
from sherman.tools.web import web_fetch


class CoreToolsPlugin:
    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        tool_registry.extend([
            Tool.from_function(shell),
            Tool.from_function(read_file),
            Tool.from_function(write_file),
            Tool.from_function(web_fetch),
        ])
