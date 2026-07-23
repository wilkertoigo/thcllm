from .file_tools import FileReadTool, FileWriteTool, FileEditTool, GlobTool, GrepTool
from .bash_tool import BashTool
from .web_tools import WebFetchTool, WebSearchTool
from .todo_tool import TodoWriteTool
from .memory_tools import MemoryReadTool, MemoryWriteTool

ALL_TOOLS = [
    FileReadTool(), FileWriteTool(), FileEditTool(),
    GlobTool(), GrepTool(), BashTool(),
    WebFetchTool(), WebSearchTool(), TodoWriteTool(),
    MemoryReadTool(), MemoryWriteTool(),
]

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
DESTRUCTIVE_TOOLS = {"write_file", "str_replace", "bash"}
