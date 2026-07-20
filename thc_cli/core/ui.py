import time
import difflib
from contextlib import contextmanager

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.status import Status
from rich.syntax import Syntax

console = Console()


def print_banner():
    banner = Panel(
        "🤖 THC CLI\n\nCliente de terminal para o THC LLM Router",
        title="[bold cyan]🤖 THC CLI[/bold cyan]",
        border_style="cyan",
        expand=False,
    )
    console.print(banner)


def print_assistant_reply(text: str, typewriter: bool = True):
    if typewriter:
        for char in text:
            console.print(char, end="", no_wrap=True)
            console.file.flush()
            time.sleep(0.008)
        console.print()
    else:
        md = Markdown(text)
        console.print(md)


@contextmanager
def print_thinking_spinner():
    with console.status("[cyan]▣ Pensando...[/cyan]", spinner="dots"):
        yield


def print_diff(original: str, new: str, filename: str):
    diff_lines = difflib.unified_diff(
        original.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    )
    diff_text = "".join(diff_lines)
    if diff_text:
        syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=False)
        console.print(syntax)


def confirm(prompt: str) -> bool:
    while True:
        choice = console.input(f"[yellow]{prompt} [s/N]: [/yellow]").strip().lower()
        if choice == "s":
            return True
        elif choice == "n" or choice == "":
            return False
        else:
            console.print("[red]Responda 's' para sim ou 'n' para não.[/red]")