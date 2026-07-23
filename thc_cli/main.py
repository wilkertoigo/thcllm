import argparse
import importlib
import pkgutil

from .core import load_config


def main():
    parser = argparse.ArgumentParser(prog="thc", description="CLI para THC LLM")
    subparsers = parser.add_subparsers(dest="command")
    
    from . import commands as commands_pkg
    for _, module_name, _ in pkgutil.iter_modules(commands_pkg.__path__, commands_pkg.__name__ + "."):
        module = importlib.import_module(module_name)
        if hasattr(module, "register"):
            module.register(subparsers)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    config = load_config()
    
    from . import commands as commands_pkg
    for _, module_name, _ in pkgutil.iter_modules(commands_pkg.__path__, commands_pkg.__name__ + "."):
        module = importlib.import_module(module_name)
        if hasattr(module, "run") and module_name.endswith(f".{args.command}"):
            module.run(args, config)
            return
    
    print(f"Comando desconhecido: {args.command}", file=__import__('sys').stderr)
    __import__('sys').exit(1)