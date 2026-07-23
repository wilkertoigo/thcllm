import argparse
import importlib
import pkgutil
import sys

from .core import load_config


def main():
    parser = argparse.ArgumentParser(prog="thc", description="CLI para THC LLM")
    subparsers = parser.add_subparsers(dest="command")
    
    from . import commands as commands_pkg
    for _, module_name, _ in pkgutil.iter_modules(commands_pkg.__path__, commands_pkg.__name__ + "."):
        module = importlib.import_module(module_name)
        if hasattr(module, "register"):
            module.register(subparsers)
    
    parser.add_argument("--skill", help="Skill a ativar (nome da skill em ~/.thc/skills/)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    config = load_config()
    
    if args.skill:
        from .core.skills import SkillStore
        store = SkillStore()
        skill = store.get(args.skill)
        if skill is None:
            print(f"Skill não encontrada: {args.skill}", file=sys.stderr)
            sys.exit(1)
        if not hasattr(args, "skill_obj"):
            args.skill_obj = None
        args.skill_obj = skill
    
    from . import commands as commands_pkg
    for _, module_name, _ in pkgutil.iter_modules(commands_pkg.__path__, commands_pkg.__name__ + "."):
        module = importlib.import_module(module_name)
        if hasattr(module, "run") and module_name.endswith(f".{args.command}"):
            module.run(args, config)
            return
    
    print(f"Comando desconhecido: {args.command}", file=sys.stderr)
    sys.exit(1)