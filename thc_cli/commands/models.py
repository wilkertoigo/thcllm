import argparse
from ..core.client import THCClient


def register(subparsers):
    subparsers.add_parser("models", help="Lista os modelos disponíveis")


def run(args, config):
    client = THCClient(config)
    result = client.list_models()
    models = result.get("text_models", [])
    
    print("key                    | backend     | label                           | ativo | pago")
    print("-" * 80)
    for m in models:
        key = m.get("key", "")
        backend = m.get("backend", "")
        label = m.get("label", "")
        active = "sim" if m.get("active", False) else "não"
        paid = "sim" if m.get("paid", False) else "não"
        print(f"{key:24} | {backend:11} | {label:35} | {active:7} | {paid:5}")