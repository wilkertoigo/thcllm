import argparse
from ..core.client import THCClient


def register(subparsers):
    subparsers.add_parser("quota", help="Mostra as quotas de uso")


def run(args, config):
    client = THCClient(config)
    result = client.get_quota()
    quotas = result.get("quotas", [])
    
    print("model                    | limit | rpd   | rpm")
    print("-" * 50)
    for q in quotas:
        model = q.get("model", "")
        limit = q.get("limit", 0)
        rpd = q.get("rpd", 0)
        rpm = q.get("rpm", 0)
        print(f"{model:26} | {limit:6} | {rpd:6} | {rpm:5}")