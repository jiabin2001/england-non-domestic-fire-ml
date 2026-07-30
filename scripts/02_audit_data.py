from fireml.acquire import acquire_and_cache
from fireml.audit import run_audit


if __name__ == "__main__":
    acquire_and_cache()
    receipt = run_audit()
    print(f"Feasible: {receipt['feasible']}; logical rows: {receipt['logical_rows']:,}")

