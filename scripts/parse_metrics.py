"""Parse and tabulate metrics JSON files for paper reporting.

Usage:
    python scripts/parse_metrics.py logs/model1/metrics/metrics.json logs/model2/metrics/metrics.json

Output: Markdown table to stdout (copy to paper).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_metrics(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_value(val) -> str:
    if isinstance(val, float):
        if abs(val) < 0.001 or abs(val) > 1000:
            return f"{val:.2e}"
        return f"{val:.4f}"
    return str(val)


def make_table(metrics_list: List[Dict], labels: List[str]) -> str:
    # Collect all unique metric keys
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    
    keys = sorted(all_keys)
    
    # Build markdown table
    header = "| Metric | " + " | ".join(labels) + " |"
    separator = "|--------|" + "|".join(["--------"] * len(labels)) + "|"
    
    rows = [header, separator]
    for key in keys:
        vals = [format_value(m.get(key, 'N/A')) for m in metrics_list]
        row = f"| {key} | " + " | ".join(vals) + " |"
        rows.append(row)
    
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Parse metrics JSONs into comparison table")
    parser.add_argument("metrics_files", nargs="+", help="Paths to metrics.json files")
    parser.add_argument("--labels", nargs="+", help="Labels for each model (default: model1, model2, ...)")
    args = parser.parse_args()
    
    if args.labels and len(args.labels) != len(args.metrics_files):
        print("Error: number of labels must match number of metrics files")
        return
    
    labels = args.labels or [f"model{i+1}" for i in range(len(args.metrics_files))]
    
    metrics_list = [load_metrics(f) for f in args.metrics_files]
    
    print("\n# Metrics Comparison\n")
    print(make_table(metrics_list, labels))
    print()


if __name__ == "__main__":
    main()
