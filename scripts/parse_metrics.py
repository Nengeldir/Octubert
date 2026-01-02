"""Parse and tabulate metrics JSON files for paper reporting.

Usage:
    python scripts/parse_metrics.py logs/model1/metrics/metrics.json logs/model2/metrics/metrics.json

Output: Markdown table to stdout (copy to paper).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


# Metric-specific "good" ranges and optimization direction
METRIC_GUIDELINES = {
    "pitch_class_js": {"lower_better": True, "good_max": 0.05, "acceptable_max": 0.15},
    "duration_js": {"lower_better": True, "good_max": 0.10, "acceptable_max": 0.25},
    "consistency_pitch": {"lower_better": False, "good_min": 0.75},
    "consistency_duration": {"lower_better": False, "good_min": 0.75},
    "variance_pitch": {"lower_better": False, "range": (0.7, 0.9)},
    "variance_duration": {"lower_better": False, "range": (0.7, 0.9)},
    "self_similarity_mean": {"lower_better": False, "range": (0.3, 0.7)},
    "self_similarity_gap_to_ref": {"lower_better": True, "good_max": 0.05, "acceptable_max": 0.05},
    "key_consistency": {"lower_better": False, "good_min": 0.5},
    "key_consistency_gap_to_ref": {"lower_better": True, "good_max": 0.05, "acceptable_max": 0.05},
    "key_agreement_with_ref": {"lower_better": False, "good_min": 0.8},
    "bar_pitch_var_mean": {"lower_better": False, "range": (50, 200)},
    "bar_onset_density_mean": {"lower_better": False, "range": (0.2, 0.6)},
    "phrase_similarity_mean": {"lower_better": False, "range": (0.4, 0.7)},
    "phrase_similarity_gap_to_ref": {"lower_better": True, "good_max": 0.05, "acceptable_max": 0.05},
    "harmonic_flux_mean": {"lower_better": False, "range": (0.2, 0.4)},
    "harmonic_flux_gap_to_ref": {"lower_better": True, "good_max": 0.05, "acceptable_max": 0.05},
    "structural_fad": {"lower_better": True, "good_max": 10, "acceptable_max": 20},
    "masked_token_accuracy": {"lower_better": False, "good_min": 0.6, "acceptable_min": 0.4},
    "infill_rhythm_correlation": {"lower_better": False, "good_min": 0.6, "acceptable_min": 0.3},
    "masked_span_embedding_similarity": {"lower_better": False, "good_min": 0.7, "acceptable_min": 0.5},
}


def load_metrics(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_in_good_range(metric_name: str, value: float) -> bool:
    """Check if value is in the 'good' range for the metric."""
    if metric_name not in METRIC_GUIDELINES:
        return False
    
    guide = METRIC_GUIDELINES[metric_name]
    
    if "good_min" in guide:
        return value >= guide["good_min"]
    elif "good_max" in guide:
        return value <= guide["good_max"]
    elif "range" in guide:
        low, high = guide["range"]
        return low <= value <= high
    
    return False


def is_best_value(metric_name: str, value: float, all_values: List[float]) -> bool:
    """Determine if this value is the best among all values for this metric."""
    if not all([isinstance(v, (int, float)) for v in all_values]):
        return False
    
    if metric_name not in METRIC_GUIDELINES:
        return False
    
    guide = METRIC_GUIDELINES[metric_name]
    
    if guide.get("lower_better", False):
        return value == min(all_values)
    else:
        return value == max(all_values)


def format_with_ci(metric_name: str, metrics_dict: Dict, include_ci_bounds: bool = False) -> Tuple[str, bool]:
    """Format value with ±std notation (optionally with CI bounds) and check if in good range.
    
    Returns: (formatted_string, is_good)
    """
    # Try to get base value and std
    if metric_name not in metrics_dict:
        return "N/A", False
    
    base_value = metrics_dict[metric_name]
    std_key = f"{metric_name}_std"
    ci_key = f"{metric_name}_ci"
    
    if not isinstance(base_value, (int, float)):
        return str(base_value), False
    
    formatted = f"{base_value:.4f}"
    
    # Add ± std if available
    if std_key in metrics_dict:
        std_val = metrics_dict[std_key]
        if isinstance(std_val, (int, float)):
            formatted += f" ± {std_val:.4f}"
    
    # Add CI bounds if requested and available
    if include_ci_bounds and ci_key in metrics_dict:
        ci_val = metrics_dict[ci_key]
        if isinstance(ci_val, list) and len(ci_val) == 2:
            formatted += f" (95% CI [{ci_val[0]:.4f}, {ci_val[1]:.4f}])"
    
    is_good = is_in_good_range(metric_name, base_value)
    return formatted, is_good


def make_table(metrics_list: List[Dict], labels: List[str], include_ci_bounds: bool = False) -> str:
    # Collect all unique metric keys (exclude *_std, *_ci keys)
    all_keys = set()
    for m in metrics_list:
        for k in m.keys():
            if not k.endswith("_std") and not k.endswith("_ci"):
                all_keys.add(k)
    
    keys = sorted(all_keys)
    
    # Build markdown table
    header = "| Metric | " + " | ".join(labels) + " |"
    separator = "|--------|" + "|".join(["--------"] * len(labels)) + "|"
    
    rows = [header, separator]
    
    for key in keys:
        row_values = []
        all_base_values = []
        formatted_values = []
        is_good_values = []
        
        # First pass: collect all values and format them
        for m in metrics_list:
            formatted, is_good = format_with_ci(key, m, include_ci_bounds=include_ci_bounds)
            formatted_values.append(formatted)
            is_good_values.append(is_good)
            
            if key in m and isinstance(m[key], (int, float)):
                all_base_values.append(m[key])
            else:
                all_base_values.append(None)
        
        # Second pass: build row with highlighting and checkmarks
        for i, formatted in enumerate(formatted_values):
            cell = formatted
            
            # Add checkmark if good
            if is_good_values[i]:
                cell += " ✅"
            
            # Highlight best value in blue
            if all_base_values[i] is not None and is_best_value(key, all_base_values[i], [v for v in all_base_values if v is not None]):
                cell = f"<span style='color:blue'><b>{cell}</b></span>"
            
            row_values.append(cell)
        
        row = f"| {key} | " + " | ".join(row_values) + " |"
        rows.append(row)
    
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Parse metrics JSONs into comparison table")
    parser.add_argument("metrics_files", nargs="+", help="Paths to metrics.json files")
    parser.add_argument("--labels", nargs="+", help="Labels for each model (default: model1, model2, ...)")
    parser.add_argument("--output", "-o", help="Output markdown file path (default: print to stdout)")
    parser.add_argument("--include-ci", action="store_true", help="Include 95% CI bounds in output")
    args = parser.parse_args()
    
    if args.labels and len(args.labels) != len(args.metrics_files):
        print("Error: number of labels must match number of metrics files")
        return
    
    labels = args.labels or [f"model{i+1}" for i in range(len(args.metrics_files))]
    
    metrics_list = [load_metrics(f) for f in args.metrics_files]
    
    table_content = "# Metrics Comparison\n\n" + make_table(metrics_list, labels, include_ci_bounds=args.include_ci) + "\n"
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(table_content)
        print(f"✓ Metrics table saved to: {output_path}")
    else:
        print("\n" + table_content)


if __name__ == "__main__":
    main()
