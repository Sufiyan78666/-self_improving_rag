"""
Evaluation reporter for the Self-Improving RAG system.

Formats evaluation metrics into human-readable tables and summaries.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


def generate_report_markdown(metrics: Dict[str, float], model_name: str = "Unknown") -> str:
    """
    Generate a markdown formatted evaluation report.

    Args:
        metrics: Dictionary of metric names and scores.
        model_name: Name of the evaluated model.

    Returns:
        str: Markdown report.
    """
    report = [
        f"## Evaluation Report: {model_name}",
        f"Generated at: {logging.Formatter().converter}",
        "",
        "| Metric | Score |",
        "| :--- | :--- |",
    ]
    
    for name, value in metrics.items():
        report.append(f"| {name.upper()} | {value:.4f} |")
        
    return "\n".join(report)


def print_report_summary(metrics: Dict[str, float]) -> None:
    """
    Print a concise summary of metrics to the console.
    """
    print("\n--- EVALUATION SUMMARY ---")
    for name, value in metrics.items():
        print(f"{name:15}: {value:.4f}")
    print("--------------------------\n")
