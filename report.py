from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COPY_GAPS = (8, 32, 64, 128)


@dataclass
class ExperimentResult:
    name: str
    model: str
    task: str
    axis_value: int | None
    summary: dict[str, Any]
    evaluation: dict[str, Any] | None
    latest: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_latest_csv_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def parse_experiment_name(name: str) -> tuple[str, str, int | None]:
    match = re.match(r"(.+)_(copy)_gap(\d+)$", name)
    if match:
        return match.group(1), match.group(2), int(match.group(3))
    match = re.match(r"(.+)_(associative_recall)_pairs(\d+)$", name)
    if match:
        return match.group(1), match.group(2), int(match.group(3))
    match = re.match(r"(.+)_(brackets)_depth(\d+)$", name)
    if match:
        return match.group(1), match.group(2), int(match.group(3))
    match = re.match(r"(.+)_(text)_char$", name)
    if match:
        return match.group(1), match.group(2), None
    return name, "unknown", None


def load_results(metrics_dir: Path) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    for summary_path in sorted(metrics_dir.glob("*.json")):
        if summary_path.name.endswith("_evaluation.json"):
            continue
        summary = read_json(summary_path)
        name = str(summary.get("experiment", summary_path.stem))
        config = summary.get("config", {})
        model, task, axis_value = parse_experiment_name(name)
        model = str(config.get("model", model))
        task = str(config.get("task", task))
        if task == "copy":
            axis_value = int(config.get("gap_length", axis_value or 0))
        elif task == "associative_recall":
            axis_value = int(config.get("num_pairs", axis_value or 0))
        elif task == "brackets":
            axis_value = int(config.get("max_depth", axis_value or 0))
        evaluation_path = metrics_dir / f"{name}_test_evaluation.json"
        evaluation = read_json(evaluation_path) if evaluation_path.exists() else None
        latest = read_latest_csv_row(metrics_dir / f"{name}.csv")
        if not latest and summary.get("history"):
            latest = summary["history"][-1]
        results.append(ExperimentResult(name, model, task, axis_value, summary, evaluation, latest))
    return results


def model_label(model: str) -> str:
    labels = {
        "gru": "GRU",
        "transformer": "Transformer",
        "local_gravity": "Gravitation lokal",
        "nested_gravity": "Gravitation + Nesting",
        "nested_gravity_repulsion": "Gravitation + Nesting + Repulsion",
        "nesting_only": "Nesting ohne lokale Gravitation",
        "gru_only": "Nested-GRU-Ablation",
    }
    return labels.get(model, model.replace("_", " ").title())


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_value(result: ExperimentResult, *keys: str) -> float | None:
    sources = [result.evaluation or {}, result.latest, result.summary]
    for source in sources:
        for key in keys:
            value = as_float(source.get(key))
            if value is not None:
                return value
    return None


def format_percent(value: float | None) -> str:
    if value is None:
        return "offen"
    return f"{value * 100:.1f}%"


def format_number(value: float | None) -> str:
    if value is None:
        return "offen"
    return f"{value:.3g}"


def comparison_rows(results: list[ExperimentResult]) -> list[tuple[str, list[str]]]:
    models = ["gru", "transformer", "local_gravity", "nested_gravity"]
    rows: list[tuple[str, list[str]]] = []
    for model in models:
        cells: list[str] = []
        for gap in COPY_GAPS:
            result = find_result(results, model, "copy", gap)
            value = metric_value(result, f"accuracy_at_gap_{gap}", f"validation_accuracy_at_gap_{gap}", "accuracy") if result else None
            cells.append(format_percent(value))
        recall = best_task_metric(results, model, "associative_recall", "accuracy_num_pairs_", "validation_accuracy_num_pairs_", "accuracy")
        brackets = best_bracket_metric(results, model, 8)
        cells.extend([format_percent(recall), format_percent(brackets)])
        rows.append((model_label(model), cells))
    return rows


def find_result(results: list[ExperimentResult], model: str, task: str, axis_value: int | None = None) -> ExperimentResult | None:
    for result in results:
        if result.model == model and result.task == task and (axis_value is None or result.axis_value == axis_value):
            return result
    return None


def best_task_metric(results: list[ExperimentResult], model: str, task: str, *key_prefixes: str) -> float | None:
    values: list[float] = []
    for result in results:
        if result.model != model or result.task != task:
            continue
        keys = []
        for prefix in key_prefixes:
            if prefix.endswith("_") and result.axis_value is not None:
                keys.append(f"{prefix}{result.axis_value}")
            else:
                keys.append(prefix)
        value = metric_value(result, *keys)
        if value is not None:
            values.append(value)
    return max(values) if values else None


def best_bracket_metric(results: list[ExperimentResult], model: str, depth: int) -> float | None:
    result = find_result(results, model, "brackets", depth)
    if result is None:
        return None
    return metric_value(result, "full_sequence_accuracy", "validation_full_sequence_accuracy", "token_accuracy", "validation_token_accuracy", "accuracy")


def nested_diagnostics(results: list[ExperimentResult]) -> dict[str, str]:
    nested = [result for result in results if result.model == "nested_gravity"]
    best_copy = max(
        nested,
        key=lambda item: metric_value(item, f"accuracy_at_gap_{item.axis_value}", "accuracy") or -1.0,
        default=None,
    )
    if best_copy is None:
        return {
            "long_term": "offen",
            "centers": "offen",
            "causal": "offen",
            "collapse": "offen",
            "runtime": "offen",
        }
    accuracy = metric_value(best_copy, f"accuracy_at_gap_{best_copy.axis_value}", "accuracy")
    entropy = metric_value(best_copy, "center_entropy", "validation_center_entropy")
    effective_centers = metric_value(best_copy, "effective_num_centers", "validation_effective_num_centers")
    local_force = metric_value(best_copy, "mean_local_force_norm", "validation_mean_local_force_norm")
    nesting_force = metric_value(best_copy, "mean_nesting_force_norm", "validation_mean_nesting_force_norm")
    return {
        "long_term": f"beste Copy-Messung: Gap {best_copy.axis_value}, {format_percent(accuracy)}",
        "centers": f"Entropie {format_number(entropy)}, effektive Zentren {format_number(effective_centers)}",
        "causal": "Plot vorhanden, wenn causality_check.png erzeugt wurde",
        "collapse": f"Nesting-Kraft {format_number(nesting_force)} vs. lokale Kraft {format_number(local_force)}",
        "runtime": "noch nicht gemessen; Laufzeitspalten werden ergänzt, sobald Trainingsläufe Timing speichern",
    }


def relative_path(from_file: Path, target: Path) -> str:
    return html.escape(os.path.relpath(target, start=from_file.parent).replace(os.sep, "/"))


def render_report(output_root: Path, report_path: Path, results: list[ExperimentResult]) -> str:
    rows = comparison_rows(results)
    diagnostics = nested_diagnostics(results)
    plots = sorted((output_root / "plots").glob("*.png"))
    generated = html.escape(str(report_path))
    metric_count = len(results)
    table_rows = "\n".join(
        "<tr><th>{}</th>{}</tr>".format(
            html.escape(label),
            "".join(f"<td>{html.escape(cell)}</td>" for cell in cells),
        )
        for label, cells in rows
    )
    plot_cards = "\n".join(
        f'<figure><img src="{relative_path(report_path, plot)}" alt="{html.escape(plot.stem)}"><figcaption>{html.escape(plot.stem.replace("_", " "))}</figcaption></figure>'
        for plot in plots
    ) or '<p class="muted">Noch keine PNG-Artefakte unter outputs/plots gefunden.</p>'
    result_items = "\n".join(
        f"<li><strong>{html.escape(result.name)}</strong>: {html.escape(result.task)}, {html.escape(model_label(result.model))}</li>"
        for result in results
    ) or '<li class="muted">Noch keine Metrikdateien gefunden.</li>'
    return f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Nested Gravity LLM Report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17211d;
      --muted: #5d6963;
      --paper: #fbfaf5;
      --panel: #ffffff;
      --line: #d8ddd2;
      --accent: #0f766e;
      --accent-soft: #d9f0eb;
      --warn: #8a4b12;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(135deg, #fbfaf5 0%, #edf4ed 48%, #f7efe2 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      line-height: 1.5;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 40px 24px 56px; }}
    header {{ margin-bottom: 28px; }}
    h1 {{ font-size: clamp(2rem, 5vw, 4.5rem); line-height: 0.95; margin: 0 0 14px; max-width: 900px; }}
    h2 {{ font-size: 1.45rem; margin: 0 0 14px; }}
    p {{ margin: 0 0 12px; }}
    .lede {{ max-width: 780px; color: var(--muted); font-size: 1.08rem; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
    .pill {{ background: var(--accent-soft); color: #10443f; border: 1px solid #b8ded7; border-radius: 999px; padding: 5px 10px; font: 0.9rem system-ui, sans-serif; }}
    section {{ margin-top: 24px; background: rgba(255, 255, 255, 0.78); border: 1px solid var(--line); border-radius: 8px; padding: 20px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 760px; font-family: system-ui, sans-serif; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 12px 10px; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    thead th {{ background: #f4f7ef; color: #2d3832; font-weight: 650; }}
    tbody th {{ font-weight: 650; }}
    .question-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .question {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; min-height: 120px; }}
    .question b {{ display: block; margin-bottom: 6px; font-family: system-ui, sans-serif; }}
    .question span {{ color: var(--muted); }}
    blockquote {{ border-left: 4px solid var(--accent); margin: 0; padding: 8px 0 8px 16px; color: #24332d; font-size: 1.1rem; }}
    .plots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }}
    figure {{ margin: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px; }}
    img {{ display: block; width: 100%; height: auto; border-radius: 4px; background: #fff; }}
    figcaption {{ margin-top: 8px; color: var(--muted); font: 0.9rem system-ui, sans-serif; }}
    ul {{ margin: 0; padding-left: 20px; }}
    .muted {{ color: var(--muted); }}
    .note {{ color: var(--warn); font-family: system-ui, sans-serif; font-size: 0.95rem; }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Wissenschaftlicher Vergleich</h1>
    <p class="lede">Dieser statische Report sammelt vorhandene Metriken und Artefakte aus <code>outputs/</code>, damit die zentrale Hypothese und die nächsten Ablationen auf einen Blick sichtbar sind.</p>
    <div class="meta"><span class="pill">{metric_count} Experimente gelesen</span><span class="pill">Report: {generated}</span></div>
  </header>

  <section>
    <h2>Erste aussagekräftige Tabelle</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Modell</th><th>Copy-8</th><th>Copy-32</th><th>Copy-64</th><th>Copy-128</th><th>Recall</th><th>Bracket-Depth-8</th></tr></thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
    <p class="note">"offen" bedeutet: Für diese Modell-Aufgaben-Kombination liegt noch keine passende Metrikdatei vor.</p>
  </section>

  <section>
    <h2>Leitfragen</h2>
    <div class="question-grid">
      <div class="question"><b>1. Lernt es Langzeitabhängigkeiten?</b><span>{html.escape(diagnostics["long_term"])}</span></div>
      <div class="question"><b>2. Speichert es Gravitationszentren?</b><span>{html.escape(diagnostics["centers"])}</span></div>
      <div class="question"><b>3. Bleibt es kausal?</b><span>{html.escape(diagnostics["causal"])}</span></div>
      <div class="question"><b>4. Verhindert Nesting Kollaps?</b><span>{html.escape(diagnostics["collapse"])}</span></div>
      <div class="question"><b>5. Wie wächst die Laufzeit?</b><span>{html.escape(diagnostics["runtime"])}</span></div>
    </div>
  </section>

  <section>
    <h2>Hypothese</h2>
    <blockquote>Lokale Gravitation allein kann Kurzzeitabhängigkeiten erfassen, während verschachtelte Gravitationszentren die effiziente Speicherung längerfristiger Information ermöglichen.</blockquote>
  </section>

  <section>
    <h2>Artefakte</h2>
    <div class="plots">{plot_cards}</div>
  </section>

  <section>
    <h2>Gelesene Experimente</h2>
    <ul>{result_items}</ul>
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    report_path = Path(args.report) if args.report else output_root / "report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results = load_results(output_root / "metrics")
    report_path.write_text(render_report(output_root, report_path, results), encoding="utf-8")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()