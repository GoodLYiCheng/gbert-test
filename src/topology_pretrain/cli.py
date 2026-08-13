from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from .training import evaluate, export_encoder, load_checkpoint, train


def _config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="topology_pretrain")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "prepare"):
        p = sub.add_parser(name); p.add_argument("--config", required=True)
    p = sub.add_parser("evaluate"); p.add_argument("--run-dir", required=True)
    p = sub.add_parser("export"); p.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        c = _config(args.config); print(json.dumps({"status": "ready", "config": c}, indent=2)); return
    if args.command == "train":
        c = _config(args.config); run_dir = Path(c["output_dir"]) / datetime.now().strftime("stage1_%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True); (run_dir / "run_manifest.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
        print(json.dumps(train(c, run_dir), indent=2)); return
    run = Path(args.run_dir); checkpoint = run / "best.pt"
    if args.command == "export":
        export_encoder(checkpoint, run / "topology_encoder.pt"); print(run / "topology_encoder.pt"); return
    model, head, c, _ = load_checkpoint(checkpoint)
    metrics = {"id_test": evaluate(model, head, c, "id_test", c["id_test_size"]), "ood_test": evaluate(model, head, c, "ood_test", c["ood_test_size"], ood=True)}
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = ["# Stage 1 Evaluation Report", "", "Evaluation basis: fixed deterministic synthetic ID/OOD splits; this is an auxiliary pretraining diagnostic, not a real-task metric.", ""]
    for name, value in metrics.items():
        iso = value["isomorphic_cosine"]
        lines.extend([f"## {name}", "", f"- samples: {value['n']}", f"- Pearson / Spearman: {value['pearson']:.4f} / {value['spearman']:.4f}",
                      f"- MAE / Huber: {value['mae']:.4f} / {value['huber']:.4f}", f"- ranking accuracy: {value['ranking_accuracy']:.4f}",
                      f"- isomorphic cosine mean / min: {iso['mean']:.6f} / {iso['min']:.6f}", ""])
    (run / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
