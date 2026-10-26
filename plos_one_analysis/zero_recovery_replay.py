#!/usr/bin/env python3
"""Replay zero-recovery AUPR values on the immune reference-panel labels.

Under HPN-DREAM and BEELINE no archived method recovers a reference edge, so
their source AUPR values depend only on where reference edges fall among the
unscored candidates. This script asks what the source AUPR does for prediction
lists that are guaranteed to recover nothing.

``export`` (repository only; needs the source h5ad and the
``bio_mech_interp`` environment with scanpy) runs the original
``network_inference`` score-evaluation pipeline once, captures its flattened
reference label vectors, and stores the positive indices and gene universe in
``inputs/immune_reference_panel/``. ``replay`` (portable) rebuilds those
vectors and scores seeded synthetic lists with the observed list lengths,
using the source formula.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PANEL = HERE / "inputs" / "immune_reference_panel"
ZERO_RECOVERY_REFERENCES = ["hpn_dream", "beeline_gsd"]


def source_full_ranking_aupr(scores: np.ndarray, labels: np.ndarray) -> float:
    """network_inference/src/evaluation/metrics.py::aupr, verbatim logic."""
    order = np.argsort(-scores)
    labels = labels[order]
    total_pos = int(labels.sum())
    if total_pos == 0:
        return 0.0
    labels = labels.astype(np.int64, copy=False)
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    precisions = tp / (tp + fp)
    recalls = tp / total_pos
    try:
        area = np.trapezoid(precisions, recalls)
    except AttributeError:
        area = np.trapz(precisions, recalls)
    return float(area)


def flat_index(source: np.ndarray, target: np.ndarray, n_genes: int) -> np.ndarray:
    """Row-major position after removing the diagonal (the source candidate mask)."""
    return source * (n_genes - 1) + np.where(target < source, target, target - 1)


def read_var_names_only(path, backed=None):
    """Minimal stand-in for scanpy.read_h5ad exposing var_names only."""
    import h5py
    from types import SimpleNamespace

    with h5py.File(path, "r") as handle:
        var = handle["var"]
        column = var.attrs["_index"]
        names = [value.decode() if isinstance(value, bytes) else str(value) for value in var[column][:]]
    return SimpleNamespace(var_names=pd.Index(names), file=None)


def export(repository: Path) -> None:
    sys.path.insert(0, str(repository))
    from network_inference.src.evaluation import metrics as metrics_module
    from network_inference.src.evaluation import score_edges
    from network_inference.src.utils.config import load_config, resolve_paths

    config_path = repository / "network_inference" / "configs" / "score_eval_grn_baselines_immune.yaml"
    config = resolve_paths(load_config(str(config_path)), str(config_path))
    eval_config = config["score_eval"]
    if eval_config.get("mapping"):
        raise RuntimeError("adata alias columns would require the full AnnData object")
    # The pipeline only reads var_names when no adata alias columns are
    # configured; reading them with h5py avoids an AnnData version mismatch
    # in unrelated .uns entries of the archived file.
    score_edges.sc.read_h5ad = read_var_names_only
    genes, _ = score_edges._load_gene_names(config["paths"], eval_config.get("mapping"))
    genes = np.asarray(genes)
    PANEL.mkdir(parents=True, exist_ok=True)
    captured: list[tuple[np.ndarray, np.ndarray]] = []
    original = metrics_module.aupr

    def capture(scores: np.ndarray, labels: np.ndarray) -> float:
        captured.append((np.flatnonzero(labels), np.flatnonzero(scores)))
        return original(scores, labels)

    with tempfile.TemporaryDirectory() as scratch:
        dummy = Path(scratch) / "dummy_edges.tsv"
        pd.DataFrame({"source": [genes[0]], "target": [genes[1]], "score": [1.0]}).to_csv(dummy, sep="\t", index=False)
        eval_config["methods"] = [{"name": "dummy", "edges_tsv": str(dummy), "source_col": "source",
                                   "target_col": "target", "score_col": "score"}]
        eval_config["output_path"] = str(Path(scratch) / "dummy.json")
        metrics_module.aupr = capture
        try:
            results = score_edges.evaluate_score_edges(config)
        finally:
            metrics_module.aupr = original
    scored = [row for row in results if row["aupr"] is not None]
    if len(scored) != len(captured):
        raise RuntimeError("captured label vectors do not match scored references")
    archived = pd.read_csv(HERE / "inputs" / "score_eval_grn_baselines_immune.csv")
    archived_true = archived.groupby("reference")["true_edges"].first()
    gene_set = set(genes)
    self_pairs = {}
    for reference in eval_config["references"]:
        # Same loading and normalization as evaluate_score_edges; the source
        # reports true_edges including self-pairs, which its candidate mask
        # (diagonal removed) excludes from the AUPR label vector.
        mapping = dict(reference.get("mapping", {})) if reference.get("mapping") else None
        if mapping and mapping.get("symbol_map_tsv"):
            mapping["symbol_map_tsv"] = score_edges._resolve_optional_path(mapping["symbol_map_tsv"], config)
        _, alias_map = score_edges._load_gene_names(config["paths"], eval_config.get("mapping"))
        for path in score_edges.collect_symbol_map_paths(mapping):
            alias_map = score_edges.extend_alias_map_from_symbol_map(alias_map, path, None)
        edges = score_edges._load_edge_table(score_edges._resolve_optional_path(reference["edges_tsv"], config),
                                             reference.get("source_col", "source"),
                                             reference.get("target_col", "target"), score_col=None)
        edges = score_edges._normalize_edges(edges, alias_map, mapping)
        edges = edges[edges["source"].isin(gene_set) & edges["target"].isin(gene_set)].drop_duplicates()
        self_pairs[reference["name"]] = int((edges["source"] == edges["target"]).sum())
    n_genes = len(genes)
    rows, checks = [], {"n_genes": n_genes, "n_candidates": n_genes * (n_genes - 1), "references": {}}
    for row, (positives, predicted) in zip(scored, captured):
        name = row["reference"]
        expected_dummy = int(flat_index(np.array([0]), np.array([1]), n_genes)[0])
        checks["references"][name] = {
            "label_vector_positives": int(len(positives)),
            "archived_true_edges": int(archived_true[name]),
            "reference_self_pairs_in_gene_universe": self_pairs[name],
            "dummy_flat_index_matches": bool(predicted.tolist() == [expected_dummy]),
            "pipeline_dummy_aupr": row["aupr"],
        }
        if len(positives) + self_pairs[name] != archived_true[name]:
            raise RuntimeError(f"{name}: label positives plus self-pairs differ from archived true_edges")
        rows.extend({"reference": name, "flat_index": int(index)} for index in positives)
    pd.DataFrame(rows).to_csv(PANEL / "label_indices.csv", index=False)
    (PANEL / "gene_universe.txt").write_text("\n".join(genes) + "\n")
    (PANEL / "export_check.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


def load_panel() -> tuple[int, dict[str, np.ndarray]]:
    genes = (PANEL / "gene_universe.txt").read_text().split()
    labels = pd.read_csv(PANEL / "label_indices.csv")
    return len(genes), {name: group["flat_index"].to_numpy() for name, group in labels.groupby("reference")}


def distinct_scored_pairs(archived: pd.DataFrame) -> dict[str, int]:
    """Distinct regulator-target pairs each list scores in the AUPR matrix.

    ``pred_edges`` counts listed rows, which can repeat a pair with different
    scores; the score matrix keeps one entry per pair. Where a list recovers
    DoRothEA/TRRUST edges, recovered edges divided by precision gives the
    distinct-pair count; otherwise the listed length is used.
    """
    rows = archived[archived["reference"] == "dorothea_trrust_union_immune"].set_index("method")
    pairs = {}
    for method, row in rows.iterrows():
        recovered = round(row["recall"] * row["true_edges"])
        pairs[method] = int(round(recovered / row["precision"])) if row["precision"] > 0 else int(row["pred_edges"])
    return pairs


def label_vector(n_candidates: int, positives: np.ndarray) -> np.ndarray:
    labels = np.zeros(n_candidates, dtype=np.int8)
    labels[positives] = 1
    return labels


def zero_recovery_scores(n_candidates: int, n_predicted: int, excluded: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    """A float32 candidate vector whose scored entries avoid every excluded index."""
    chosen = np.empty(0, dtype=np.int64)
    while len(chosen) < n_predicted:
        draw = rng.integers(0, n_candidates, size=2 * n_predicted)
        draw = np.setdiff1d(np.unique(draw), excluded, assume_unique=False)
        chosen = np.unique(np.concatenate([chosen, draw]))
    chosen = rng.permutation(chosen)[:n_predicted]
    scores = np.zeros(n_candidates, dtype=np.float32)
    scores[chosen] = rng.uniform(0.01, 1.0, n_predicted).astype(np.float32)
    return scores


def replay(replicates: int, seed: int, output_dir: Path) -> None:
    n_genes, positives = load_panel()
    n_candidates = n_genes * (n_genes - 1)
    check = json.loads((PANEL / "export_check.json").read_text())
    archived = pd.read_csv(HERE / "inputs" / "score_eval_grn_baselines_immune.csv")
    lengths = distinct_scored_pairs(archived)
    excluded = np.unique(np.concatenate([positives[name] for name in ZERO_RECOVERY_REFERENCES]))
    labels = {name: label_vector(n_candidates, positives[name]) for name in positives}
    dummy = np.zeros(n_candidates, dtype=np.float32)
    dummy[int(flat_index(np.array([0]), np.array([1]), n_genes)[0])] = 1.0
    fidelity = {name: {"replay": source_full_ranking_aupr(dummy, labels[name]),
                       "pipeline": check["references"][name]["pipeline_dummy_aupr"]}
                for name in labels}
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        for method, length in sorted(lengths.items()):
            scores = zero_recovery_scores(n_candidates, int(length), excluded, rng)
            for name in ZERO_RECOVERY_REFERENCES:
                if labels[name][scores > 0].any():
                    raise RuntimeError("synthetic list recovered a reference edge")
                rows.append({"replicate": replicate, "list_length_from": method,
                             "n_predicted": int(length), "reference": name,
                             "base_rate": len(positives[name]) / n_candidates,
                             "aupr": source_full_ranking_aupr(scores, labels[name])})
    draws = pd.DataFrame(rows)
    draws["aupr_over_base_rate"] = draws["aupr"] / draws["base_rate"]
    draws.to_csv(output_dir / "extended_zero_recovery_replay.csv", index=False)
    (output_dir / "extended_zero_recovery_fidelity.json").write_text(json.dumps(
        {"seed": seed, "replicates": replicates, "n_candidates": n_candidates,
         "pipeline_fidelity_single_edge_list": fidelity}, indent=2) + "\n")
    summarize(output_dir)


def summarize(output_dir: Path) -> None:
    """Compare replayed zero-recovery values and orderings with the archive."""
    draws = pd.read_csv(output_dir / "extended_zero_recovery_replay.csv")
    summary = json.loads((output_dir / "extended_zero_recovery_fidelity.json").read_text())
    for name, values in summary["pipeline_fidelity_single_edge_list"].items():
        if values["replay"] != values["pipeline"]:
            raise RuntimeError(f"{name}: replay differs from the source pipeline")
    archived = pd.read_csv(HERE / "inputs" / "score_eval_grn_baselines_immune.csv")
    observed = archived[archived["reference"].isin(ZERO_RECOVERY_REFERENCES)]
    summary["references"] = {}
    for name in ZERO_RECOVERY_REFERENCES:
        values = draws[draws["reference"] == name]
        within = values.groupby("replicate")["aupr"].agg(lambda v: v.max() - v.min())
        seen = observed[observed["reference"] == name]["aupr"]
        low, high = values["aupr"].min(), values["aupr"].max()
        summary["references"][name] = {
            "replay_min": float(low), "replay_max": float(high),
            "replay_median_within_replicate_spread": float(within.median()),
            "replay_within_replicate_spread_min": float(within.min()),
            "replay_within_replicate_spread_max": float(within.max()),
            "observed_min": float(seen.min()), "observed_max": float(seen.max()),
            "observed_spread": float(seen.max() - seen.min()),
            "observed_values_inside_replay_range": int(seen.between(low, high).sum()),
            "observed_values": int(len(seen)),
        }
    reversal_fractions = []
    for _, group in draws.groupby("replicate"):
        wide = group.pivot(index="list_length_from", columns="reference", values="aupr")
        flips = [np.sign(wide.loc[a, "beeline_gsd"] - wide.loc[b, "beeline_gsd"])
                 * np.sign(wide.loc[a, "hpn_dream"] - wide.loc[b, "hpn_dream"]) < 0
                 for a, b in itertools.combinations(wide.index, 2)]
        reversal_fractions.append(float(np.mean(flips)))
    rows = pd.read_csv(HERE / "inputs" / "reference_shift_pairwise_rows.csv")
    archived_transition = rows[(rows["reference_from"] == "beeline_gsd") & (rows["reference_to"] == "hpn_dream")]
    summary["beeline_hpn_reversal_fraction"] = {
        "replay_mean": float(np.mean(reversal_fractions)), "replay_min": float(np.min(reversal_fractions)),
        "replay_max": float(np.max(reversal_fractions)), "replay_per_replicate": reversal_fractions,
        "archived_k": int(archived_transition["reversal"].sum()), "archived_n": int(len(archived_transition)),
        "archived_fraction": float(archived_transition["reversal"].mean()),
    }
    (output_dir / "extended_zero_recovery_replay_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exporter = sub.add_parser("export")
    exporter.add_argument("--repository", type=Path, default=HERE.parents[1])
    replayer = sub.add_parser("replay")
    replayer.add_argument("--replicates", type=int, default=10)
    replayer.add_argument("--seed", type=int, default=42)
    replayer.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    summarizer = sub.add_parser("summarize")
    summarizer.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    arguments = parser.parse_args()
    if arguments.command == "export":
        export(arguments.repository)
    elif arguments.command == "summarize":
        summarize(arguments.output_dir)
    else:
        replay(arguments.replicates, arguments.seed, arguments.output_dir)


if __name__ == "__main__":
    main()
