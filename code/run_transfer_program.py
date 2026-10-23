from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "code"))

from forge_ym_transfer import (
    simulate_rewrite_slab,
    slab_metrics,
    su2_reflection_transfer_benchmark,
    temporal_gauge_invariance_error,
)

MASTER_SEED = 20260726


def slab_stage() -> pd.DataFrame:
    rows = []
    for lower_generation in (0, 1, 2, 3):
        replicates = 1 if lower_generation == 0 else 8
        for replicate in range(replicates):
            seed = MASTER_SEED + 100_000 * lower_generation + replicate
            slab = simulate_rewrite_slab(
                lower_generation=lower_generation,
                k=4,
                q=2,
                block_size=6,
                closure_mode="face_cycle",
                seed=seed,
            )
            row = slab_metrics(slab)
            row["replicate"] = replicate
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(
        ["lower_generation", "replicate"]
    )
    frame.to_csv(RESULTS / "temporal_slab_ensemble.csv", index=False)

    means = (
        frame.groupby("lower_generation", as_index=False)
        .agg(
            lower_primary_edges=("lower_primary_edges", "mean"),
            upper_inherited_edges=("upper_inherited_edges", "mean"),
            temporal_plaquettes=("temporal_plaquettes", "mean"),
            vertical_edges=("vertical_edges", "mean"),
            matching_fraction=("matching_fraction", "mean"),
        )
    )
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(
        means["lower_generation"],
        means["matching_fraction"],
        marker="o",
        label="Persistent-core matching coverage",
    )
    plt.ylim(0.95, 1.005)
    plt.xlabel("Lower FORGE generation n")
    plt.ylabel("Matched fraction of lower primary edges")
    plt.title("Injective inherited-edge core across FORGE-YM time slabs")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "persistent_core_matching.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(
        means["lower_generation"],
        means["temporal_plaquettes"],
        marker="o",
        label="Temporal plaquettes",
    )
    plt.plot(
        means["lower_generation"],
        means["lower_primary_edges"],
        marker="s",
        label="Lower primary edges",
    )
    plt.yscale("log")
    plt.xlabel("Lower FORGE generation n")
    plt.ylabel("Count (log scale)")
    plt.title("Euclidean slab growth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "euclidean_slab_scaling.png", dpi=180)
    plt.close()
    return frame


def gauge_stage() -> dict:
    slab = simulate_rewrite_slab(
        lower_generation=2,
        k=4,
        q=2,
        block_size=6,
        closure_mode="face_cycle",
        seed=MASTER_SEED,
    )
    errors = [
        temporal_gauge_invariance_error(slab, MASTER_SEED + index)
        for index in range(8)
    ]
    result = {
        "lower_generation": 2,
        "upper_generation": 3,
        "tests": len(errors),
        "maximum_temporal_plaquette_gauge_error": max(errors),
        "mean_temporal_plaquette_gauge_error": float(np.mean(errors)),
    }
    (RESULTS / "temporal_gauge_invariance.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def transfer_stage() -> tuple[pd.DataFrame, dict]:
    rows, metadata = su2_reflection_transfer_benchmark()
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "su2_reflection_transfer_benchmark.csv", index=False)
    (RESULTS / "su2_reflection_transfer_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    positive = frame[frame["epsilon"] > 0]
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(
        positive["epsilon"],
        positive["operator_norm_error"],
        marker="o",
        label="Observed ||T_h - T_0||",
    )
    plt.plot(
        positive["epsilon"],
        positive["rigorous_norm_bound"],
        marker="s",
        linestyle="--",
        label="Bound 1-exp(-M_+)",
    )
    plt.xlabel("Upper-slice interaction scale epsilon")
    plt.ylabel("Operator norm")
    plt.title("Controlled relation to the heat-kernel transfer operator")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "heat_kernel_relation_bound.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(
        frame["epsilon"],
        frame["energy_gap"],
        marker="o",
        label="Reflected transfer gap",
    )
    plt.axhline(
        metadata["unperturbed_energy_gap"],
        linestyle="--",
        label="Unperturbed heat-kernel gap",
    )
    plt.xlabel("Upper-slice interaction scale epsilon")
    plt.ylabel("Delta = tau^{-1} log(lambda0/lambda1)")
    plt.title("Two-channel SU(2) transfer-gap benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "reflected_transfer_gap.png", dpi=180)
    plt.close()
    return frame, metadata


def write_summary(slab: pd.DataFrame, gauge: dict, transfer: pd.DataFrame, metadata: dict) -> dict:
    ensemble = slab[slab["lower_generation"].isin((1, 2, 3))]
    summary = {
        "version": "0.2",
        "fixed_history_reflection_positivity": True,
        "full_multigeneration_OS_reconstruction": False,
        "temporal_plaquette_count_rule": "P_time = C(lower) * binom(4,2) = 6 C(lower)",
        "slab_ensemble_runs_generations_1_to_3": int(len(ensemble)),
        "all_vertical_graphs_forests": bool(ensemble["vertical_graph_is_forest"].all()),
        "minimum_persistent_matching_fraction": float(ensemble["matching_fraction"].min()),
        "maximum_temporal_gauge_error": gauge["maximum_temporal_plaquette_gauge_error"],
        "su2_basis_dimension": metadata["basis_dimension"],
        "maximum_OS_gram_negative_part": float(
            max(0.0, -transfer["min_OS_gram_eigenvalue"].min())
        ),
        "maximum_ratio_observed_error_to_bound": float(
            transfer[transfer["epsilon"] > 0]["bound_ratio"].max()
        ),
        "exact_heat_kernel_limit_error_epsilon_0": float(
            transfer.loc[transfer["epsilon"] == 0, "operator_norm_error"].iloc[0]
        ),
        "claim_boundary": (
            "The result proves reflection positivity only for the doubled finite slab "
            "associated with a fixed rewrite history (and positive mixtures of such mirrored histories). "
            "It does not prove reflection positivity of arbitrary undoubled or asynchronous multigeneration histories, "
            "nor a four-dimensional continuum limit or a uniform mass gap."
        ),
    }
    (RESULTS / "transfer_program_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def write_manifest() -> None:
    entries = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name != "manifest_sha256.txt":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    (ROOT / "manifest_sha256.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("all", "slab", "gauge", "transfer"), default="all"
    )
    args = parser.parse_args()

    if args.stage == "slab":
        slab_stage()
    elif args.stage == "gauge":
        gauge_stage()
    elif args.stage == "transfer":
        transfer_stage()
    else:
        slab = slab_stage()
        gauge = gauge_stage()
        transfer, metadata = transfer_stage()
        summary = write_summary(slab, gauge, transfer, metadata)
        print(json.dumps(summary, indent=2))
    write_manifest()


if __name__ == "__main__":
    main()
