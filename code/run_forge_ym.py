from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "code"))

from forge_local import graph_summary, simulate
from forge_ym import (
    build_gauge_complex,
    connected_correlations_by_distance,
    finite_volume_g_critical,
    finite_volume_strong_coupling_bound,
    gauge_complex_metrics,
    gauge_invariance_error,
    plaquette_dual_graph,
    run_su2_wilson_monte_carlo,
    sample_pairs_by_dual_distance,
    simulate_extended,
    su2_single_loop_spectrum,
    su2_triangle_strong_coupling_bound,
)

MASTER_SEED = 20260726


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def smoke_test() -> dict:
    simulation = simulate(
        k=4,
        q=2,
        generations=2,
        block_size=6,
        closure_mode="face_cycle",
        seed=MASTER_SEED,
    )
    l_count = sum(cell.typ == "L" for cell in simulation.active)
    s_count = sum(cell.typ == "S" for cell in simulation.active)
    result = {
        "active_cells": len(simulation.active),
        "L": l_count,
        "S": s_count,
        "ratio": l_count / s_count,
        "vertices": simulation.spatial_graph.number_of_nodes(),
        "edges": simulation.spatial_graph.number_of_edges(),
        "giant_fraction": graph_summary(simulation.spatial_graph, seed=1)[
            "giant_fraction"
        ],
    }
    expected = {
        "active_cells": 108,
        "L": 72,
        "S": 36,
        "ratio": 2.0,
        "vertices": 460,
        "edges": 1470,
        "giant_fraction": 1.0,
    }
    result["matches_expected"] = all(
        math.isclose(result[key], value, rel_tol=0, abs_tol=1e-12)
        if isinstance(value, float)
        else result[key] == value
        for key, value in expected.items()
    )
    write_json(RESULTS / "smoke_test.json", result)
    return result


def structure_stage() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for q in (2, 3):
        for generation in (1, 2, 3, 4):
            extended = simulate_extended(
                k=4,
                q=q,
                generations=generation,
                block_size=math.comb(4, q),
                closure_mode="face_cycle",
                seed=MASTER_SEED,
                max_active=100_000,
            )
            complex_ = build_gauge_complex(extended)
            row = gauge_complex_metrics(complex_)
            row["seed"] = MASTER_SEED
            rows.append(row)
    structure = pd.DataFrame(rows)
    structure.to_csv(RESULTS / "gauge_complex_structure.csv", index=False)

    ensemble_rows = []
    for generation, replicates in ((2, 12), (3, 8), (4, 4)):
        for replicate in range(replicates):
            seed = MASTER_SEED + 1000 * generation + replicate
            extended = simulate_extended(
                q=2,
                generations=generation,
                block_size=6,
                closure_mode="face_cycle",
                seed=seed,
                max_active=100_000,
            )
            complex_ = build_gauge_complex(extended)
            row = gauge_complex_metrics(complex_)
            row.update(seed=seed, replicate=replicate)
            ensemble_rows.append(row)
    ensemble = pd.DataFrame(ensemble_rows)
    ensemble.to_csv(RESULTS / "gauge_complex_ensemble.csv", index=False)

    q2 = structure[structure["q"] == 2].sort_values("generation")
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(q2["cells"], q2["plaquettes"], marker="o")
    plt.xlabel("Active FORGE-L cells")
    plt.ylabel("FORGE-YM plaquettes")
    plt.title("Linear plaquette growth in the q=2 closure completion")
    plt.tight_layout()
    plt.savefig(FIGURES / "plaquette_count_scaling.png", dpi=180)
    plt.close()

    return structure, ensemble


def finite_gap_bounds_stage(structure: pd.DataFrame) -> pd.DataFrame:
    rows = []
    q2 = structure[structure["q"] == 2].sort_values("generation")
    for _, row in q2.iterrows():
        item = {
            "generation": int(row["generation"]),
            "cells": int(row["cells"]),
            "plaquettes": int(row["plaquettes"]),
            "g_critical_crude_bound": finite_volume_g_critical(int(row["plaquettes"])),
        }
        for g in (4, 6, 8, 10, 20):
            item[f"gap_lower_bound_g{g}"] = finite_volume_strong_coupling_bound(
                g, int(row["plaquettes"])
            )
        rows.append(item)
    bounds = pd.DataFrame(rows)
    bounds.to_csv(RESULTS / "finite_volume_gap_bounds.csv", index=False)
    return bounds


def screening_mass_stage(correlations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for beta, values in correlations.groupby("beta"):
        by_distance = values.set_index("distance")
        def get_value(distance: int, field: str) -> float:
            if distance not in by_distance.index:
                return float("nan")
            return float(by_distance.loc[distance, field])
        c1, c2, c3 = (get_value(d, "correlation") for d in (1, 2, 3))
        e1, e2, e3 = (get_value(d, "standard_error") for d in (1, 2, 3))
        rows.append({
            "beta": float(beta),
            "C1": c1,
            "C2": c2,
            "C1_snr": c1 / e1 if e1 > 0 else float("nan"),
            "C2_snr": c2 / e2 if e2 > 0 else float("nan"),
            "m_eff_12": math.log(c1 / c2) if c1 > 0 and c2 > 0 else float("nan"),
            "C3": c3,
            "C3_snr": c3 / e3 if e3 > 0 else float("nan"),
            "m_eff_23": math.log(c2 / c3) if c2 > 0 and c3 > 0 else float("nan"),
        })
    masses = pd.DataFrame(rows).sort_values("beta")
    masses.to_csv(RESULTS / "su2_screening_mass_estimates.csv", index=False)
    return masses

def spectrum_stage() -> tuple[pd.DataFrame, pd.DataFrame]:
    spectrum_rows = []
    for g in np.linspace(0.30, 4.0, 100):
        values = su2_single_loop_spectrum(g, loop_length=3, j_max=60, eigenvalues=4)
        spectrum_rows.append(
            {
                "g": g,
                "E0": values[0],
                "E1": values[1],
                "E2": values[2],
                "gap": values[1] - values[0],
                "strong_coupling_bound": su2_triangle_strong_coupling_bound(g),
            }
        )
    spectrum = pd.DataFrame(spectrum_rows)
    spectrum.to_csv(RESULTS / "su2_single_triangle_spectrum.csv", index=False)

    convergence_rows = []
    for g in (0.4, 0.6, 1.0, 2.0):
        reference = su2_single_loop_spectrum(g, loop_length=3, j_max=80, eigenvalues=3)
        reference_gap = reference[1] - reference[0]
        for j_max in (5, 10, 20, 40, 60):
            values = su2_single_loop_spectrum(
                g, loop_length=3, j_max=j_max, eigenvalues=3
            )
            gap = values[1] - values[0]
            convergence_rows.append(
                {
                    "g": g,
                    "j_max": j_max,
                    "gap": gap,
                    "reference_gap_jmax80": reference_gap,
                    "absolute_error": abs(gap - reference_gap),
                }
            )
    convergence = pd.DataFrame(convergence_rows)
    convergence.to_csv(RESULTS / "su2_truncation_convergence.csv", index=False)

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(spectrum["g"], spectrum["gap"], label="Exact truncated gap")
    plt.plot(
        spectrum["g"],
        spectrum["strong_coupling_bound"],
        linestyle="--",
        label="Rigorous lower bound",
    )
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("Gauge coupling g")
    plt.ylabel("Energy gap (a=1)")
    plt.title("SU(2) one-triangle Hamiltonian benchmark")
    plt.ylim(bottom=min(-2.0, spectrum["strong_coupling_bound"].min()))
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "su2_triangle_gap.png", dpi=180)
    plt.close()

    return spectrum, convergence


def monte_carlo_stage() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    complexes = {}
    for generation in (1, 2):
        extended = simulate_extended(
            q=2,
            generations=generation,
            block_size=6,
            closure_mode="face_cycle",
            seed=MASTER_SEED,
        )
        complexes[generation] = build_gauge_complex(extended)

    raw_rows = []
    invariance_errors = []
    betas = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    for generation, complex_ in complexes.items():
        for beta in betas:
            for replicate in range(3):
                seed = MASTER_SEED + generation * 100_000 + int(beta * 1000) + replicate
                result = run_su2_wilson_monte_carlo(
                    complex_,
                    beta=beta,
                    seed=seed,
                    burn_sweeps=1_000,
                    measurement_sweeps=3_000,
                    thin=10,
                    store_configurations=False,
                )
                error = gauge_invariance_error(result, seed + 7)
                invariance_errors.append(error)
                raw_rows.append(
                    {
                        "generation": generation,
                        "beta": beta,
                        "replicate": replicate,
                        "seed": seed,
                        "plaquette_mean": result["plaquette_mean_series"].mean(),
                        "triangle_mean": result["triangle_mean_series"].mean(),
                        "quadrilateral_mean": result[
                            "quadrilateral_mean_series"
                        ].mean(),
                        "action_density": result["action_density_series"].mean(),
                        "acceptance": result["acceptance"],
                        "epsilon": result["epsilon"],
                        "gauge_invariance_max_error": error,
                    }
                )
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(RESULTS / "su2_monte_carlo_raw.csv", index=False)

    summary = (
        raw.groupby(["generation", "beta"], as_index=False)
        .agg(
            plaquette_mean=("plaquette_mean", "mean"),
            plaquette_sd=("plaquette_mean", "std"),
            triangle_mean=("triangle_mean", "mean"),
            quadrilateral_mean=("quadrilateral_mean", "mean"),
            action_density=("action_density", "mean"),
            acceptance=("acceptance", "mean"),
            gauge_error_max=("gauge_invariance_max_error", "max"),
        )
        .sort_values(["generation", "beta"])
    )
    summary["plaquette_se"] = summary["plaquette_sd"] / math.sqrt(3.0)
    summary.to_csv(RESULTS / "su2_monte_carlo_summary.csv", index=False)

    correlation_rows = []
    complex_ = complexes[2]
    dual = plaquette_dual_graph(complex_)
    pairs = sample_pairs_by_dual_distance(
        dual,
        max_distance=6,
        sources=150,
        pairs_per_distance=5_000,
        seed=MASTER_SEED,
    )
    for beta in (1.0, 2.0, 4.0, 8.0):
        seed = MASTER_SEED + 900_000 + int(beta * 1000)
        result = run_su2_wilson_monte_carlo(
            complex_,
            beta=beta,
            seed=seed,
            burn_sweeps=1_500,
            measurement_sweeps=5_000,
            thin=10,
            store_configurations=True,
        )
        correlations = connected_correlations_by_distance(
            result["configurations"], pairs, batch_size=50
        )
        for row in correlations:
            row.update(beta=beta, generation=2, seed=seed)
            correlation_rows.append(row)
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(RESULTS / "su2_screening_correlations.csv", index=False)
    screening_mass_stage(correlations)

    plt.figure(figsize=(7.2, 4.8))
    for generation, values in summary.groupby("generation"):
        plt.errorbar(
            values["beta"],
            values["plaquette_mean"],
            yerr=values["plaquette_se"],
            marker="o",
            capsize=3,
            label=f"Generation {generation}",
        )
    plt.xscale("log")
    plt.xlabel("Wilson beta")
    plt.ylabel("Mean normalized SU(2) plaquette trace")
    plt.title("Finite FORGE-YM Wilson-measure simulations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "su2_plaquette_vs_beta.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    for beta, values in correlations.groupby("beta"):
        plt.errorbar(
            values["distance"],
            values["correlation"],
            yerr=values["standard_error"],
            marker="o",
            capsize=3,
            label=f"beta={beta:g}",
        )
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("Plaquette-dual graph distance")
    plt.ylabel("Connected plaquette correlation")
    plt.title("Short-range screening on the generation-2 gauge complex")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "su2_screening_correlations.png", dpi=180)
    plt.close()

    invariance = {
        "maximum_gauge_invariance_error": max(invariance_errors),
        "median_gauge_invariance_error": float(np.median(invariance_errors)),
        "number_of_checks": len(invariance_errors),
    }
    write_json(RESULTS / "gauge_invariance_checks.json", invariance)
    return raw, summary, correlations, invariance


def complex_incidence_hash(complex_) -> str:
    graph = nx.Graph()
    for vertex in complex_.simulation.spatial_graph.nodes():
        graph.add_node(("v", vertex), kind="vertex")
    for p_index, plaquette in enumerate(complex_.plaquettes):
        p_node = ("p", p_index)
        graph.add_node(p_node, kind=complex_.plaquette_kind[p_index])
        for vertex in plaquette:
            graph.add_edge(("v", vertex), p_node)
    return nx.weisfeiler_lehman_graph_hash(graph, node_attr="kind", iterations=5)


def permutation_check() -> dict:
    extended = simulate_extended(q=2, generations=2, block_size=6, seed=MASTER_SEED)
    complex_ = build_gauge_complex(extended)
    rng = np.random.default_rng(MASTER_SEED + 77)
    vertices = list(complex_.simulation.spatial_graph.nodes())
    shuffled = list(vertices)
    rng.shuffle(shuffled)
    mapping = dict(zip(vertices, shuffled))

    relabeled_graph = nx.relabel_nodes(complex_.simulation.spatial_graph, mapping, copy=True)
    relabeled_plaquettes = [tuple(mapping[v] for v in p) for p in complex_.plaquettes]
    relabeled = type(complex_)(
        simulation=type(complex_.simulation)(
            complex_.simulation.k,
            complex_.simulation.q,
            complex_.simulation.block_size,
            complex_.simulation.closure_mode,
            complex_.simulation.generations,
            complex_.simulation.active,
            complex_.simulation.closure_edges,
            relabeled_graph,
            complex_.simulation.incidence_graph,
            complex_.simulation.causal_graph,
            complex_.simulation.history,
            complex_.simulation.seed,
        ),
        plaquettes=relabeled_plaquettes,
        plaquette_kind=list(complex_.plaquette_kind),
        oriented_edges=sorted(tuple(sorted(e)) for e in relabeled_graph.edges()),
    )

    base_hash = complex_incidence_hash(complex_)
    relabeled_hash = complex_incidence_hash(relabeled)
    result = {
        "base_hash": base_hash,
        "relabeled_hash": relabeled_hash,
        "hashes_equal": base_hash == relabeled_hash,
        "base_vertices": complex_.simulation.spatial_graph.number_of_nodes(),
        "relabeled_vertices": relabeled_graph.number_of_nodes(),
        "base_edges": complex_.simulation.spatial_graph.number_of_edges(),
        "relabeled_edges": relabeled_graph.number_of_edges(),
        "base_plaquettes": len(complex_.plaquettes),
        "relabeled_plaquettes": len(relabeled_plaquettes),
    }
    write_json(RESULTS / "permutation_check.json", result)
    return result


def build_summary(
    smoke: dict,
    structure: pd.DataFrame,
    ensemble: pd.DataFrame,
    spectrum: pd.DataFrame,
    mc_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    invariance: dict,
    permutation: dict,
) -> dict:
    q2 = structure[structure["q"] == 2].sort_values("generation")
    min_gap_row = spectrum.loc[spectrum["gap"].idxmin()]
    generation2_beta2 = mc_summary[
        (mc_summary["generation"] == 2) & (mc_summary["beta"] == 2.0)
    ].iloc[0]
    masses = screening_mass_stage(correlations)
    beta2_mass = masses[masses["beta"] == 2.0].iloc[0]

    summary = {
        "smoke_test_passed": smoke["matches_expected"],
        "q2_generation4": q2.iloc[-1].to_dict(),
        "q2_ensemble_all_plaquette_duals_connected": bool(
            np.allclose(ensemble["plaquette_dual_giant_fraction"], 1.0)
        ),
        "q2_ensemble_max_edge_plaquette_incidence": int(
            ensemble["max_edge_plaquette_incidence"].max()
        ),
        "su2_single_triangle_minimum_gap_on_grid": float(min_gap_row["gap"]),
        "su2_single_triangle_minimum_gap_g": float(min_gap_row["g"]),
        "generation2_beta2_plaquette_mean": float(
            generation2_beta2["plaquette_mean"]
        ),
        "generation2_beta2_effective_screening_mass_r1_r2": float(
            beta2_mass["m_eff_12"]
        ),
        "generation2_beta2_C2_signal_to_noise": float(beta2_mass["C2_snr"]),
        "maximum_gauge_invariance_error": invariance[
            "maximum_gauge_invariance_error"
        ],
        "permutation_hashes_equal": permutation["hashes_equal"],
        "millennium_problem_solved": False,
        "proven_finite_result": (
            "For every finite FORGE-YM 2-complex and compact connected gauge group, "
            "the heat-kernel transfer operator defined in the report is gauge-equivariant, "
            "self-adjoint, compact, positivity improving, and has a strictly positive finite-volume gap."
        ),
    }
    write_json(RESULTS / "summary.json", summary)
    return summary


def make_manifest() -> None:
    files = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if (
            path.is_file()
            and path.name != "manifest_sha256.txt"
            and "__pycache__" not in relative.parts
            and path.suffix != ".pyc"
            and "docx_render" not in relative.parts
        ):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append(f"{digest}  {relative.as_posix()}")
    (ROOT / "manifest_sha256.txt").write_text("\n".join(files) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("all", "smoke", "structure", "spectrum", "mc"),
        default="all",
    )
    args = parser.parse_args()

    if args.stage == "smoke":
        print(json.dumps(smoke_test(), indent=2))
        return
    if args.stage == "structure":
        structure, ensemble = structure_stage()
        finite_gap_bounds_stage(structure)
        print(structure.to_string(index=False))
        print(ensemble.groupby("generation")["plaquette_dual_giant_fraction"].min())
        return
    if args.stage == "spectrum":
        spectrum, convergence = spectrum_stage()
        print(spectrum.head())
        print(convergence.to_string(index=False))
        return
    if args.stage == "mc":
        _, summary, correlations, invariance = monte_carlo_stage()
        print(summary.to_string(index=False))
        print(correlations.to_string(index=False))
        print(invariance)
        return

    smoke = smoke_test()
    structure, ensemble = structure_stage()
    finite_gap_bounds_stage(structure)
    spectrum, _ = spectrum_stage()
    _, mc_summary, correlations, invariance = monte_carlo_stage()
    permutation = permutation_check()
    summary = build_summary(
        smoke,
        structure,
        ensemble,
        spectrum,
        mc_summary,
        correlations,
        invariance,
        permutation,
    )
    make_manifest()
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
