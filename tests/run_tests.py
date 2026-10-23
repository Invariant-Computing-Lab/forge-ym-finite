from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from forge_local import simulate
from forge_ym import simulate_extended, build_gauge_complex, gauge_complex_metrics
from forge_ym_transfer import (
    simulate_rewrite_slab,
    slab_metrics,
    temporal_gauge_invariance_error,
    su2_reflection_transfer_benchmark,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    # Published FORGE-L generation-2 smoke test.
    sim = simulate(k=4, q=2, generations=2, block_size=6,
                   closure_mode="face_cycle", seed=20260726)
    require(len(sim.active) == 108, f"active cells: {len(sim.active)}")
    require(sim.spatial_graph.number_of_nodes() == 460,
            f"vertices: {sim.spatial_graph.number_of_nodes()}")
    require(sim.spatial_graph.number_of_edges() == 1470,
            f"edges: {sim.spatial_graph.number_of_edges()}")

    # Exact finite gauge-complex audit at generation 2.
    extended = simulate_extended(k=4, q=2, generations=2, block_size=6,
                                 closure_mode="face_cycle", seed=20260726)
    complex_ = build_gauge_complex(extended)
    metrics = gauge_complex_metrics(complex_)
    require(metrics["cells"] == 108, str(metrics))
    require(metrics["plaquettes"] == 1296, str(metrics))
    require(math.isclose(metrics["plaquettes_per_cell"], 12.0,
                         rel_tol=0.0, abs_tol=1e-12), str(metrics))
    require(metrics["plaquette_dual_components"] == 1, str(metrics))

    # Consecutive-generation slab audit.
    slab = simulate_rewrite_slab(lower_generation=2, seed=20260726)
    slab_info = slab_metrics(slab)
    require(slab_info["vertical_graph_is_forest"], str(slab_info))
    require(math.isclose(slab_info["matching_fraction"], 1.0,
                         rel_tol=0.0, abs_tol=1e-12), str(slab_info))
    require(slab_info["temporal_plaquettes"] == 648, str(slab_info))
    gauge_error = temporal_gauge_invariance_error(slab, seed=20260727)
    require(gauge_error < 1e-12, f"temporal gauge error: {gauge_error}")

    # Reflection-positive finite Peter–Weyl benchmark and heat-kernel bound.
    rows, metadata = su2_reflection_transfer_benchmark()
    require(metadata["two_link_basis_orthogonality_error"] < 1e-11,
            str(metadata))
    require(abs(rows[0]["operator_norm_error"]) < 1e-12, str(rows[0]))
    for row in rows:
        require(row["min_OS_gram_eigenvalue"] >= -1e-12, str(row))
        require(row["min_transfer_eigenvalue"] >= -1e-12, str(row))
        require(row["energy_gap"] > 0.0, str(row))
        require(row["operator_norm_error"] <= row["rigorous_norm_bound"] + 1e-12,
                str(row))

    print("FORGE-YM v0.3 deterministic audit: PASS")
    print({
        "smoke": {"cells": 108, "vertices": 460, "edges": 1470},
        "gauge_complex": metrics,
        "slab": slab_info,
        "temporal_gauge_error": gauge_error,
        "transfer_basis_dimension": metadata["basis_dimension"],
        "max_operator_bound_ratio": max(row["bound_ratio"] for row in rows),
    })


if __name__ == "__main__":
    main()
