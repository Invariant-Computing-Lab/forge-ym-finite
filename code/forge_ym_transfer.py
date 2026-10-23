from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import numpy as np

from forge_local import Cell, _combinations, _partition_exchangeably, build_spatial_graph
from forge_ym import quaternion_conjugate, quaternion_multiply, random_su2


@dataclass(frozen=True)
class ClosureRecord:
    face: frozenset[int]
    left: int
    right: int
    mediator: int


@dataclass
class SliceSnapshot:
    generation: int
    active: list[Cell]
    closure_edges: list[frozenset[int]]
    closure_records: list[ClosureRecord]
    spatial_graph: nx.Graph


@dataclass(frozen=True)
class TemporalChannel:
    parent_index: int
    support: tuple[int, int]
    lower_edge: tuple[int, int]
    upper_edge: tuple[int, int]
    vertical_left: tuple[int, int]
    vertical_right: tuple[int, int]


@dataclass
class RewriteSlab:
    lower: SliceSnapshot
    upper: SliceSnapshot
    channels: list[TemporalChannel]
    vertical_edges: list[tuple[int, int]]
    successor: dict[tuple[int, int], int]
    seed: int


def _canonical_edge(left: int, right: int) -> tuple[int, int]:
    if left == right:
        raise ValueError("Self-loop is not an admissible gauge edge")
    return (left, right) if left < right else (right, left)


def primary_edges(active: list[Cell]) -> list[tuple[int, int]]:
    values: set[tuple[int, int]] = set()
    for cell in active:
        for edge in itertools.combinations(sorted(cell.vertices), 2):
            values.add(edge)
    return sorted(values)


def inherited_upper_edges(slab: RewriteSlab) -> list[tuple[int, int]]:
    return sorted({channel.upper_edge for channel in slab.channels})


def _make_snapshot(
    generation: int,
    active: list[Cell],
    closure_edges: list[frozenset[int]],
    closure_records: list[ClosureRecord],
) -> SliceSnapshot:
    return SliceSnapshot(
        generation=generation,
        active=list(active),
        closure_edges=list(closure_edges),
        closure_records=list(closure_records),
        spatial_graph=build_spatial_graph(active, closure_edges),
    )


def simulate_rewrite_slab(
    lower_generation: int,
    k: int = 4,
    q: int = 2,
    block_size: int | None = None,
    closure_mode: str = "face_cycle",
    seed: int = 20260726,
    max_active: int = 200_000,
    type_rule: dict[str, tuple[str, ...]] | None = None,
) -> RewriteSlab:
    """Preserve the exact FORGE-L rewrite data for H_n -> H_{n+1}."""
    if lower_generation < 0:
        raise ValueError("lower_generation must be nonnegative")
    if k < 2 or not (1 <= q < k):
        raise ValueError("Require k >= 2 and 1 <= q < k")
    if closure_mode not in {"none", "parent_center", "face_matching", "face_cycle"}:
        raise ValueError(f"Unknown closure mode: {closure_mode}")
    if block_size is None:
        block_size = math.comb(k, q)
    if type_rule is None:
        type_rule = {"L": ("L", "S"), "S": ("L",)}

    rng = random.Random(seed)
    next_vertex = k
    next_event = 1
    active = [Cell(frozenset(range(k)), "L", {0}, 0)]
    closure_edges: list[frozenset[int]] = []
    closure_records: list[ClosureRecord] = []

    lower_snapshot: SliceSnapshot | None = None

    for generation in range(1, lower_generation + 2):
        predicted = sum(
            math.comb(k, q) * len(type_rule[cell.typ]) for cell in active
        )
        if predicted > max_active:
            raise RuntimeError(
                f"Requested slab exceeds max_active: predicted {predicted}"
            )

        if generation == lower_generation + 1:
            lower_snapshot = _make_snapshot(
                lower_generation, active, closure_edges, closure_records
            )

        parents = active
        incident_parents: dict[int, list[int]] = defaultdict(list)
        for parent_index, parent in enumerate(parents):
            for vertex in parent.vertices:
                incident_parents[vertex].append(parent_index)

        successor: dict[tuple[int, int], int] = {}
        vertical_edges: set[tuple[int, int]] = set()
        for vertex in sorted(incident_parents):
            blocks = _partition_exchangeably(
                incident_parents[vertex], block_size, rng
            )
            for block in blocks:
                successor_vertex = next_vertex
                next_vertex += 1
                next_event += 1
                vertical_edges.add((vertex, successor_vertex))
                for parent_index in block:
                    successor[(vertex, parent_index)] = successor_vertex

        children: list[Cell] = []
        by_face: dict[frozenset[int], list[int]] = defaultdict(list)
        by_parent_orbit: dict[tuple[int, int], list[int]] = defaultdict(list)
        channel_map: dict[tuple[int, tuple[int, ...]], TemporalChannel] = {}

        for parent_index, parent in enumerate(parents):
            for orbit_index, output_type in enumerate(type_rule[parent.typ]):
                for support_tuple in _combinations(parent.vertices, q):
                    transported_tuple = tuple(
                        sorted(successor[(vertex, parent_index)] for vertex in support_tuple)
                    )
                    transported_support = frozenset(transported_tuple)
                    fresh = frozenset(range(next_vertex, next_vertex + (k - q)))
                    next_vertex += k - q
                    event_id = next_event
                    next_event += 1
                    child_index = len(children)
                    children.append(
                        Cell(
                            transported_support | fresh,
                            output_type,
                            {event_id},
                            event_id,
                            parent_index,
                            transported_support,
                            fresh,
                            orbit_index,
                        )
                    )
                    by_parent_orbit[(parent_index, orbit_index)].append(child_index)
                    face_size = 1 if q == 1 else q - 1
                    for face in _combinations(transported_support, face_size):
                        by_face[frozenset(face)].append(child_index)

                    if q == 2 and orbit_index == 0:
                        left, right = sorted(support_tuple)
                        upper_left = successor[(left, parent_index)]
                        upper_right = successor[(right, parent_index)]
                        key = (parent_index, tuple(sorted(support_tuple)))
                        channel_map[key] = TemporalChannel(
                            parent_index=parent_index,
                            support=(left, right),
                            lower_edge=_canonical_edge(left, right),
                            upper_edge=_canonical_edge(upper_left, upper_right),
                            vertical_left=(left, upper_left),
                            vertical_right=(right, upper_right),
                        )

        generation_closure_edges: list[frozenset[int]] = []
        generation_records: list[ClosureRecord] = []

        if closure_mode == "parent_center":
            for indices in by_parent_orbit.values():
                mediator = next_vertex
                next_vertex += 1
                next_event += 1
                for index in indices:
                    child = children[index]
                    for vertex in child.new_vertices:
                        generation_closure_edges.append(
                            frozenset((mediator, vertex))
                        )

        elif closure_mode in {"face_matching", "face_cycle"}:
            for face in sorted(by_face, key=lambda value: tuple(sorted(value))):
                indices = sorted(set(by_face[face]))
                rng.shuffle(indices)
                if len(indices) < 2:
                    continue
                if closure_mode == "face_matching":
                    pairs = [
                        (indices[i], indices[i + 1])
                        for i in range(0, len(indices) - 1, 2)
                    ]
                elif len(indices) == 2:
                    pairs = [(indices[0], indices[1])]
                else:
                    pairs = [
                        (indices[i], indices[(i + 1) % len(indices)])
                        for i in range(len(indices))
                    ]

                for left_index, right_index in pairs:
                    left = children[left_index]
                    right = children[right_index]
                    mediator = next_vertex
                    next_vertex += 1
                    next_event += 1
                    for vertex in left.new_vertices | right.new_vertices:
                        generation_closure_edges.append(
                            frozenset((mediator, vertex))
                        )
                    generation_records.append(
                        ClosureRecord(
                            frozenset(face), left_index, right_index, mediator
                        )
                    )

        active = children
        closure_edges = generation_closure_edges
        closure_records = generation_records

        if generation == lower_generation + 1:
            if lower_snapshot is None:
                raise RuntimeError("Internal slab construction error")
            upper_snapshot = _make_snapshot(
                generation, active, closure_edges, closure_records
            )
            channels = sorted(
                channel_map.values(),
                key=lambda item: (item.parent_index, item.support),
            )
            return RewriteSlab(
                lower=lower_snapshot,
                upper=upper_snapshot,
                channels=channels,
                vertical_edges=sorted(vertical_edges),
                successor=successor,
                seed=seed,
            )

    raise RuntimeError("Slab construction did not terminate")


def persistent_edge_matching(
    slab: RewriteSlab,
) -> tuple[dict[tuple[int, int], tuple[int, int]], int]:
    """Maximum injective lower-edge to inherited-upper-edge transport core."""
    lower = primary_edges(slab.lower.active)
    upper = inherited_upper_edges(slab)
    graph = nx.Graph()
    lower_nodes = [("lower", edge) for edge in lower]
    upper_nodes = [("upper", edge) for edge in upper]
    graph.add_nodes_from(lower_nodes, bipartite=0)
    graph.add_nodes_from(upper_nodes, bipartite=1)
    for channel in slab.channels:
        graph.add_edge(("lower", channel.lower_edge), ("upper", channel.upper_edge))
    matching = nx.algorithms.bipartite.maximum_matching(
        graph, top_nodes=lower_nodes
    )
    result: dict[tuple[int, int], tuple[int, int]] = {}
    for node in lower_nodes:
        partner = matching.get(node)
        if partner is not None:
            result[node[1]] = partner[1]
    return result, len(lower)


def slab_metrics(slab: RewriteSlab) -> dict:
    lower_edges = primary_edges(slab.lower.active)
    upper_edges = inherited_upper_edges(slab)
    matching, lower_count = persistent_edge_matching(slab)
    vertical_graph = nx.Graph()
    vertical_graph.add_edges_from(slab.vertical_edges)
    channel_graph = nx.Graph()
    for channel in slab.channels:
        channel_graph.add_edge(
            ("lower", channel.lower_edge), ("upper", channel.upper_edge)
        )
    lower_multiplicity = defaultdict(int)
    upper_multiplicity = defaultdict(int)
    for channel in slab.channels:
        lower_multiplicity[channel.lower_edge] += 1
        upper_multiplicity[channel.upper_edge] += 1
    return {
        "lower_generation": slab.lower.generation,
        "upper_generation": slab.upper.generation,
        "seed": slab.seed,
        "lower_cells": len(slab.lower.active),
        "upper_cells": len(slab.upper.active),
        "lower_primary_edges": len(lower_edges),
        "upper_inherited_edges": len(upper_edges),
        "temporal_channels": len(slab.channels),
        "temporal_plaquettes": len(slab.channels),
        "vertical_edges": len(slab.vertical_edges),
        "vertical_graph_is_forest": nx.is_forest(vertical_graph),
        "vertical_components": nx.number_connected_components(vertical_graph),
        "matching_size": len(matching),
        "matching_fraction": len(matching) / max(1, lower_count),
        "channel_graph_components": nx.number_connected_components(channel_graph),
        "max_lower_channel_multiplicity": max(lower_multiplicity.values(), default=0),
        "max_upper_channel_multiplicity": max(upper_multiplicity.values(), default=0),
    }


def _link_value(
    links: dict[tuple[int, int], np.ndarray], left: int, right: int
) -> np.ndarray:
    edge = _canonical_edge(left, right)
    value = links[edge]
    return value if edge == (left, right) else quaternion_conjugate(value)


def temporal_plaquette_scalars(
    slab: RewriteSlab,
    lower_links: dict[tuple[int, int], np.ndarray],
    upper_links: dict[tuple[int, int], np.ndarray],
    vertical_links: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    values = np.empty(len(slab.channels), dtype=float)
    all_links = {**lower_links, **upper_links, **vertical_links}
    for index, channel in enumerate(slab.channels):
        x, y = channel.support
        xp = slab.successor[(x, channel.parent_index)]
        yp = slab.successor[(y, channel.parent_index)]
        holonomy = np.array((1.0, 0.0, 0.0, 0.0))
        for left, right in ((x, y), (y, yp), (yp, xp), (xp, x)):
            holonomy = quaternion_multiply(
                holonomy, _link_value(all_links, left, right)
            )
        values[index] = holonomy[0]
    return values


def temporal_gauge_invariance_error(slab: RewriteSlab, seed: int = 1) -> float:
    rng = np.random.default_rng(seed)
    lower_edges = primary_edges(slab.lower.active)
    upper_edges = inherited_upper_edges(slab)
    vertical_edges = [_canonical_edge(*edge) for edge in slab.vertical_edges]

    lower_links = {edge: random_su2(rng) for edge in lower_edges}
    upper_links = {edge: random_su2(rng) for edge in upper_edges}
    vertical_links = {edge: random_su2(rng) for edge in vertical_edges}
    original = temporal_plaquette_scalars(
        slab, lower_links, upper_links, vertical_links
    )

    vertices = sorted(
        {
            vertex
            for edge in itertools.chain(lower_edges, upper_edges, vertical_edges)
            for vertex in edge
        }
    )
    gauges = {vertex: random_su2(rng) for vertex in vertices}

    def transform(
        links: dict[tuple[int, int], np.ndarray]
    ) -> dict[tuple[int, int], np.ndarray]:
        transformed: dict[tuple[int, int], np.ndarray] = {}
        for edge, value in links.items():
            source, target = edge
            transformed[edge] = quaternion_multiply(
                quaternion_multiply(gauges[source], value),
                quaternion_conjugate(gauges[target]),
            )
        return transformed

    changed = temporal_plaquette_scalars(
        slab, transform(lower_links), transform(upper_links), transform(vertical_links)
    )
    return float(np.max(np.abs(original - changed)))


def _su2_class_quadrature(points: int) -> tuple[np.ndarray, np.ndarray]:
    if points < 4:
        raise ValueError("At least four quadrature points are required")
    index = np.arange(1, points + 1, dtype=float)
    theta = index * math.pi / (points + 1.0)
    weights = 2.0 / (points + 1.0) * np.sin(theta) ** 2
    return theta, weights


def _su2_characters(theta: np.ndarray, j_max: float) -> tuple[np.ndarray, np.ndarray]:
    count = int(round(2.0 * j_max)) + 1
    js = np.arange(count, dtype=float) / 2.0
    numerator = np.sin(np.outer(theta, 2.0 * js + 1.0))
    denominator = np.sin(theta)[:, None]
    return js, numerator / denominator


def su2_reflection_transfer_benchmark(
    epsilons: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4),
    tau: float = 0.35,
    j_max: float = 4.0,
    quadrature_points: int = 64,
    beta_minus: float = 0.25,
    beta_plus: float = 0.40,
    interaction: float = 0.30,
    random_seed: int = 20260726,
) -> tuple[list[dict], dict]:
    """
    Two-channel SU(2) class-function benchmark of R*R reflection positivity.

    H_{tau/2} is diagonal in the Peter-Weyl character basis. Spatial weights are
    represented by positive multiplication operators compressed to the truncated
    basis. The full theorem is not a truncation theorem; this routine is a finite
    numerical audit of the exact operator identities used in the proof.
    """
    if tau <= 0 or j_max < 0:
        raise ValueError("Invalid benchmark parameters")
    theta, weights = _su2_class_quadrature(quadrature_points)
    js, phi = _su2_characters(theta, j_max)
    gram1 = phi.T @ (weights[:, None] * phi)
    one_basis_error = float(np.linalg.norm(gram1 - np.eye(len(js)), ord=2))

    psi = np.einsum("ia,jb->ijab", phi, phi).reshape(
        quadrature_points * quadrature_points, len(js) * len(js)
    )
    weights2 = np.outer(weights, weights).reshape(-1)
    gram2 = psi.T @ (weights2[:, None] * psi)
    basis_error = float(np.linalg.norm(gram2 - np.eye(gram2.shape[0]), ord=2))

    theta1, theta2 = np.meshgrid(theta, theta, indexing="ij")
    cos1 = np.cos(theta1).reshape(-1)
    cos2 = np.cos(theta2).reshape(-1)
    v_minus = beta_minus * ((1.0 - cos1) + (1.0 - cos2))
    v_plus_template = beta_plus * ((1.0 - cos1) + (1.0 - cos2)) + interaction * (
        1.0 - cos1 * cos2
    )

    def multiplication_matrix(values: np.ndarray) -> np.ndarray:
        matrix = psi.T @ ((weights2 * values)[:, None] * psi)
        return 0.5 * (matrix + matrix.T)

    a_minus = multiplication_matrix(np.exp(-0.5 * v_minus))
    casimir = js * (js + 1.0)
    casimir2 = (casimir[:, None] + casimir[None, :]).reshape(-1)
    h_half = np.diag(np.exp(-0.5 * tau * casimir2))
    h_full = np.diag(np.exp(-tau * casimir2))
    t0 = a_minus @ h_full @ a_minus
    t0 = 0.5 * (t0 + t0.T)
    eig0 = np.linalg.eigvalsh(t0)[::-1]
    gap0 = float(math.log(eig0[0] / eig0[1]) / tau)

    rng = np.random.default_rng(random_seed)
    q, _ = np.linalg.qr(rng.normal(size=(t0.shape[0], min(12, t0.shape[0]))))

    rows: list[dict] = []
    for epsilon in epsilons:
        v_plus = epsilon * v_plus_template
        a_plus = multiplication_matrix(np.exp(-0.5 * v_plus))
        refinement = a_plus @ h_half @ a_minus
        transfer = refinement.T @ refinement
        transfer = 0.5 * (transfer + transfer.T)
        difference = transfer - t0
        observed_error = float(np.linalg.norm(difference, ord=2))
        m_plus = float(np.max(v_plus))
        perturbation_bound = float(1.0 - math.exp(-m_plus))
        eigenvalues = np.linalg.eigvalsh(transfer)[::-1]
        energy_gap = float(math.log(eigenvalues[0] / eigenvalues[1]) / tau)
        os_gram = q.T @ transfer @ q
        os_gram = 0.5 * (os_gram + os_gram.T)
        os_min = float(np.min(np.linalg.eigvalsh(os_gram)))
        min_transfer = float(np.min(np.linalg.eigvalsh(transfer)))
        gap_lower = float("nan")
        if eig0[0] > observed_error and eig0[1] + observed_error > 0:
            ratio = (eig0[0] - observed_error) / (eig0[1] + observed_error)
            if ratio > 1.0:
                gap_lower = float(math.log(ratio) / tau)
        rows.append(
            {
                "epsilon": float(epsilon),
                "M_plus": m_plus,
                "operator_norm_error": observed_error,
                "rigorous_norm_bound": perturbation_bound,
                "bound_ratio": observed_error / perturbation_bound
                if perturbation_bound > 0
                else 0.0,
                "min_transfer_eigenvalue": min_transfer,
                "min_OS_gram_eigenvalue": os_min,
                "lambda0": float(eigenvalues[0]),
                "lambda1": float(eigenvalues[1]),
                "energy_gap": energy_gap,
                "unperturbed_energy_gap": gap0,
                "weyl_gap_lower_bound": gap_lower,
            }
        )

    metadata = {
        "tau": tau,
        "j_max": j_max,
        "basis_dimension": int(t0.shape[0]),
        "quadrature_points_per_link": quadrature_points,
        "one_link_basis_orthogonality_error": one_basis_error,
        "two_link_basis_orthogonality_error": basis_error,
        "beta_minus": beta_minus,
        "beta_plus": beta_plus,
        "interaction": interaction,
        "unperturbed_energy_gap": gap0,
    }
    return rows, metadata
