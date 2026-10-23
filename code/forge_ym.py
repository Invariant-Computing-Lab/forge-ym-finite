from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np
from scipy.linalg import eigh_tridiagonal

try:
    import numba as nb
except ImportError:  # pragma: no cover
    nb = None

from forge_local import (
    Cell,
    Simulation,
    _combinations,
    _partition_exchangeably,
    build_incidence_graph,
    build_spatial_graph,
)


@dataclass
class ClosureRecord:
    face: frozenset[int]
    left: int
    right: int
    mediator: int


@dataclass
class ExtendedSimulation:
    simulation: Simulation
    closure_records: list[ClosureRecord]


@dataclass
class GaugeComplex:
    simulation: Simulation
    plaquettes: list[tuple[int, ...]]
    plaquette_kind: list[str]
    oriented_edges: list[tuple[int, int]]


def simulate_extended(
    k: int = 4,
    q: int = 2,
    generations: int = 4,
    block_size: int | None = None,
    closure_mode: str = "face_cycle",
    seed: int = 20260726,
    max_active: int = 50_000,
    seed_labels: list[int] | None = None,
    type_rule: dict[str, tuple[str, ...]] | None = None,
) -> ExtendedSimulation:
    """Run FORGE-L while preserving the final-generation closure-pair data."""
    if k < 2 or not (1 <= q < k):
        raise ValueError("Require k >= 2 and 1 <= q < k")
    if closure_mode not in {"none", "parent_center", "face_matching", "face_cycle"}:
        raise ValueError(f"Unknown closure mode: {closure_mode}")
    if block_size is None:
        block_size = math.comb(k, q)
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if type_rule is None:
        type_rule = {"L": ("L", "S"), "S": ("L",)}

    labels = list(range(k)) if seed_labels is None else list(seed_labels)
    if len(labels) != k or len(set(labels)) != k:
        raise ValueError("seed_labels must contain k distinct labels")

    rng = random.Random(seed)
    next_vertex = max(labels) + 1
    next_event = 1
    active = [Cell(frozenset(labels), "L", {0}, 0)]
    causal = nx.DiGraph()
    causal.add_node(0, generation=0, typ="seed")
    closure_edges: list[frozenset[int]] = []
    history: list[dict] = []
    final_records: list[ClosureRecord] = []

    for generation in range(1, generations + 1):
        orbit_size = math.comb(k, q)
        predicted = sum(orbit_size * len(type_rule[cell.typ]) for cell in active)
        if predicted > max_active:
            break

        incident_parents: dict[int, list[int]] = defaultdict(list)
        for parent_index, parent in enumerate(active):
            for vertex in parent.vertices:
                incident_parents[vertex].append(parent_index)

        successor: dict[tuple[int, int], tuple[int, int]] = {}
        for vertex in sorted(incident_parents):
            blocks = _partition_exchangeably(incident_parents[vertex], block_size, rng)
            for block in blocks:
                successor_vertex = next_vertex
                next_vertex += 1
                event_id = next_event
                next_event += 1
                causal.add_node(event_id, generation=generation, typ="transport")
                dependencies: set[int] = set()
                for parent_index in block:
                    dependencies.update(active[parent_index].deps)
                    successor[(vertex, parent_index)] = (successor_vertex, event_id)
                for dependency in sorted(dependencies):
                    causal.add_edge(dependency, event_id)

        children: list[Cell] = []
        by_face: dict[frozenset[int], list[int]] = defaultdict(list)
        by_parent_orbit: dict[tuple[int, int], list[int]] = defaultdict(list)

        for parent_index, parent in enumerate(active):
            for orbit_index, output_type in enumerate(type_rule[parent.typ]):
                for old_support_tuple in _combinations(parent.vertices, q):
                    transported_support = frozenset(
                        successor[(vertex, parent_index)][0] for vertex in old_support_tuple
                    )
                    fresh = frozenset(range(next_vertex, next_vertex + (k - q)))
                    next_vertex += k - q
                    event_id = next_event
                    next_event += 1
                    causal.add_node(event_id, generation=generation, typ=output_type)
                    dependencies = set(parent.deps)
                    dependencies.update(
                        successor[(vertex, parent_index)][1] for vertex in old_support_tuple
                    )
                    for dependency in sorted(dependencies):
                        causal.add_edge(dependency, event_id)
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

        generation_closure_edges: list[frozenset[int]] = []
        generation_records: list[ClosureRecord] = []

        if closure_mode == "parent_center":
            for indices in by_parent_orbit.values():
                mediator = next_vertex
                next_vertex += 1
                event_id = next_event
                next_event += 1
                causal.add_node(event_id, generation=generation, typ="closure")
                for index in indices:
                    child = children[index]
                    causal.add_edge(child.event_id, event_id)
                    child.deps.add(event_id)
                    for vertex in child.new_vertices:
                        generation_closure_edges.append(frozenset((mediator, vertex)))

        elif closure_mode in {"face_matching", "face_cycle"}:
            for face in sorted(by_face, key=lambda value: tuple(sorted(value))):
                indices = sorted(set(by_face[face]))
                rng.shuffle(indices)
                if len(indices) < 2:
                    continue
                if closure_mode == "face_matching":
                    pairs = [(indices[i], indices[i + 1]) for i in range(0, len(indices) - 1, 2)]
                elif len(indices) == 2:
                    pairs = [(indices[0], indices[1])]
                else:
                    pairs = [(indices[i], indices[(i + 1) % len(indices)]) for i in range(len(indices))]

                for left_index, right_index in pairs:
                    left = children[left_index]
                    right = children[right_index]
                    mediator = next_vertex
                    next_vertex += 1
                    event_id = next_event
                    next_event += 1
                    causal.add_node(event_id, generation=generation, typ="closure")
                    causal.add_edge(left.event_id, event_id)
                    causal.add_edge(right.event_id, event_id)
                    left.deps.add(event_id)
                    right.deps.add(event_id)
                    for vertex in left.new_vertices | right.new_vertices:
                        generation_closure_edges.append(frozenset((mediator, vertex)))
                    generation_records.append(
                        ClosureRecord(frozenset(face), left_index, right_index, mediator)
                    )

        active = children
        closure_edges = generation_closure_edges
        final_records = generation_records
        spatial = build_spatial_graph(active, closure_edges)
        type_counts = {typ: sum(cell.typ == typ for cell in active) for typ in type_rule}
        history.append(
            {
                "generation": generation,
                "active_cells": len(active),
                "L_cells": type_counts.get("L", 0),
                "S_cells": type_counts.get("S", 0),
                "L_S_ratio": type_counts.get("L", 0) / max(1, type_counts.get("S", 0)),
                "vertices": spatial.number_of_nodes(),
                "edges": spatial.number_of_edges(),
                "closure_edges": len(closure_edges),
                "causal_events": causal.number_of_nodes(),
                "causal_edges": causal.number_of_edges(),
            }
        )

    spatial = build_spatial_graph(active, closure_edges)
    incidence = build_incidence_graph(active, closure_edges)
    simulation = Simulation(
        k,
        q,
        block_size,
        closure_mode,
        len(history),
        active,
        closure_edges,
        spatial,
        incidence,
        causal,
        history,
        seed,
    )
    return ExtendedSimulation(simulation, final_records)


def canonical_cycle(cycle: Iterable[int]) -> tuple[int, ...]:
    values = list(cycle)
    if len(values) < 3 or len(set(values)) != len(values):
        raise ValueError("A plaquette must be a simple cycle with at least three vertices")
    rotations: list[tuple[int, ...]] = []
    for orientation in (values, list(reversed(values))):
        rotations.extend(
            tuple(orientation[i:] + orientation[:i]) for i in range(len(orientation))
        )
    return min(rotations)


def build_gauge_complex(
    extended: ExtendedSimulation,
    include_primary_triangles: bool = True,
    include_closure_quads: bool = True,
) -> GaugeComplex:
    """
    Promote a FORGE-L slice to a finite oriented 2-complex.

    Primary 2-cells are triangular faces of each k=4 cell. For each closure event
    joining children c,c' across intrinsic face f with mediator m, add every
    quadrilateral (m,x,a,y), where a in f, x is fresh in c, and y is fresh in c'.
    Every side of this quadrilateral is already an edge of the FORGE-L spatial graph.
    """
    simulation = extended.simulation
    plaquette_map: dict[tuple[int, ...], str] = {}

    if include_primary_triangles:
        for cell in simulation.active:
            if len(cell.vertices) < 3:
                continue
            for triangle in itertools.combinations(sorted(cell.vertices), 3):
                plaquette_map[canonical_cycle(triangle)] = "primary_triangle"

    if include_closure_quads:
        for record in extended.closure_records:
            left = simulation.active[record.left]
            right = simulation.active[record.right]
            for shared_vertex in sorted(record.face):
                for left_fresh in sorted(left.new_vertices):
                    for right_fresh in sorted(right.new_vertices):
                        cycle = canonical_cycle(
                            (record.mediator, left_fresh, shared_vertex, right_fresh)
                        )
                        plaquette_map[cycle] = "closure_quad"

    plaquettes = list(plaquette_map)
    kinds = [plaquette_map[p] for p in plaquettes]
    oriented_edges = sorted(tuple(sorted(edge)) for edge in simulation.spatial_graph.edges())

    for plaquette in plaquettes:
        for left, right in zip(plaquette, plaquette[1:] + plaquette[:1]):
            if not simulation.spatial_graph.has_edge(left, right):
                raise RuntimeError(f"Invalid plaquette edge {(left, right)}")

    return GaugeComplex(simulation, plaquettes, kinds, oriented_edges)


def plaquette_dual_graph(complex_: GaugeComplex) -> nx.Graph:
    edge_to_plaquettes: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, plaquette in enumerate(complex_.plaquettes):
        for left, right in zip(plaquette, plaquette[1:] + plaquette[:1]):
            edge_to_plaquettes[tuple(sorted((left, right)))].append(index)
    dual = nx.Graph()
    dual.add_nodes_from(range(len(complex_.plaquettes)))
    for indices in edge_to_plaquettes.values():
        dual.add_edges_from(itertools.combinations(indices, 2))
    return dual


def gauge_complex_metrics(complex_: GaugeComplex) -> dict:
    simulation = complex_.simulation
    dual = plaquette_dual_graph(complex_)
    edge_incidence: dict[tuple[int, int], int] = defaultdict(int)
    for plaquette in complex_.plaquettes:
        for left, right in zip(plaquette, plaquette[1:] + plaquette[:1]):
            edge_incidence[tuple(sorted((left, right)))] += 1
    if dual.number_of_nodes():
        giant = max(map(len, nx.connected_components(dual))) / dual.number_of_nodes()
        components = nx.number_connected_components(dual)
    else:
        giant = 0.0
        components = 0
    primary = sum(kind == "primary_triangle" for kind in complex_.plaquette_kind)
    closure = sum(kind == "closure_quad" for kind in complex_.plaquette_kind)
    graph = simulation.spatial_graph
    return {
        "generation": simulation.generations,
        "q": simulation.q,
        "cells": len(simulation.active),
        "vertices": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "plaquettes": len(complex_.plaquettes),
        "primary_triangles": primary,
        "closure_quads": closure,
        "plaquettes_per_cell": len(complex_.plaquettes) / max(1, len(simulation.active)),
        "plaquette_dual_giant_fraction": giant,
        "plaquette_dual_components": components,
        "max_edge_plaquette_incidence": max(edge_incidence.values(), default=0),
        "graph_cycle_rank": graph.number_of_edges()
        - graph.number_of_nodes()
        + nx.number_connected_components(graph),
    }


def su2_single_loop_spectrum(
    g: float,
    loop_length: int = 3,
    j_max: float = 30.0,
    lattice_spacing: float = 1.0,
    eigenvalues: int = 6,
) -> np.ndarray:
    """
    Exact character-basis truncation for one SU(2) Wilson loop.

    Basis states are normalized characters chi_j, j=0,1/2,...,j_max. The electric
    term is (g^2/(2a))*L*j(j+1). The magnetic term is
    (1/(2g^2 a))*(2-chi_{1/2}), and chi_{1/2}chi_j=chi_{j+1/2}+chi_{j-1/2}.
    """
    if g <= 0 or lattice_spacing <= 0 or loop_length < 3 or j_max < 0:
        raise ValueError("Invalid spectrum parameters")
    count = int(round(2 * j_max)) + 1
    js = np.arange(count, dtype=float) / 2.0
    diagonal = (
        (g * g / (2.0 * lattice_spacing)) * loop_length * js * (js + 1.0)
        + 1.0 / (g * g * lattice_spacing)
    )
    off_diagonal = np.full(count - 1, -1.0 / (2.0 * g * g * lattice_spacing))
    upper = min(eigenvalues - 1, count - 1)
    return eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select="i",
        select_range=(0, upper),
        check_finite=True,
    )[0]


def su2_triangle_strong_coupling_bound(g: float, lattice_spacing: float = 1.0) -> float:
    """One-triangle lower bound: Delta >= 9 g^2/(8a) - 1/(g^2 a)."""
    return 9.0 * g * g / (8.0 * lattice_spacing) - 1.0 / (
        g * g * lattice_spacing
    )


def finite_volume_strong_coupling_bound(
    g: float,
    plaquettes: int,
    lattice_spacing: float = 1.0,
) -> float:
    """
    Crude global bound for the Kogut-Susskind Hamiltonian on a graph containing
    a triangle: Delta >= 9g^2/(8a) - P/(g^2 a).
    """
    return 9.0 * g * g / (8.0 * lattice_spacing) - plaquettes / (
        g * g * lattice_spacing
    )


def finite_volume_g_critical(plaquettes: int) -> float:
    """Coupling above which the crude finite-volume bound is positive."""
    return (8.0 * plaquettes / 9.0) ** 0.25


def _prepare_lattice(complex_: GaugeComplex):
    edges = complex_.oriented_edges
    edge_index = {edge: index for index, edge in enumerate(edges)}
    max_length = max(map(len, complex_.plaquettes), default=0)
    plaquette_edges = np.full((len(complex_.plaquettes), max_length), -1, dtype=np.int64)
    plaquette_signs = np.zeros((len(complex_.plaquettes), max_length), dtype=np.int8)
    plaquette_lengths = np.zeros(len(complex_.plaquettes), dtype=np.int64)
    incidence: list[list[int]] = [[] for _ in edges]

    for p_index, plaquette in enumerate(complex_.plaquettes):
        plaquette_lengths[p_index] = len(plaquette)
        for position, (left, right) in enumerate(
            zip(plaquette, plaquette[1:] + plaquette[:1])
        ):
            edge = tuple(sorted((left, right)))
            e_index = edge_index[edge]
            plaquette_edges[p_index, position] = e_index
            plaquette_signs[p_index, position] = 1 if left < right else -1
            incidence[e_index].append(p_index)

    pointers = np.zeros(len(edges) + 1, dtype=np.int64)
    for index, values in enumerate(incidence):
        pointers[index + 1] = pointers[index] + len(values)
    flattened = np.empty(pointers[-1], dtype=np.int64)
    for index, values in enumerate(incidence):
        flattened[pointers[index] : pointers[index + 1]] = values

    return edges, plaquette_edges, plaquette_signs, plaquette_lengths, pointers, flattened


def random_su2_array(edge_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(edge_count, 4))
    values /= np.linalg.norm(values, axis=1)[:, None]
    return values


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )


def quaternion_conjugate(value: np.ndarray) -> np.ndarray:
    return np.array((value[0], -value[1], -value[2], -value[3]), dtype=float)


def random_su2(rng: np.random.Generator) -> np.ndarray:
    value = rng.normal(size=4)
    return value / np.linalg.norm(value)


def gauge_transform_links(
    links: np.ndarray,
    edges: list[tuple[int, int]],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vertices = sorted({vertex for edge in edges for vertex in edge})
    gauges = {vertex: random_su2(rng) for vertex in vertices}
    transformed = np.empty_like(links)
    for index, (source, target) in enumerate(edges):
        transformed[index] = quaternion_multiply(
            quaternion_multiply(gauges[source], links[index]),
            quaternion_conjugate(gauges[target]),
        )
    return transformed


if nb is not None:

    @nb.njit(cache=True)
    def _seed_numba(seed: int) -> None:
        np.random.seed(seed)


    @nb.njit(cache=True)
    def _plaquette_scalar_numba(
        links: np.ndarray,
        edge_row: np.ndarray,
        sign_row: np.ndarray,
        length: int,
    ) -> float:
        hw, hx, hy, hz = 1.0, 0.0, 0.0, 0.0
        for position in range(length):
            link = links[edge_row[position]]
            bw, bx, by, bz = link[0], link[1], link[2], link[3]
            if sign_row[position] < 0:
                bx, by, bz = -bx, -by, -bz
            nw = hw * bw - hx * bx - hy * by - hz * bz
            nx = hw * bx + hx * bw + hy * bz - hz * by
            ny = hw * by - hx * bz + hy * bw + hz * bx
            nz = hw * bz + hx * by - hy * bx + hz * bw
            hw, hx, hy, hz = nw, nx, ny, nz
        return hw


    @nb.njit(cache=True)
    def _all_plaquette_scalars_numba(
        links: np.ndarray,
        plaquette_edges: np.ndarray,
        plaquette_signs: np.ndarray,
        plaquette_lengths: np.ndarray,
    ) -> np.ndarray:
        result = np.empty(plaquette_edges.shape[0])
        for index in range(plaquette_edges.shape[0]):
            result[index] = _plaquette_scalar_numba(
                links,
                plaquette_edges[index],
                plaquette_signs[index],
                plaquette_lengths[index],
            )
        return result


    @nb.njit(cache=True)
    def _metropolis_sweep_numba(
        links: np.ndarray,
        plaquette_edges: np.ndarray,
        plaquette_signs: np.ndarray,
        plaquette_lengths: np.ndarray,
        incidence_pointers: np.ndarray,
        incidence_values: np.ndarray,
        beta: float,
        epsilon: float,
        order: np.ndarray,
    ) -> float:
        accepted = 0
        for order_index in range(order.shape[0]):
            edge_index = order[order_index]
            old_sum = 0.0
            for pointer in range(
                incidence_pointers[edge_index], incidence_pointers[edge_index + 1]
            ):
                plaquette_index = incidence_values[pointer]
                old_sum += _plaquette_scalar_numba(
                    links,
                    plaquette_edges[plaquette_index],
                    plaquette_signs[plaquette_index],
                    plaquette_lengths[plaquette_index],
                )

            old0, old1, old2, old3 = (
                links[edge_index, 0],
                links[edge_index, 1],
                links[edge_index, 2],
                links[edge_index, 3],
            )
            x, y, z = np.random.normal(), np.random.normal(), np.random.normal()
            norm = math.sqrt(x * x + y * y + z * z)
            x, y, z = x / norm, y / norm, z / norm
            angle = (2.0 * np.random.random() - 1.0) * epsilon
            rw, sine = math.cos(angle), math.sin(angle)
            rx, ry, rz = sine * x, sine * y, sine * z

            links[edge_index, 0] = rw * old0 - rx * old1 - ry * old2 - rz * old3
            links[edge_index, 1] = rw * old1 + rx * old0 + ry * old3 - rz * old2
            links[edge_index, 2] = rw * old2 - rx * old3 + ry * old0 + rz * old1
            links[edge_index, 3] = rw * old3 + rx * old2 - ry * old1 + rz * old0

            new_sum = 0.0
            for pointer in range(
                incidence_pointers[edge_index], incidence_pointers[edge_index + 1]
            ):
                plaquette_index = incidence_values[pointer]
                new_sum += _plaquette_scalar_numba(
                    links,
                    plaquette_edges[plaquette_index],
                    plaquette_signs[plaquette_index],
                    plaquette_lengths[plaquette_index],
                )

            delta_action = -beta * (new_sum - old_sum)
            if delta_action <= 0.0 or np.random.random() < math.exp(-delta_action):
                accepted += 1
            else:
                links[edge_index, 0] = old0
                links[edge_index, 1] = old1
                links[edge_index, 2] = old2
                links[edge_index, 3] = old3

        return accepted / order.shape[0]


def all_plaquette_scalars(
    links: np.ndarray,
    plaquette_edges: np.ndarray,
    plaquette_signs: np.ndarray,
    plaquette_lengths: np.ndarray,
) -> np.ndarray:
    if nb is None:
        raise RuntimeError("Numba is required for the Monte Carlo stage")
    return _all_plaquette_scalars_numba(
        links, plaquette_edges, plaquette_signs, plaquette_lengths
    )


def run_su2_wilson_monte_carlo(
    complex_: GaugeComplex,
    beta: float,
    seed: int,
    burn_sweeps: int = 2_000,
    measurement_sweeps: int = 6_000,
    thin: int = 10,
    epsilon: float = 0.5,
    store_configurations: bool = False,
) -> dict:
    if nb is None:
        raise RuntimeError("Numba is required for the Monte Carlo stage")
    if beta < 0 or burn_sweeps < 0 or measurement_sweeps <= 0 or thin <= 0:
        raise ValueError("Invalid Monte Carlo parameters")

    (
        edges,
        plaquette_edges,
        plaquette_signs,
        plaquette_lengths,
        incidence_pointers,
        incidence_values,
    ) = _prepare_lattice(complex_)
    links = random_su2_array(len(edges), seed)
    order_rng = np.random.default_rng(seed + 100_000)
    _seed_numba(seed + 200_000)

    epsilon = float(epsilon)
    recent_acceptance: list[float] = []
    for sweep in range(burn_sweeps):
        acceptance = _metropolis_sweep_numba(
            links,
            plaquette_edges,
            plaquette_signs,
            plaquette_lengths,
            incidence_pointers,
            incidence_values,
            float(beta),
            epsilon,
            order_rng.permutation(len(edges)),
        )
        recent_acceptance.append(acceptance)
        if (sweep + 1) % 50 == 0:
            mean_acceptance = float(np.mean(recent_acceptance[-50:]))
            if mean_acceptance < 0.45:
                epsilon *= 0.88
            elif mean_acceptance > 0.65:
                epsilon *= 1.12
            epsilon = min(max(epsilon, 0.02), 2.8)

    samples = measurement_sweeps // thin
    plaquette_mean = np.empty(samples)
    triangle_mean = np.empty(samples)
    quadrilateral_mean = np.empty(samples)
    action_density = np.empty(samples)
    configurations = (
        np.empty((samples, len(complex_.plaquettes)), dtype=np.float32)
        if store_configurations
        else None
    )
    triangle_mask = plaquette_lengths == 3
    quadrilateral_mask = plaquette_lengths == 4
    acceptances: list[float] = []
    sample_index = 0

    for sweep in range(measurement_sweeps):
        acceptance = _metropolis_sweep_numba(
            links,
            plaquette_edges,
            plaquette_signs,
            plaquette_lengths,
            incidence_pointers,
            incidence_values,
            float(beta),
            epsilon,
            order_rng.permutation(len(edges)),
        )
        acceptances.append(acceptance)
        if (sweep + 1) % thin == 0:
            values = _all_plaquette_scalars_numba(
                links, plaquette_edges, plaquette_signs, plaquette_lengths
            )
            plaquette_mean[sample_index] = values.mean()
            triangle_mean[sample_index] = values[triangle_mask].mean()
            quadrilateral_mean[sample_index] = values[quadrilateral_mask].mean()
            action_density[sample_index] = (1.0 - values).mean()
            if configurations is not None:
                configurations[sample_index] = values
            sample_index += 1

    return {
        "beta": float(beta),
        "seed": int(seed),
        "links": links,
        "edges": edges,
        "plaquette_edges": plaquette_edges,
        "plaquette_signs": plaquette_signs,
        "plaquette_lengths": plaquette_lengths,
        "incidence_pointers": incidence_pointers,
        "incidence_values": incidence_values,
        "epsilon": epsilon,
        "acceptance": float(np.mean(acceptances)),
        "plaquette_mean_series": plaquette_mean,
        "triangle_mean_series": triangle_mean,
        "quadrilateral_mean_series": quadrilateral_mean,
        "action_density_series": action_density,
        "configurations": configurations,
    }


def gauge_invariance_error(result: dict, seed: int = 1) -> float:
    original = all_plaquette_scalars(
        result["links"],
        result["plaquette_edges"],
        result["plaquette_signs"],
        result["plaquette_lengths"],
    )
    transformed_links = gauge_transform_links(result["links"], result["edges"], seed)
    transformed = all_plaquette_scalars(
        transformed_links,
        result["plaquette_edges"],
        result["plaquette_signs"],
        result["plaquette_lengths"],
    )
    return float(np.max(np.abs(original - transformed)))


def sample_pairs_by_dual_distance(
    dual: nx.Graph,
    max_distance: int = 6,
    sources: int = 150,
    pairs_per_distance: int = 5_000,
    seed: int = 0,
) -> dict[int, list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    nodes = np.array(list(dual.nodes()), dtype=int)
    source_nodes = rng.choice(nodes, size=min(sources, len(nodes)), replace=False)
    buckets: dict[int, list[tuple[int, int]]] = {
        distance: [] for distance in range(1, max_distance + 1)
    }
    for source in source_nodes:
        distances = nx.single_source_shortest_path_length(
            dual, int(source), cutoff=max_distance
        )
        for target, distance in distances.items():
            if 1 <= distance <= max_distance:
                buckets[distance].append(tuple(sorted((int(source), int(target)))))

    result: dict[int, list[tuple[int, int]]] = {}
    for distance, pairs in buckets.items():
        unique_pairs = list(set(pairs))
        if len(unique_pairs) > pairs_per_distance:
            selected = rng.choice(
                len(unique_pairs), size=pairs_per_distance, replace=False
            )
            unique_pairs = [unique_pairs[index] for index in selected]
        result[distance] = unique_pairs
    return result


def connected_correlations_by_distance(
    configurations: np.ndarray,
    pairs: dict[int, list[tuple[int, int]]],
    batch_size: int = 50,
) -> list[dict]:
    if configurations is None:
        raise ValueError("Monte Carlo configurations were not stored")
    values = configurations.astype(float)
    means = values.mean(axis=0)
    centered = values - means
    output: list[dict] = []

    for distance, distance_pairs in sorted(pairs.items()):
        if not distance_pairs:
            continue
        pair_left = np.array([pair[0] for pair in distance_pairs], dtype=int)
        pair_right = np.array([pair[1] for pair in distance_pairs], dtype=int)
        per_sample = np.mean(
            centered[:, pair_left] * centered[:, pair_right], axis=1
        )
        batches = len(per_sample) // batch_size
        if batches >= 2:
            batch_values = per_sample[: batches * batch_size].reshape(
                batches, batch_size
            ).mean(axis=1)
            standard_error = float(
                batch_values.std(ddof=1) / math.sqrt(len(batch_values))
            )
        else:
            standard_error = float("nan")
        output.append(
            {
                "distance": distance,
                "correlation": float(per_sample.mean()),
                "standard_error": standard_error,
                "pairs": len(distance_pairs),
            }
        )
    return output
