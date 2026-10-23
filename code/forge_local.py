from __future__ import annotations
 
import hashlib
import itertools
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable
 
import networkx as nx
import numpy as np
from scipy import sparse
 
PHI = (1.0 + math.sqrt(5.0)) / 2.0
 
 
@dataclass
class Cell:
    """An unordered typed k-hyperedge in one spatial slice."""
    vertices: frozenset[int]
    typ: str
    deps: set[int]
    event_id: int
    parent_id: int = -1
    support: frozenset[int] = field(default_factory=frozenset)
    new_vertices: frozenset[int] = field(default_factory=frozenset)
    orbit_id: int = 0
 
 
@dataclass
class Simulation:
    k: int
    q: int
    block_size: int
    closure_mode: str
    generations: int
    active: list[Cell]
    closure_edges: list[frozenset[int]]
    spatial_graph: nx.Graph
    incidence_graph: nx.Graph
    causal_graph: nx.DiGraph
    history: list[dict]
    seed: int
 
 
def _combinations(values: Iterable[int], r: int):
    return itertools.combinations(sorted(values), r)
 
 
def build_spatial_graph(active: list[Cell], closure_edges: list[frozenset[int]]) -> nx.Graph:
    """Clique projection for primary hyperedges; closure relations are barycentric stars."""
    graph = nx.Graph()
    for cell in active:
        vertices = sorted(cell.vertices)
        graph.add_nodes_from(vertices)
        graph.add_edges_from(itertools.combinations(vertices, 2))
    for edge in closure_edges:
        vertices = sorted(edge)
        graph.add_nodes_from(vertices)
        graph.add_edges_from(itertools.combinations(vertices, 2))
    return graph
 
 
def build_incidence_graph(active: list[Cell], closure_edges: list[frozenset[int]]) -> nx.Graph:
    graph = nx.Graph()
    for index, cell in enumerate(active):
        hyperedge = ("h", index)
        graph.add_node(hyperedge, kind="hyperedge", typ=cell.typ)
        for vertex in cell.vertices:
            node = ("v", vertex)
            graph.add_node(node, kind="vertex", typ="V")
            graph.add_edge(node, hyperedge)
    offset = len(active)
    for index, edge in enumerate(closure_edges):
        hyperedge = ("c", offset + index)
        graph.add_node(hyperedge, kind="closure", typ="C")
        for vertex in edge:
            node = ("v", vertex)
            graph.add_node(node, kind="vertex", typ="V")
            graph.add_edge(node, hyperedge)
    return graph
 
 
def _partition_exchangeably(items: list[int], block_size: int, rng: random.Random) -> list[list[int]]:
    """Uniform shuffle followed by fixed-size blocks; its law is invariant under item permutations."""
    shuffled = list(items)
    rng.shuffle(shuffled)
    return [shuffled[i:i + block_size] for i in range(0, len(shuffled), block_size)]
 
 
def simulate(
    k: int = 4,
    q: int = 2,
    generations: int = 4,
    block_size: int | None = None,
    closure_mode: str = "face_cycle",
    seed: int = 20260726,
    max_active: int = 50_000,
    seed_labels: list[int] | None = None,
    type_rule: dict[str, tuple[str, ...]] | None = None,
) -> Simulation:
    """
    FORGE-L local stochastic hypergraph rewrite.
 
    The rewrite law is vertex-permutation invariant. Individual histories break the
    symmetry through exchangeable random partitions/cycles, avoiding deterministic
    high-degree hubs.
    """
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
 
    for generation in range(1, generations + 1):
        orbit_size = math.comb(k, q)
        predicted = sum(orbit_size * len(type_rule[cell.typ]) for cell in active)
        if predicted > max_active:
            break
 
        # Incidence transport. Old slice vertices are retired. Each old vertex's
        # incident parent cells are exchangeably partitioned into bounded blocks;
        # one successor is shared only inside a block. This is the spontaneous
        # symmetry-breaking step that prevents immortal high-valence hubs.
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
 
        # Complete q-subset orbit makes the local rule equivariant under every
        # permutation of a parent's k vertices. Fibonacci typing is inseparable
        # from topology: L emits an L-orbit and an S-orbit; S emits one L-orbit.
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
                    dependencies.update(successor[(vertex, parent_index)][1] for vertex in old_support_tuple)
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
            # Every intrinsic (q-1)-face receives a random matching or cycle.
            # The random law is exchangeable, so it is permutation invariant,
            # while each realization spontaneously chooses a local neighborhood.
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
 
        active = children
        closure_edges = generation_closure_edges
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
    return Simulation(
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
 
 
def largest_connected(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        return graph.copy()
    if nx.is_connected(graph):
        return graph.copy()
    component = max(nx.connected_components(graph), key=len)
    return graph.subgraph(component).copy()
 
 
def degree_gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.allclose(values.sum(), 0):
        return 0.0
    values = np.sort(values)
    n = values.size
    return float((2 * np.dot(np.arange(1, n + 1), values) / (n * values.sum())) - (n + 1) / n)
 
 
def graph_summary(graph: nx.Graph, distance_sources: int = 24, seed: int = 0) -> dict:
    if graph.number_of_nodes() == 0:
        return {}
    components = list(nx.connected_components(graph))
    giant_fraction = max(map(len, components)) / graph.number_of_nodes()
    giant = largest_connected(graph)
    degrees = np.array([degree for _, degree in giant.degree()], dtype=float)
    local_rng = np.random.default_rng(seed)
    nodes = list(giant.nodes())
    chosen = nodes
    if len(nodes) > distance_sources:
        chosen = [nodes[int(i)] for i in local_rng.choice(len(nodes), distance_sources, replace=False)]
    mean_distances = []
    eccentricities = []
    for source in chosen:
        distances = nx.single_source_shortest_path_length(giant, source)
        values = np.fromiter(distances.values(), dtype=float)
        if values.size > 1:
            mean_distances.append(float(values.mean()))
            eccentricities.append(float(values.max()))
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "giant_fraction": giant_fraction,
        "avg_degree": float(degrees.mean()),
        "max_degree": int(degrees.max()),
        "max_degree_fraction": float(degrees.max() / max(1, giant.number_of_nodes())),
        "degree_gini": degree_gini(degrees),
        "sample_mean_distance": float(np.mean(mean_distances)) if mean_distances else math.nan,
        "sample_eccentricity": float(np.mean(eccentricities)) if eccentricities else math.nan,
    }
 
 
def _lazy_symmetric_matrix(graph: nx.Graph):
    graph = largest_connected(graph)
    adjacency = nx.to_scipy_sparse_array(graph, dtype=float, format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_sqrt = np.zeros_like(degree)
    mask = degree > 0
    inverse_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
    symmetric = sparse.diags(inverse_sqrt) @ adjacency @ sparse.diags(inverse_sqrt)
    return (0.5 * sparse.identity(adjacency.shape[0], format="csr") + 0.5 * symmetric).tocsr()
 
 
def spectral_dimension_curve(
    graph: nx.Graph,
    max_time: int = 48,
    probes: int = 12,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    graph = largest_connected(graph)
    if graph.number_of_nodes() < 30:
        return np.array([]), np.array([]), np.array([])
    transition = _lazy_symmetric_matrix(graph)
    n = transition.shape[0]
    local_rng = np.random.default_rng(seed)
    return_probability = np.zeros(max_time, dtype=float)
    for _ in range(probes):
        probe = local_rng.choice(np.array([-1.0, 1.0]), size=n)
        state = probe.copy()
        for time_index in range(max_time):
            state = transition @ state
            return_probability[time_index] += np.dot(probe, state) / n
    return_probability = np.maximum(return_probability / probes, 1e-15)
    times = np.arange(1, max_time + 1, dtype=float)
    dimension = -2.0 * np.gradient(np.log(return_probability), np.log(times))
    return times, return_probability, dimension
 
 
def plateau_dimension(
    graph: nx.Graph,
    max_time: int = 48,
    probes: int = 12,
    seed: int = 0,
) -> tuple[float, float, int, np.ndarray, np.ndarray, np.ndarray]:
    times, probability, dimension = spectral_dimension_curve(graph, max_time, probes, seed)
    if times.size == 0:
        return math.nan, math.nan, 0, times, probability, dimension
    n = largest_connected(graph).number_of_nodes()
    valid = (
        (times >= 5)
        & (times <= max_time - 3)
        & (probability > 4.0 / n)
        & np.isfinite(dimension)
        & (dimension > 0.0)
        & (dimension < 12.0)
    )
    indices = np.where(valid)[0]
    if indices.size < 5:
        return math.nan, math.nan, 0, times, probability, dimension
 
    # Select the longest locally stable window of length 5-12, minimizing robust spread.
    best = None
    for window in range(5, min(13, indices.size + 1)):
        for start in range(0, indices.size - window + 1):
            selected = indices[start:start + window]
            if not np.all(np.diff(selected) == 1):
                continue
            values = dimension[selected]
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            score = mad + 0.015 / window
            candidate = (score, median, mad, window, selected)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        values = dimension[indices]
        return float(np.median(values)), float(np.median(np.abs(values - np.median(values)))), len(values), times, probability, dimension
    _, median, mad, window, _ = best
    return median, mad, window, times, probability, dimension
 
 
def wl_hash(graph: nx.Graph, node_attr: str | None = None) -> str:
    return nx.weisfeiler_lehman_graph_hash(graph, node_attr=node_attr, iterations=5)
 
 
def metric_fingerprint(graph: nx.Graph) -> str:
    summary = graph_summary(graph, seed=17)
    degree_sequence = sorted(degree for _, degree in graph.degree())
    payload = repr((summary, degree_sequence)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
