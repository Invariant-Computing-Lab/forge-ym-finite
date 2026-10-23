# FORGE-YM v0.3 — Code and Reproducibility Package

This archive accompanies the manuscript **“FORGE-YM: reflection-positive gauge dynamics on rewriting hypergraphs and a falsifiable continuum mass-gap conjecture.”** It is supplied separately from the article and Supplementary Information for peer review and computational audit.

## Scientific status

The package establishes and tests finite-volume constructions: a gauge 2-complex on FORGE-L slices, SU(2) Wilson observables, a consecutive-generation Euclidean slab, reflection positivity of the mirrored fixed-history slab, and a positive round-trip transfer operator. It also states a falsifiable continuum mass-gap conjecture.

It **does not** claim that the four-dimensional Yang–Mills existence and mass-gap Millennium Problem has already been proved. The continuum/projective limit, the full Osterwalder–Schrader reconstruction, nontriviality, universality, and a scale-uniform physical gap remain explicit proof obligations.

## Environment

- Python 3.11 or later
- Dependencies in `requirements.txt`

Create an isolated environment and install dependencies:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Fast audit

```bash
python tests/run_tests.py
```

The test script checks the published generation-2 smoke result, the exact q=2 plaquette count, dual-complex connectivity, the Euclidean-slab forest and matching properties, temporal SU(2) gauge invariance, reflection positivity in the Peter–Weyl benchmark, and the heat-kernel perturbation bound.

## Reproduction commands

```bash
python code/run_forge_ym.py --stage smoke
python code/run_forge_ym.py --stage structure
python code/run_forge_ym.py --stage spectrum
python code/run_forge_ym.py --stage mc
python code/run_transfer_program.py --stage all
```

The Monte Carlo stage is the slowest. Frozen outputs used in the article are included in `results/` and publication figures in `figures/`.

## Directory map

- `code/forge_local.py` — original FORGE-L rewrite and graph diagnostics.
- `code/forge_ym.py` — finite gauge complex, SU(2) spectra, Wilson Monte Carlo, and correlations.
- `code/forge_ym_transfer.py` — consecutive-slice slab, temporal plaquettes, persistent core, gauge audit, and reflected transfer benchmark.
- `code/run_forge_ym.py` — finite gauge experiments.
- `code/run_transfer_program.py` — Euclidean-slab and reflection-positivity experiments.
- `tests/run_tests.py` — deterministic high-value audit.
- `results/` — frozen CSV/JSON outputs.
- `figures/` — frozen figures.
- `FALSIFICATION_PROTOCOL.json` — preregisterable rejection criteria for the continuum conjecture.
- `CLAIM_BOUNDARIES.md` — exact boundary between proved finite results and open continuum claims.
- `manifest_sha256.txt` — SHA-256 digest for every distributed file.

## Seed and reproducibility

The master seed is `20260726`. All deterministic tests use fixed derived seeds. Small Monte Carlo estimates can fluctuate if chains, compiler, or random-number semantics are changed; exact structural identities and gauge-invariance tolerances should not.
