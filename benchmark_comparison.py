"""Benchmark: SC-only vs. Greedy vs. Lookahead vs. Global ILP Router.

Compares tangible circuit-level metrics across randomly generated circuits:

  * **Total 2-qubit operations** – CX gates + SWAP gates in the output.
    Each SWAP is counted as one 2-qubit operation here; if decomposed into
    native CX gates the overhead is 3× worse for SC-only.
  * **Transduction count** – hybrid-only overhead (1-qubit markers).

The comparison is performed across **multiple cost-parameter regimes**
to show how the router's decision threshold affects physical overhead.

Usage
-----
    python benchmark_comparison.py [--num-trials N] [--seed S]
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass

import numpy as np

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from hybrid_greedy_router import HybridGreedyRouter
from hybrid_lookahead_router import HybridLookaheadRouter
from hybrid_global_ilp import HybridGlobalStaticILP


# ======================================================================
# Random circuit generator
# ======================================================================

def generate_random_cx_circuit(
    num_qubits: int,
    num_gates: int,
    cx_fraction: float = 0.7,
    seed: int | None = None,
) -> QuantumCircuit:
    """Generate a random circuit with single-qubit gates and CX gates.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    num_gates : int
        Total number of gates to insert.
    cx_fraction : float
        Probability that a given gate is a CX (vs. a random 1-qubit gate).
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    QuantumCircuit
        A circuit containing only 1-qubit gates (h, x, z, s, t) and CX.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    single_gates = ["h", "x", "z", "s", "t"]

    for _ in range(num_gates):
        if rng.random() >= cx_fraction:
            gate_name = rng.choice(single_gates)
            qubit = int(rng.integers(0, num_qubits))
            # Equivalent to qc.gate_name(qubit)
            getattr(qc, gate_name)(qubit)
        else:
            q1, q2 = rng.choice(num_qubits, size=2, replace=False)
            qc.cx(int(q1), int(q2))

    return qc


# ======================================================================
# Result container
# ======================================================================

@dataclass
class BenchmarkResult:
    """Stores the comparison results for a single circuit."""

    circuit_id: int
    num_qubits: int
    num_gates: int

    # Original circuit stats.
    original_cx_count: int = 0
    original_1q_count: int = 0

    # SC-only (homogeneous) results.
    sc_cx_count: int = 0
    sc_swap_count: int = 0
    sc_total_2q: int = 0          # cx + swap

    # Hybrid greedy results.
    hy_cx_count: int = 0
    hy_swap_count: int = 0
    hy_transduct_count: int = 0
    hy_total_2q: int = 0          # cx + swap

    # Hybrid lookahead results.
    la_cx_count: int = 0
    la_swap_count: int = 0
    la_transduct_count: int = 0
    la_total_2q: int = 0          # cx + swap

    # Global ILP results.
    ilp_cx_count: int = 0
    ilp_swap_count: int = 0
    ilp_transduct_count: int = 0
    ilp_total_2q: int = 0         # cx + swap

    # Deltas (positive = hybrid is better than SC-only).
    delta_2q_pct: float = 0.0     # greedy: (sc - hy) / sc × 100
    la_delta_2q_pct: float = 0.0  # lookahead: (sc - la) / sc × 100
    ilp_delta_2q_pct: float = 0.0 # ILP: (sc - ilp) / sc × 100


# ======================================================================
# Gate counting helpers
# ======================================================================

def _count_ops(qc: QuantumCircuit) -> dict[str, int]:
    """Return a dict gate_name → count."""
    counts: dict[str, int] = {}
    for instr in qc.data:
        name = instr.operation.name
        counts[name] = counts.get(name, 0) + 1
    return counts


# ======================================================================
# Single-circuit benchmark
# ======================================================================

def benchmark_single_circuit(
    circuit_id: int,
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    c_sc: float,
    c_na: float,
    c_swap: float,
    c_trans: float,
    seed_transpiler: int = 42,
) -> BenchmarkResult:
    """Compile *qc* with both compilers and return the comparison."""
    original_ops = _count_ops(qc)
    original_cx = original_ops.get("cx", 0)
    original_1q = sum(v for k, v in original_ops.items() if k != "cx")

    result = BenchmarkResult(
        circuit_id=circuit_id,
        num_qubits=qc.num_qubits,
        num_gates=sum(original_ops.values()),
        original_cx_count=original_cx,
        original_1q_count=original_1q,
    )

    # ---- SC-only (homogeneous) transpilation ----
    pm = generate_preset_pass_manager(
        optimization_level=1,
        coupling_map=coupling_map,
        seed_transpiler=seed_transpiler,
    )
    sc_circuit = pm.run(qc)
    sc_ops = _count_ops(sc_circuit)

    result.sc_cx_count = sc_ops.get("cx", 0)
    result.sc_swap_count = sc_ops.get("swap", 0)
    result.sc_total_2q = result.sc_cx_count + result.sc_swap_count

    # ---- Hybrid greedy transpilation ----
    layout = Layout.from_intlist(
        list(range(qc.num_qubits)), qc.qregs[0],
    )
    dag = circuit_to_dag(qc)
    router = HybridGreedyRouter(
        coupling_map=coupling_map,
        initial_layout=layout,
        c_sc=c_sc,
        c_na=c_na,
        c_swap=c_swap,
        c_trans=c_trans,
    )
    hybrid_dag = router.run(dag)
    hybrid_circuit = dag_to_circuit(hybrid_dag)
    hybrid_ops = _count_ops(hybrid_circuit)

    result.hy_cx_count = hybrid_ops.get("cx", 0)
    result.hy_swap_count = hybrid_ops.get("swap", 0)
    result.hy_transduct_count = hybrid_ops.get("transduct", 0)
    result.hy_total_2q = result.hy_cx_count + result.hy_swap_count

    # ---- Hybrid lookahead transpilation ----
    dag_la = circuit_to_dag(qc)
    la_router = HybridLookaheadRouter(
        coupling_map=coupling_map,
        initial_layout=layout,
        c_sc=c_sc,
        c_na=c_na,
        c_swap=c_swap,
        c_trans=c_trans,
    )
    la_dag = la_router.run(dag_la)
    la_circuit = dag_to_circuit(la_dag)
    la_ops = _count_ops(la_circuit)

    result.la_cx_count = la_ops.get("cx", 0)
    result.la_swap_count = la_ops.get("swap", 0)
    result.la_transduct_count = la_ops.get("transduct", 0)
    result.la_total_2q = result.la_cx_count + result.la_swap_count

    # ---- Global ILP transpilation ----
    dag_ilp = circuit_to_dag(qc)
    ilp_router = HybridGlobalStaticILP(
        coupling_map=coupling_map,
        initial_layout=layout,
        c_sc=c_sc,
        c_na=c_na,
        c_swap=c_swap,
        c_trans=c_trans,
    )
    ilp_dag = ilp_router.run(dag_ilp)
    ilp_circuit = dag_to_circuit(ilp_dag)
    ilp_ops = _count_ops(ilp_circuit)

    result.ilp_cx_count = ilp_ops.get("cx", 0)
    result.ilp_swap_count = ilp_ops.get("swap", 0)
    result.ilp_transduct_count = ilp_ops.get("transduct", 0)
    result.ilp_total_2q = result.ilp_cx_count + result.ilp_swap_count

    # ---- Deltas (positive = hybrid is better than SC-only) ----
    if result.sc_total_2q > 0:
        result.delta_2q_pct = (
            (result.sc_total_2q - result.hy_total_2q)
            / result.sc_total_2q * 100.0
        )
        result.la_delta_2q_pct = (
            (result.sc_total_2q - result.la_total_2q)
            / result.sc_total_2q * 100.0
        )
        result.ilp_delta_2q_pct = (
            (result.sc_total_2q - result.ilp_total_2q)
            / result.sc_total_2q * 100.0
        )

    return result


# ======================================================================
# Circuit configurations
# ======================================================================

CIRCUIT_CONFIGS = [
    # (num_qubits, num_gates, cx_fraction)
    (5, 20, 1.0),
    (5, 30, 0.7),
    (7, 30, 0.7),
    (7, 50, 0.7),
    (10, 40, 0.7),
    (10, 60, 0.7),
    (10, 80, 0.7),
    (15, 80, 0.7),
    (15, 120, 0.7),
    (20, 100, 0.7),
]

# Cost-parameter regimes that drive the router's decisions.
COST_REGIMES = {
    "Regime A: High trans. cost": {
        "c_sc": 1.0, "c_na": 5.0, "c_swap": 3.0, "c_trans": 10.0,
    },
    "Regime B: Moderate trans.":  {
        "c_sc": 1.0, "c_na": 3.0, "c_swap": 3.0, "c_trans": 5.0,
    },
    "Regime C: Low trans. cost":  {
        "c_sc": 1.0, "c_na": 2.0, "c_swap": 3.0, "c_trans": 2.0,
    },
    "Regime D: Ultra-low trans.": {
        "c_sc": 1.0, "c_na": 1.5, "c_swap": 3.0, "c_trans": 1.0,
    },
}


# ======================================================================
# Batch benchmark
# ======================================================================

def run_batch_benchmark(
    cost_params: dict[str, float],
    num_trials: int = 3,
    base_seed: int = 0,
) -> list[BenchmarkResult]:
    """Run the benchmark across all configurations and trials."""
    results: list[BenchmarkResult] = []
    circuit_id = 0

    for num_qubits, num_gates, cx_frac in CIRCUIT_CONFIGS:
        cm = CouplingMap.from_line(num_qubits)
        for trial in range(num_trials):
            seed = base_seed + circuit_id
            qc = generate_random_cx_circuit(
                num_qubits, num_gates, cx_fraction=cx_frac, seed=seed,
            )
            res = benchmark_single_circuit(
                circuit_id, qc, cm,
                seed_transpiler=seed,
                **cost_params,
            )
            results.append(res)
            circuit_id += 1

    return results


# ======================================================================
# Pretty-print helpers
# ======================================================================

def _header() -> str:
    return (
        f"{'ID':>3}  "
        f"{'Qb':>3}  "
        f"{'Orig':>4}  │  "
        f"{'SC_2q':>6}  │  "
        f"{'GR_2q':>6}  "
        f"{'g_tr':>6}  "
        f"{'Gd2q%':>6}     │  "
        f"{'LA_2q':>6}  "
        f"{'l_tr':>6}  "
        f"{'Ld2q%':>6}     │  "
        f"{'IL_2q':>6}  "
        f"{'i_tr':>6}  "
        f"{'Id2q%':>6}"
    )


def _row(r: BenchmarkResult) -> str:
    return (
        f"{r.circuit_id:>3}  "
        f"{r.num_qubits:>3}  "
        f"{r.original_cx_count:>4}  │  "
        f"{r.sc_total_2q:>6}  │  "
        f"{r.hy_total_2q:>6}  "
        f"{r.hy_transduct_count:>6}  "
        f"{r.delta_2q_pct:>+6.1f}%   │  "
        f"{r.la_total_2q:>6}  "
        f"{r.la_transduct_count:>6}  "
        f"{r.la_delta_2q_pct:>+6.1f}%   │  "
        f"{r.ilp_total_2q:>6}  "
        f"{r.ilp_transduct_count:>6}  "
        f"{r.ilp_delta_2q_pct:>+6.1f}%"
    )


def print_results(
    regime_name: str,
    cost_params: dict[str, float],
    results: list[BenchmarkResult],
) -> None:
    """Print a formatted table and summary statistics."""
    w = 170
    print()
    print("=" * w)
    print(f"  {regime_name}")
    print(
        f"  c_sc={cost_params['c_sc']}  c_na={cost_params['c_na']}  "
        f"c_swap={cost_params['c_swap']}  c_trans={cost_params['c_trans']}"
    )
    print("=" * w)
    print()
    print(_header())
    print("-" * w)

    for r in results:
        print(_row(r))

    print("-" * w)

    # ---- Summary stats ----
    d2q = [r.delta_2q_pct for r in results]
    la_d2q = [r.la_delta_2q_pct for r in results]
    ilp_d2q = [r.ilp_delta_2q_pct for r in results]
    sc_2q = [r.sc_total_2q for r in results]
    hy_2q = [r.hy_total_2q for r in results]
    la_2q = [r.la_total_2q for r in results]
    ilp_2q = [r.ilp_total_2q for r in results]
    hy_tr = [r.hy_transduct_count for r in results]
    la_tr = [r.la_transduct_count for r in results]
    ilp_tr = [r.ilp_transduct_count for r in results]

    print()
    print("  2-QUBIT GATE COMPARISON")
    print(f"    {'Avg SC total 2q ops:':<32} {np.mean(sc_2q):.1f}")
    print(f"    {'Avg Greedy total 2q ops:':<32} {np.mean(hy_2q):.1f}")
    print(f"    {'Avg Lookahead total 2q ops:':<32} {np.mean(la_2q):.1f}")
    print(f"    {'Avg ILP total 2q ops:':<32} {np.mean(ilp_2q):.1f}")
    print(f"    {'Greedy avg d2q%:':<32} {np.mean(d2q):+.1f}%")
    print(f"    {'Lookahead avg d2q%:':<32} {np.mean(la_d2q):+.1f}%")
    print(f"    {'ILP avg d2q%:':<32} {np.mean(ilp_d2q):+.1f}%")
    print(f"    {'Greedy median d2q%:':<32} {np.median(d2q):+.1f}%")
    print(f"    {'Lookahead median d2q%:':<32} {np.median(la_d2q):+.1f}%")
    print(f"    {'ILP median d2q%:':<32} {np.median(ilp_d2q):+.1f}%")
    print()
    print("  TRANSDUCTION OVERHEAD")
    print(f"    {'Greedy avg transductions:':<32} {np.mean(hy_tr):.1f}")
    print(f"    {'Lookahead avg transductions:':<32} {np.mean(la_tr):.1f}")
    print(f"    {'ILP avg transductions:':<32} {np.mean(ilp_tr):.1f}")
    print(f"    {'Greedy total transductions:':<32} {sum(hy_tr)}")
    print(f"    {'Lookahead total transductions:':<32} {sum(la_tr)}")
    print(f"    {'ILP total transductions:':<32} {sum(ilp_tr)}")
    print()

    n = len(results)
    g_wins_2q = sum(1 for v in d2q if v > 0)
    l_wins_2q = sum(1 for v in la_d2q if v > 0)
    i_wins_2q = sum(1 for v in ilp_d2q if v > 0)
    print(f"  Greedy wins on 2q:      {g_wins_2q}/{n}  |  "
          f"Lookahead wins on 2q:   {l_wins_2q}/{n}  |  "
          f"ILP wins on 2q:   {i_wins_2q}/{n}")
    print()


def print_cross_regime_summary(
    all_results: dict[str, list[BenchmarkResult]],
) -> None:
    """Print a compact table comparing all regimes."""
    w = 180
    print()
    print("=" * w)
    print("  CROSS-REGIME SUMMARY  (G = Greedy, L = Lookahead, I = ILP)")
    print("=" * w)
    print()
    print(
        f"  {'Regime':<32} │ "
        f"{'G d2q%':>7}  "
        f"{'L d2q%':>7}  "
        f"{'I d2q%':>7}  "
        f"{'G 2qW':>5}  "
        f"{'L 2qW':>5}  "
        f"{'I 2qW':>5}    │ "
        f"{'G trans':>7}  "
        f"{'L trans':>7}  "
        f"{'I trans':>7}"
    )
    print("-" * w)

    for name, results in all_results.items():
        d2q = [r.delta_2q_pct for r in results]
        la_d2q = [r.la_delta_2q_pct for r in results]
        ilp_d2q = [r.ilp_delta_2q_pct for r in results]
        hy_tr = [r.hy_transduct_count for r in results]
        la_tr = [r.la_transduct_count for r in results]
        ilp_tr = [r.ilp_transduct_count for r in results]
        n = len(results)
        g_w2q = sum(1 for v in d2q if v > 0)
        l_w2q = sum(1 for v in la_d2q if v > 0)
        i_w2q = sum(1 for v in ilp_d2q if v > 0)
        print(
            f"  {name:<32} │ "
            f"{np.mean(d2q):>+6.1f}%  "
            f"{np.mean(la_d2q):>+6.1f}%  "
            f"{np.mean(ilp_d2q):>+6.1f}%  "
            f"{g_w2q:>2}/{n:<2}   "
            f"{l_w2q:>2}/{n:<2}   "
            f"{i_w2q:>2}/{n:<2}  │ "
            f"{np.mean(hy_tr):>6.1f}  "
            f"{np.mean(la_tr):>6.1f}  "
            f"{np.mean(ilp_tr):>6.1f}"
        )

    print("-" * w)
    print()
    print("  Positive Δ% = hybrid is better than SC-only  |  "
          "Negative Δ% = SC-only is better")
    print()
    print("=" * w)


# ======================================================================
# CLI entry-point
# ======================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark SC-only vs. Greedy vs. Lookahead vs. Global ILP Router.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Compares total 2-qubit gate count between
            SC-only transpilation and the Hybrid Greedy Router across
            multiple cost-parameter regimes.
        """),
    )
    parser.add_argument(
        "--num-trials", type=int, default=3,
        help="Number of random trials per configuration (default: 3).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base random seed (default: 0).",
    )
    args = parser.parse_args()

    all_results: dict[str, list[BenchmarkResult]] = {}

    for regime_name, cost_params in COST_REGIMES.items():
        results = run_batch_benchmark(
            cost_params=cost_params,
            num_trials=args.num_trials,
            base_seed=args.seed,
        )
        all_results[regime_name] = results
        print_results(regime_name, cost_params, results)

    print_cross_regime_summary(all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
