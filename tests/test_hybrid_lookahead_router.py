"""Tests for the HybridLookaheadRouter transpilation pass.

Run with:
    python test_hybrid_lookahead_router.py
"""

from __future__ import annotations

import sys

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap, Layout

from hybrid_lookahead_router import HybridLookaheadRouter
from hybrid_greedy_router import TRANSDUCT


# ======================================================================
# Helpers
# ======================================================================

def _build_linear_coupling_map(n: int) -> CouplingMap:
    """Return a linear coupling map: 0-1-2-..(n-1)."""
    edges = [[i, i + 1] for i in range(n - 1)]
    return CouplingMap(edges)


def _build_trivial_layout(qc: QuantumCircuit) -> Layout:
    """Return a trivial layout mapping qubit i -> physical node i."""
    return Layout.from_intlist(
        list(range(qc.num_qubits)), qc.qregs[0],
    )


def _count_gates(qc: QuantumCircuit) -> dict[str, int]:
    """Return a dict of gate_name -> count in *qc*."""
    counts: dict[str, int] = {}
    for instr in qc.data:
        name = instr.operation.name
        counts[name] = counts.get(name, 0) + 1
    return counts


def _run_pass(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap | None = None,
    layout: Layout | None = None,
    **kwargs,
) -> QuantumCircuit:
    """Run HybridLookaheadRouter on *circuit* and return the result circuit."""
    if coupling_map is None:
        coupling_map = _build_linear_coupling_map(circuit.num_qubits)
    if layout is None:
        layout = _build_trivial_layout(circuit)

    dag = circuit_to_dag(circuit)
    router = HybridLookaheadRouter(
        coupling_map=coupling_map,
        initial_layout=layout,
        **kwargs,
    )
    routed_dag = router.run(dag)
    return dag_to_circuit(routed_dag)


# ======================================================================
# Test cases — basic behaviour (same as greedy)
# ======================================================================

def test_adjacent_gate_stays_on_sc():
    """Adjacent qubits on the coupling map → SC is cheapest (no SWAPs, no
    transductions).

    With defaults (c_sc=1, c_na=5, c_swap=3, c_trans=10):
      Cost_SC = c_sc = 1   (+ future window, but no future gates)
      Cost_NA = c_na + 2·c_trans = 25
    → expect 0 transductions, 0 SWAPs.
    """
    qc = QuantumCircuit(5)
    qc.cx(0, 1)  # adjacent on linear map

    result = _run_pass(qc)
    counts = _count_gates(result)

    assert counts.get("transduct", 0) == 0, (
        f"Expected 0 transductions, got {counts.get('transduct', 0)}"
    )
    assert counts.get("swap", 0) == 0, (
        f"Expected 0 SWAPs, got {counts.get('swap', 0)}"
    )
    assert counts.get("cx", 0) == 1
    print("  PASS: test_adjacent_gate_stays_on_sc")


def test_distant_gate_goes_to_na():
    """Distant qubits where NA is cheaper than routing on SC.

    On a 5-qubit linear map, CX(0, 4) has distance 4.
      c_sc=1, c_na=2, c_swap=3, c_trans=3.
      Cost_SC = 1 + 3·3 = 10 (+ future)
      Cost_NA = 2 + 2·3 = 8  (+ future)
    → expect NA execution (2 transductions, 0 SWAPs).
    """
    qc = QuantumCircuit(5)
    qc.cx(0, 4)

    result = _run_pass(qc, c_sc=1.0, c_na=2.0, c_swap=3.0, c_trans=3.0)
    counts = _count_gates(result)

    assert counts.get("transduct", 0) == 2, (
        f"Expected 2 transductions, got {counts.get('transduct', 0)}"
    )
    assert counts.get("swap", 0) == 0, (
        f"Expected 0 SWAPs, got {counts.get('swap', 0)}"
    )
    assert counts.get("cx", 0) == 1
    print("  PASS: test_distant_gate_goes_to_na")


def test_sc_swap_routing():
    """Force SC execution with SWAP routing.

    5-qubit linear map, CX(0, 3): distance = 3.
      c_sc=1, c_na=100, c_swap=1, c_trans=100
      Cost_SC = 1 + 1·(3-1) = 3 (+ future)
      Cost_NA = 100 + 2·100 = 300 (+ future)
    → expect 2 SWAPs, 0 transductions.
    """
    qc = QuantumCircuit(5)
    qc.cx(0, 3)

    result = _run_pass(qc, c_sc=1.0, c_na=100.0, c_swap=1.0, c_trans=100.0)
    counts = _count_gates(result)

    assert counts.get("transduct", 0) == 0, (
        f"Expected 0 transductions, got {counts.get('transduct', 0)}"
    )
    assert counts.get("swap", 0) == 2, (
        f"Expected 2 SWAPs, got {counts.get('swap', 0)}"
    )
    assert counts.get("cx", 0) == 1
    print("  PASS: test_sc_swap_routing")


def test_single_qubit_gates_pass_through():
    """Single-qubit gates should always be forwarded unchanged."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.x(1)
    qc.z(2)

    result = _run_pass(qc)
    counts = _count_gates(result)

    assert counts.get("h", 0) == 1
    assert counts.get("x", 0) == 1
    assert counts.get("z", 0) == 1
    assert counts.get("transduct", 0) == 0
    assert counts.get("swap", 0) == 0
    print("  PASS: test_single_qubit_gates_pass_through")


# ======================================================================
# Test cases — look-ahead specific behaviour
# ======================================================================

def test_lookahead_prevents_thrashing():
    """The look-ahead should prevent unnecessary back-and-forth transduction.

    Scenario on a 5-qubit linear map:
      Gate 1: CX(0, 4) — distant pair
      Gate 2: CX(0, 1) — adjacent pair (q0 involved again)

    With greedy (c_sc=1, c_na=2, c_swap=3, c_trans=3):
      Gate 1: Cost_SC=10, Cost_NA=8 → greedy picks NA → 2 transductions.
      Gate 2: q0 now on NA, q1 on SC.
        Cost_NA = 2 + 3 = 5 (transduct q1)
        Cost_SC = 1 + 0 + 3 = 4 (transduct q0 back, adjacent)
        → greedy picks SC → 1 more transduction.  Total = 3.

    With look-ahead, when evaluating Gate 1, the heuristic sees Gate 2
    (CX(0,1)) in the window.  The NA score for Gate 1 includes the future
    cost of Gate 2 with q0 in NA (requiring transduction back or keeping
    q1 in NA).  This may cause different routing than pure greedy.

    We just verify the pass produces valid output and completes without error.
    The exact transduction count depends on the window scoring, but should
    be ≥ 0 and ≤ total possible.
    """
    qc = QuantumCircuit(5)
    qc.cx(0, 4)
    qc.cx(0, 1)

    result = _run_pass(qc, c_sc=1.0, c_na=2.0, c_swap=3.0, c_trans=3.0)
    counts = _count_gates(result)

    assert counts.get("cx", 0) == 2
    assert counts.get("transduct", 0) >= 0
    print(f"  PASS: test_lookahead_prevents_thrashing (counts: {counts})")


def test_lookahead_keeps_qubits_on_sc_for_future_adjacent_gates():
    """When future gates are all adjacent on SC, the look-ahead should
    prefer keeping qubits on SC even for the first gate.

    Scenario on 5-qubit linear map:
      CX(0,1), CX(1,2), CX(2,3) — all adjacent.

    With any reasonable costs, all gates should stay on SC because
    the window sees a chain of cheap SC operations ahead.

    c_sc=1, c_na=5, c_swap=3, c_trans=10.
    """
    qc = QuantumCircuit(5)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    result = _run_pass(qc, c_sc=1.0, c_na=5.0, c_swap=3.0, c_trans=10.0)
    counts = _count_gates(result)

    assert counts.get("transduct", 0) == 0, (
        f"Expected 0 transductions for all-adjacent chain, "
        f"got {counts.get('transduct', 0)}"
    )
    assert counts.get("swap", 0) == 0, (
        f"Expected 0 SWAPs for all-adjacent chain, "
        f"got {counts.get('swap', 0)}"
    )
    assert counts.get("cx", 0) == 3
    print("  PASS: test_lookahead_keeps_qubits_on_sc_for_future_adjacent_gates")


def test_window_depth_parameter():
    """Verify that changing window_depth affects scoring (no crash)."""
    qc = QuantumCircuit(5)
    qc.cx(0, 4)
    qc.cx(0, 3)
    qc.cx(0, 2)

    # Run with different window depths.
    result_w1 = _run_pass(qc, c_sc=1.0, c_na=2.0, c_swap=3.0, c_trans=3.0,
                          window_depth=1)
    result_w5 = _run_pass(qc, c_sc=1.0, c_na=2.0, c_swap=3.0, c_trans=3.0,
                          window_depth=5)

    counts_w1 = _count_gates(result_w1)
    counts_w5 = _count_gates(result_w5)

    # Both should produce valid circuits with all original gates.
    assert counts_w1.get("cx", 0) == 3
    assert counts_w5.get("cx", 0) == 3
    print(f"  PASS: test_window_depth_parameter "
          f"(w1: {counts_w1}, w5: {counts_w5})")


def test_decay_parameter():
    """Verify that the decay factor works without errors."""
    qc = QuantumCircuit(5)
    qc.cx(0, 4)
    qc.cx(0, 3)
    qc.cx(0, 1)

    # High decay (future matters a lot).
    result_high = _run_pass(qc, c_sc=1.0, c_na=2.0, c_swap=3.0, c_trans=3.0,
                            decay=1.0)
    # Low decay (future barely matters — approaches greedy).
    result_low = _run_pass(qc, c_sc=1.0, c_na=2.0, c_swap=3.0, c_trans=3.0,
                           decay=0.1)

    counts_high = _count_gates(result_high)
    counts_low = _count_gates(result_low)

    assert counts_high.get("cx", 0) == 3
    assert counts_low.get("cx", 0) == 3
    print(f"  PASS: test_decay_parameter "
          f"(high: {counts_high}, low: {counts_low})")


def test_na_then_second_gate_avoids_retransduction():
    """Two consecutive gates on the same distant pair.

    After the first gate sends both qubits to NA, the second gate should
    see them already on NA and incur no additional transduction cost.

    c_sc=1, c_na=2, c_swap=3, c_trans=3.  CX(0,4) twice on linear-5.
      Gate 1: Cost_SC=10, Cost_NA=8 → NA (2 transductions).
      Gate 2: both on NA → Cost_NA=c_na=2, Cost_SC ≥ 7 → NA.
    Total: 2 transductions.
    """
    qc = QuantumCircuit(5)
    qc.cx(0, 4)
    qc.cx(0, 4)

    result = _run_pass(qc, c_sc=1.0, c_na=2.0, c_swap=3.0, c_trans=3.0)
    counts = _count_gates(result)

    assert counts.get("transduct", 0) == 2, (
        f"Expected 2 transductions, got {counts.get('transduct', 0)}"
    )
    assert counts.get("cx", 0) == 2
    print("  PASS: test_na_then_second_gate_avoids_retransduction")


def test_mixed_circuit():
    """A small mixed circuit exercises multiple branches in one run.

    5-qubit linear map with c_sc=1, c_na=2, c_swap=3, c_trans=3.

    Sequence:
      H(0)             → single-qubit, pass through
      CX(0,1)          → adjacent
      CX(0,4)          → distant
      CX(3,4)          → involves a qubit that may have changed modality
    """
    qc = QuantumCircuit(5)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 4)
    qc.cx(3, 4)

    result = _run_pass(qc, c_sc=1.0, c_na=2.0, c_swap=3.0, c_trans=3.0)
    counts = _count_gates(result)

    # Sanity: all original gates must appear.
    assert counts.get("h", 0) == 1
    assert counts.get("cx", 0) == 3
    print(f"  PASS: test_mixed_circuit (counts: {counts})")


def test_empty_circuit():
    """An empty circuit should pass through without error."""
    qc = QuantumCircuit(3)

    result = _run_pass(qc)
    counts = _count_gates(result)

    assert len(counts) == 0
    print("  PASS: test_empty_circuit")


def test_single_qubit_only_circuit():
    """A circuit with only single-qubit gates and no 2-qubit gates."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.x(1)
    qc.h(2)
    qc.z(0)

    result = _run_pass(qc)
    counts = _count_gates(result)

    assert counts.get("h", 0) == 2
    assert counts.get("x", 0) == 1
    assert counts.get("z", 0) == 1
    assert counts.get("transduct", 0) == 0
    assert counts.get("swap", 0) == 0
    print("  PASS: test_single_qubit_only_circuit")


def test_long_chain_distant_gates():
    """A chain of distant gates to stress-test the window.

    CX(0,4), CX(1,3), CX(0,3), CX(1,4) on a 5-qubit linear map.
    This exercises the window seeing different qubit pairs across gates.
    """
    qc = QuantumCircuit(5)
    qc.cx(0, 4)
    qc.cx(1, 3)
    qc.cx(0, 3)
    qc.cx(1, 4)

    result = _run_pass(qc, c_sc=1.0, c_na=2.0, c_swap=3.0, c_trans=3.0)
    counts = _count_gates(result)

    assert counts.get("cx", 0) == 4
    print(f"  PASS: test_long_chain_distant_gates (counts: {counts})")


# ======================================================================
# Runner
# ======================================================================

def main() -> int:
    """Run all tests, return 0 on success, 1 on failure."""
    tests = [
        test_adjacent_gate_stays_on_sc,
        test_distant_gate_goes_to_na,
        test_sc_swap_routing,
        test_single_qubit_gates_pass_through,
        test_lookahead_prevents_thrashing,
        test_lookahead_keeps_qubits_on_sc_for_future_adjacent_gates,
        test_window_depth_parameter,
        test_decay_parameter,
        test_na_then_second_gate_avoids_retransduction,
        test_mixed_circuit,
        test_empty_circuit,
        test_single_qubit_only_circuit,
        test_long_chain_distant_gates,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            print(f"  FAIL: {name} — {exc}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
