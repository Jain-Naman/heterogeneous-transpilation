"""Tests for the HybridGlobalStaticILP transpilation pass.

Run with:
    python test_hybrid_global_ilp.py
"""

from __future__ import annotations

import sys

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap, Layout

from hybrid_global_ilp import HybridGlobalStaticILP
from hybrid_greedy_router import HybridGreedyRouter, TRANSDUCT


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


def _run_ilp(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap | None = None,
    layout: Layout | None = None,
    init_modality_sc: bool = False,
    **cost_kwargs,
) -> QuantumCircuit:
    """Run HybridGlobalStaticILP on *circuit* and return the result."""
    if coupling_map is None:
        coupling_map = _build_linear_coupling_map(circuit.num_qubits)
    if layout is None:
        layout = _build_trivial_layout(circuit)

    dag = circuit_to_dag(circuit)
    router = HybridGlobalStaticILP(
        coupling_map=coupling_map,
        initial_layout=layout,
        **cost_kwargs,
    )
    res = router.run(dag, init_modality_sc=init_modality_sc)
    routed_dag = res[0] if isinstance(res, tuple) else res
    return dag_to_circuit(routed_dag)


def _run_greedy(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap | None = None,
    layout: Layout | None = None,
    **cost_kwargs,
) -> QuantumCircuit:
    """Run HybridGreedyRouter on *circuit* and return the result."""
    if coupling_map is None:
        coupling_map = _build_linear_coupling_map(circuit.num_qubits)
    if layout is None:
        layout = _build_trivial_layout(circuit)

    dag = circuit_to_dag(circuit)
    router = HybridGreedyRouter(
        coupling_map=coupling_map,
        initial_layout=layout,
        **cost_kwargs,
    )
    routed_dag = router.run(dag)
    return dag_to_circuit(routed_dag)


def _compute_cost(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    layout: Layout,
    c_sc: float,
    c_na: float,
    c_swap: float,
    c_trans: float,
) -> float:
    """Compute the total execution cost of a transpiled circuit.

    Replays the modality state through the circuit and sums:
    - c_trans for each transduct gate
    - c_swap for each SWAP gate
    - gate cost for each 2-qubit gate (c_sc or c_na depending on modality)
    """
    modality: dict[int, str] = {i: "SC" for i in range(qc.num_qubits)}
    total = 0.0

    for instr in qc.data:
        op = instr.operation
        indices = [qc.find_bit(q).index for q in instr.qubits]

        if op.name == "transduct":
            total += c_trans
            qi = indices[0]
            modality[qi] = "NA" if modality[qi] == "SC" else "SC"
        elif op.name == "swap":
            total += c_swap
        elif op.num_qubits == 2:
            # 2-qubit gate: cost depends on current modality of first qubit.
            tag = modality[indices[0]]
            if tag == "SC":
                total += c_sc
            else:
                total += c_na

    return total


# ======================================================================
# Test cases — basic behaviour
# ======================================================================

def test_adjacent_gate_stays_on_sc():
    """Adjacent qubits on the coupling map → SC is cheapest.

    With defaults (c_sc=1, c_na=5, c_swap=3, c_trans=10):
      Cost_SC = c_sc = 1
      Cost_NA = c_na = 5  (no transduction penalty with free initial state,
                           but the ILP should still prefer SC since c_sc < c_na)
    → expect 0 transductions, 0 SWAPs.
    """
    qc = QuantumCircuit(5)
    qc.cx(0, 1)  # adjacent on linear map

    result = _run_ilp(qc)
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
      Cost on SC = c_sc + c_swap·(4-1) = 1 + 9 = 10
      Cost on NA = c_na = 2  (free initial state — both start on NA)
    → ILP should pick NA.
    """
    qc = QuantumCircuit(5)
    qc.cx(0, 4)

    result = _run_ilp(qc, c_sc=1.0, c_na=2.0, c_swap=3.0, c_trans=3.0)
    counts = _count_gates(result)

    # ILP with free initial state: both qubits can start on NA → 0 transductions.
    assert counts.get("transduct", 0) == 0, (
        f"Expected 0 transductions (free initial state), "
        f"got {counts.get('transduct', 0)}"
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

    With free initial state the ILP puts idle qubits (q1, q2, q4) on NA,
    vacating physical nodes 1, 2, 4.  The path 0→1→2→3 has all-vacant
    intermediate nodes, so SWAPs become free MOVEs (0 SwapGates).
    """
    qc = QuantumCircuit(5)
    qc.cx(0, 3)

    result = _run_ilp(qc, c_sc=1.0, c_na=100.0, c_swap=1.0, c_trans=100.0)
    counts = _count_gates(result)

    assert counts.get("transduct", 0) == 0, (
        f"Expected 0 transductions, got {counts.get('transduct', 0)}"
    )
    # Free initial state → idle qubits on NA → MOVEs, not SWAPs.
    assert counts.get("swap", 0) == 0, (
        f"Expected 0 SWAPs (MOVEs replace SWAPs), got {counts.get('swap', 0)}"
    )
    assert counts.get("cx", 0) == 1
    print("  PASS: test_sc_swap_routing")


def test_sc_swap_routing_forced():
    """Force genuine SWAPs by making all qubits participate in gates.

    Two adjacent gates CX(0,1) and CX(2,3) force all four qubits onto SC.
    Then CX(0,3) at distance 3 has occupied intermediate nodes → real SWAPs.
    c_sc=1, c_na=100, c_swap=1, c_trans=100.
    """
    qc = QuantumCircuit(5)
    qc.cx(0, 1)  # forces q0, q1 on SC
    qc.cx(2, 3)  # forces q2, q3 on SC
    qc.cx(0, 3)  # d=3, path through occupied nodes → real SWAPs

    result = _run_ilp(qc, c_sc=1.0, c_na=100.0, c_swap=1.0, c_trans=100.0)
    counts = _count_gates(result)

    assert counts.get("cx", 0) == 3
    assert counts.get("transduct", 0) == 0
    # q0 on node 0, q3 on node 3, path 0→1→2→3, q1 on 1 and q2 on 2
    # → 2 real SWAPs needed.
    assert counts.get("swap", 0) == 2, (
        f"Expected 2 SWAPs, got {counts.get('swap', 0)}"
    )
    print("  PASS: test_sc_swap_routing_forced")


def test_single_qubit_gates_pass_through():
    """Single-qubit gates should always be forwarded unchanged."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.x(1)
    qc.z(2)

    result = _run_ilp(qc)
    counts = _count_gates(result)

    assert counts.get("h", 0) == 1
    assert counts.get("x", 0) == 1
    assert counts.get("z", 0) == 1
    assert counts.get("transduct", 0) == 0
    assert counts.get("swap", 0) == 0
    print("  PASS: test_single_qubit_gates_pass_through")


def test_empty_circuit():
    """An empty circuit should pass through without error."""
    qc = QuantumCircuit(3)

    result = _run_ilp(qc)
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

    result = _run_ilp(qc)
    counts = _count_gates(result)

    assert counts.get("h", 0) == 2
    assert counts.get("x", 0) == 1
    assert counts.get("z", 0) == 1
    assert counts.get("transduct", 0) == 0
    assert counts.get("swap", 0) == 0
    print("  PASS: test_single_qubit_only_circuit")


# ======================================================================
# Test cases — multi-gate and transition behaviour
# ======================================================================

def test_two_gates_same_pair():
    """Two consecutive gates on the same distant pair.

    CX(0,4) twice on linear-5, c_sc=1, c_na=2, c_swap=3, c_trans=3.
    ILP should assign both to NA with free initial state → 0 transductions.
    """
    qc = QuantumCircuit(5)
    qc.cx(0, 4)
    qc.cx(0, 4)

    result = _run_ilp(qc, c_sc=1.0, c_na=2.0, c_swap=3.0, c_trans=3.0)
    counts = _count_gates(result)

    assert counts.get("cx", 0) == 2
    # Free initial state: both qubits start on NA, no transductions needed.
    assert counts.get("transduct", 0) == 0, (
        f"Expected 0 transductions, got {counts.get('transduct', 0)}"
    )
    print("  PASS: test_two_gates_same_pair")


def test_mixed_circuit():
    """A small mixed circuit with adjacent and distant gates.

    5-qubit linear map with c_sc=1, c_na=2, c_swap=3, c_trans=3.

    Sequence:
      H(0)             → single-qubit
      CX(0,1)          → adjacent (d=1)
      CX(0,4)          → distant  (d=4)
      CX(3,4)          → adjacent (d=1)
    """
    qc = QuantumCircuit(5)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 4)
    qc.cx(3, 4)

    result = _run_ilp(qc, c_sc=1.0, c_na=2.0, c_swap=3.0, c_trans=3.0)
    counts = _count_gates(result)

    # All original gates must appear.
    assert counts.get("h", 0) == 1
    assert counts.get("cx", 0) == 3
    print(f"  PASS: test_mixed_circuit (counts: {counts})")


def test_modality_transition_occurs():
    """Verify transductions are inserted when a modality switch is forced.

    CX(0,1) adjacent (d=1) then CX(0,4) distant (d=4).

    With c_sc=1, c_na=20, c_swap=10, c_trans=2:
      Layer 0 CX(0,1): SC cost coeff = (1 - 20)/2 = -9.5  (SC strongly favoured)
      Layer 1 CX(0,4): SC cost coeff = (1+30 - 20)/2 = 5.5 (NA strongly favoured)
      Both-NA cost = 20 + 20 = 40
      SC-then-NA   = 1 + 2(trans q0) + 2(trans q1) + 20 = 25 → cheaper!
    The ILP must switch qubit 0 from SC to NA → at least 1 transduction.
    """
    qc = QuantumCircuit(5)
    qc.cx(0, 1)
    qc.cx(0, 4)

    result = _run_ilp(qc, c_sc=1.0, c_na=20.0, c_swap=10.0, c_trans=2.0)
    counts = _count_gates(result)

    assert counts.get("cx", 0) == 2
    # Qubit 0 switches SC → NA between layers → transductions required.
    assert counts.get("transduct", 0) >= 1, (
        f"Expected at least 1 transduction for modality switch, "
        f"got {counts.get('transduct', 0)}"
    )
    print(f"  PASS: test_modality_transition_occurs (counts: {counts})")


# ======================================================================
# Test cases — optimality vs. greedy
# ======================================================================

def test_ilp_cost_leq_greedy_adjacent_chain():
    """ILP cost should be ≤ greedy cost for an all-adjacent chain."""
    qc = QuantumCircuit(5)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    costs = dict(c_sc=1.0, c_na=5.0, c_swap=3.0, c_trans=10.0)
    cm = _build_linear_coupling_map(5)
    layout = _build_trivial_layout(qc)

    ilp_qc = _run_ilp(qc, cm, layout, **costs)
    greedy_qc = _run_greedy(qc, cm, layout, **costs)

    ilp_cost = _compute_cost(ilp_qc, cm, layout, **costs)
    greedy_cost = _compute_cost(greedy_qc, cm, layout, **costs)

    assert ilp_cost <= greedy_cost + 1e-9, (
        f"ILP cost {ilp_cost} > greedy cost {greedy_cost}"
    )
    print(f"  PASS: test_ilp_cost_leq_greedy_adjacent_chain "
          f"(ILP={ilp_cost}, greedy={greedy_cost})")


def test_ilp_cost_leq_greedy_distant():
    """ILP cost should be ≤ greedy cost for a distant-gate circuit."""
    qc = QuantumCircuit(5)
    qc.cx(0, 4)
    qc.cx(0, 1)

    costs = dict(c_sc=1.0, c_na=2.0, c_swap=3.0, c_trans=3.0)
    cm = _build_linear_coupling_map(5)
    layout = _build_trivial_layout(qc)

    ilp_qc = _run_ilp(qc, cm, layout, **costs)
    greedy_qc = _run_greedy(qc, cm, layout, **costs)

    ilp_cost = _compute_cost(ilp_qc, cm, layout, **costs)
    greedy_cost = _compute_cost(greedy_qc, cm, layout, **costs)

    assert ilp_cost <= greedy_cost + 1e-9, (
        f"ILP cost {ilp_cost} > greedy cost {greedy_cost}"
    )
    print(f"  PASS: test_ilp_cost_leq_greedy_distant "
          f"(ILP={ilp_cost}, greedy={greedy_cost})")


def test_ilp_cost_leq_greedy_mixed():
    """ILP cost ≤ greedy cost on a mixed circuit across multiple regimes."""
    qc = QuantumCircuit(5)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(2, 4)
    qc.h(2)
    qc.cx(0, 3)
    qc.cx(1, 4)
    qc.cx(3, 4)
    qc.cx(0, 4)

    regimes = {
        "A": dict(c_sc=1.0, c_na=5.0, c_swap=3.0, c_trans=10.0),
        "B": dict(c_sc=1.0, c_na=3.0, c_swap=3.0, c_trans=5.0),
        "C": dict(c_sc=1.0, c_na=2.0, c_swap=3.0, c_trans=2.0),
        "D": dict(c_sc=1.0, c_na=1.5, c_swap=3.0, c_trans=1.0),
    }

    cm = _build_linear_coupling_map(5)
    layout = _build_trivial_layout(qc)

    for label, costs in regimes.items():
        ilp_qc = _run_ilp(qc, cm, layout, **costs)
        greedy_qc = _run_greedy(qc, cm, layout, **costs)

        ilp_cost = _compute_cost(ilp_qc, cm, layout, **costs)
        greedy_cost = _compute_cost(greedy_qc, cm, layout, **costs)

        assert ilp_cost <= greedy_cost + 1e-9, (
            f"Regime {label}: ILP cost {ilp_cost} > greedy cost {greedy_cost}"
        )
        print(f"    Regime {label}: ILP={ilp_cost:.1f}, greedy={greedy_cost:.1f}")

    print("  PASS: test_ilp_cost_leq_greedy_mixed")


def test_long_chain_distant_gates():
    """A chain of distant gates to stress-test the ILP."""
    qc = QuantumCircuit(5)
    qc.cx(0, 4)
    qc.cx(1, 3)
    qc.cx(0, 3)
    qc.cx(1, 4)

    result = _run_ilp(qc, c_sc=1.0, c_na=2.0, c_swap=3.0, c_trans=3.0)
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
        test_sc_swap_routing_forced,
        test_single_qubit_gates_pass_through,
        test_empty_circuit,
        test_single_qubit_only_circuit,
        test_two_gates_same_pair,
        test_mixed_circuit,
        test_modality_transition_occurs,
        test_ilp_cost_leq_greedy_adjacent_chain,
        test_ilp_cost_leq_greedy_distant,
        test_ilp_cost_leq_greedy_mixed,
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
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
