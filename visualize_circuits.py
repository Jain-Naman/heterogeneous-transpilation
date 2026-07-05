"""Visualize transpiled circuits with modality annotations.

Produces colour-coded circuit diagrams showing which modality (SC or NA)
each gate executes on, for all three compilation methods: SC-only,
Hybrid Greedy, and Hybrid Lookahead.

Gate colour scheme
------------------
* **Blue**   — SC (superconducting) 2-qubit gates
* **Amber**  — NA (neutral-atom) 2-qubit gates
* **Purple** — SWAP gates (SC routing overhead)
* **Red**    — Transduction markers (modality switch)
* **Grey**   — Single-qubit gates (modality-agnostic)

Usage
-----
    python visualize_circuits.py [--output-dir DIR] [--regime A|B|C|D]
"""

from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.image as mpimg         # noqa: E402

from qiskit import QuantumCircuit                                  # noqa: E402
from qiskit.circuit import Instruction                             # noqa: E402
from qiskit.converters import circuit_to_dag, dag_to_circuit       # noqa: E402
from qiskit.transpiler import CouplingMap, Layout                  # noqa: E402
from qiskit.transpiler.preset_passmanagers import (                # noqa: E402
    generate_preset_pass_manager,
)

from hybrid_greedy_router import HybridGreedyRouter                # noqa: E402
from hybrid_lookahead_router import HybridLookaheadRouter          # noqa: E402
from hybrid_global_ilp import HybridGlobalStaticILP                # noqa: E402


# ======================================================================
# Colour palette
# ======================================================================

SC_CLR    = ("#3B82F6", "#FFFFFF")   # Blue  – Superconducting
NA_CLR    = ("#F59E0B", "#000000")   # Amber – Neutral Atom
SWAP_CLR  = ("#7C3AED", "#FFFFFF")   # Purple – SWAP routing
TRANS_CLR = ("#EF4444", "#FFFFFF")   # Red   – Transduction marker
SQ_CLR    = ("#9CA3AF", "#FFFFFF")   # Grey  – single-qubit (neutral)

_2Q_BASES = ("cx", "cz", "ecr", "rzz", "cp")
_1Q_BASES = (
    "h", "x", "y", "z", "s", "t", "sdg", "tdg",
    "sx", "rz", "ry", "rx", "id", "u", "u1", "u2", "u3",
)


def _hybrid_style() -> dict:
    """Qiskit ``draw(output='mpl')`` style for modality-annotated circuits."""
    dc: dict[str, tuple[str, str]] = {}
    for g in _2Q_BASES:
        dc[f"{g} [SC]"] = SC_CLR
        dc[f"{g} [NA]"] = NA_CLR
    dc["swap [SC]"] = SWAP_CLR
    dc["swap [NA]"] = NA_CLR
    dc["TRANS"] = TRANS_CLR
    for g in _1Q_BASES:
        dc[g] = SQ_CLR
    return {"displaycolor": dc}


def _sc_only_style() -> dict:
    """Style for SC-only circuits (all 2-qubit gates coloured blue)."""
    dc: dict[str, tuple[str, str]] = {}
    for g in _2Q_BASES:
        dc[g] = SC_CLR
    dc["swap"] = SWAP_CLR
    for g in _1Q_BASES:
        dc[g] = SQ_CLR
    return {"displaycolor": dc}


# ======================================================================
# Modality annotation
# ======================================================================

def annotate_hybrid_circuit(qc: QuantumCircuit, modality_state: dict[int, str] | None = None) -> QuantumCircuit:
    """Replay modality state through *qc* and return a copy whose gate
    names are tagged with their execution modality (e.g. ``cx [SC]``).

    ``transduct`` instructions are replaced with visible ``TRANS`` markers.
    """
    modality: dict[int, str] = {i: "SC" for i in range(qc.num_qubits)}
    if modality_state:
        modality = modality_state

    new_qc = QuantumCircuit(qc.num_qubits)

    for instr in qc.data:
        op = instr.operation
        indices = [qc.find_bit(q).index for q in instr.qubits]

        if op.name == "transduct":
            qi = indices[0]
            modality[qi] = "NA" if modality[qi] == "SC" else "SC"
            new_qc.append(Instruction("TRANS", 1, 0, []), indices)
        elif op.num_qubits >= 2:
            tag = modality[indices[0]]
            new_qc.append(
                Instruction(f"{op.name} [{tag}]", op.num_qubits, 0, []),
                indices,
            )
        else:
            new_qc.append(op, indices)

    return new_qc


# ======================================================================
# Helpers
# ======================================================================

def _count_ops(qc: QuantumCircuit) -> dict[str, int]:
    counts: dict[str, int] = {}
    for instr in qc.data:
        name = instr.operation.name
        counts[name] = counts.get(name, 0) + 1
    return counts


def _draw_to_buf(qc: QuantumCircuit, style: dict, title: str = "") -> BytesIO:
    """Render *qc* with the Qiskit matplotlib backend and return PNG bytes."""
    fig = qc.draw(output="mpl", style=style, fold=-1)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    buf.seek(0)
    return buf


def _save_buf(buf: BytesIO, path: str) -> None:
    buf.seek(0)
    with open(path, "wb") as f:
        f.write(buf.read())
    print(f"  → saved {path}")

# ======================================================================
# Cost regimes (mirrors benchmark_comparison.py)
# ======================================================================

COST_REGIMES: dict[str, dict[str, float]] = {
    "A": dict(c_sc=1.0, c_na=5.0, c_swap=3.0, c_trans=10.0),
    "B": dict(c_sc=1.0, c_na=3.0, c_swap=3.0, c_trans=5.0),
    "C": dict(c_sc=1.0, c_na=2.0, c_swap=3.0, c_trans=2.0),
    "D": dict(c_sc=1.0, c_na=1.5, c_swap=3.0, c_trans=1.0),
}

# ======================================================================
# Example circuits 
# ======================================================================

def simple1():
    qc = QuantumCircuit(5)
    qc.h(0)
    qc.cx(0, 1)          # adjacent  (d = 1)
    qc.cx(2, 4)          # moderate  (d = 2)
    qc.h(2)
    qc.cx(0, 3)          # distant   (d = 3)
    qc.cx(1, 4)          # distant   (d = 3)
    qc.cx(3, 4)          # adjacent  (d = 1)
    qc.cx(0, 4)          # very far  (d = 4)
    return qc

def simple2():
    qc = QuantumCircuit(5)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.cx(0, 2)
    qc.cx(2, 0)

    qc.cx(2, 4)
    qc.h(2)
    qc.cx(0, 4)
    qc.cx(0, 3)
    qc.cx(4, 0)

    qc.x(3)
    qc.cx(0, 1)
    qc.cx(1, 0)
    return qc

def ref_example():
    qc = QuantumCircuit(7)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(3, 4)
    qc.cx(0, 2)
    qc.cx(1, 3)
    qc.cx(2, 4)
    qc.cx(1, 2)
    qc.cx(3, 4)
    qc.cx(5, 6)
    qc.cx(4, 5)
    qc.cx(5, 6)
    qc.cx(4, 6)
    qc.cx(0, 4)
    qc.cx(2, 6)
    qc.cx(1, 5)
    qc.cx(0, 1)
    return qc

def quick_switching_example():
    qc = QuantumCircuit(6)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(0, 4)
    qc.cx(2, 5)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(0, 4)
    qc.cx(2, 5)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(0, 4)
    qc.cx(2, 5)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(0, 4)
    qc.cx(2, 5)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(0, 4)
    qc.cx(2, 5)
    return qc

def multi_layer_heavy_swaps():
    qc = QuantumCircuit(8)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(1, 2)
    qc.cx(1, 0)
    qc.cx(1, 3)
    qc.cx(3, 0)

    qc.cx(0, 7)
    qc.cx(1, 7)
    qc.cx(6, 7)
    qc.cx(7, 0)

    qc.cx(3, 4)
    qc.cx(5, 4)
    qc.cx(5, 7)
    qc.cx(6, 5)

    return qc


# ======================================================================
# Main
# ======================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize modality-annotated transpiled circuits.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./visualizations",
        help="Directory for output images (default: ./visualizations ).",
    )
    parser.add_argument(
        "--regime", type=str, default="C", choices=list(COST_REGIMES),
        help="Cost-parameter regime A/B/C/D (default: C).",
    )
    parser.add_argument(
        "--window-depth", type=int, default=5,
        help="Window depth for lookahead router (default: 5).",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    costs = COST_REGIMES[args.regime]

    # ---- example circuit ----
    # qc = multi_layer_heavy_swaps()
    qc = quick_switching_example()

    cm = CouplingMap.from_line(qc.num_qubits)
    layout = Layout.from_intlist(list(range(qc.num_qubits)), qc.qregs[0])

    print(f"Circuit : {qc.num_qubits} qubits, {len(qc.data)} gates")
    print(f"Topology: linear-{qc.num_qubits}")
    print(f"Regime  : {args.regime}  {costs}")
    print()

    # ---- 1. Original ----
    buf_orig = _draw_to_buf(qc, {}, "Original Circuit")
    _save_buf(buf_orig, str(out / "circuit_original.png"))
    print(f"     ops: {_count_ops(qc)}")

    # ---- 2. SC-only ----
    pm = generate_preset_pass_manager(
        optimization_level=1, coupling_map=cm, seed_transpiler=42, initial_layout=layout,
    )
    sc_qc = pm.run(qc)
    buf_sc = _draw_to_buf(sc_qc, _sc_only_style(), "SC-Only Transpilation")
    _save_buf(buf_sc, str(out / "circuit_sc_only.png"))
    print(f"     ops: {_count_ops(sc_qc)}")

    # ---- 3. Hybrid Greedy ----
    greedy_qc = dag_to_circuit(
        HybridGreedyRouter(coupling_map=cm, initial_layout=layout, **costs)
        .run(circuit_to_dag(qc))
    )
    greedy_ann = annotate_hybrid_circuit(greedy_qc)
    buf_greedy = _draw_to_buf(
        greedy_ann, _hybrid_style(), "Hybrid Greedy Router",
    )
    _save_buf(buf_greedy, str(out / "circuit_greedy.png"))
    print(f"     ops: {_count_ops(greedy_qc)}")

    # ---- 4. Hybrid Lookahead ----
    la_qc = dag_to_circuit(
        HybridLookaheadRouter(coupling_map=cm, initial_layout=layout, window_depth=args.window_depth, **costs)
        .run(circuit_to_dag(qc))
    )
    la_ann = annotate_hybrid_circuit(la_qc)
    buf_la = _draw_to_buf(
        la_ann, _hybrid_style(), "Hybrid Lookahead Router",
    )
    _save_buf(buf_la, str(out / "circuit_lookahead.png"))
    print(f"     ops: {_count_ops(la_qc)}")

    # ---- 5. Hybrid Global ILP ----
    transfomred_dag = HybridGlobalStaticILP(coupling_map=cm, initial_layout=layout, **costs).run(circuit_to_dag(qc))
    ilp_qc = dag_to_circuit(transfomred_dag)
    ilp_ann = annotate_hybrid_circuit(ilp_qc)
    buf_ilp = _draw_to_buf(
        ilp_ann, _hybrid_style(), "Hybrid Global ILP Router",
    )
    _save_buf(buf_ilp, str(out / "circuit_global_ilp.png"))
    print(f"     ops: {_count_ops(ilp_qc)}")

    print()
    print("Output files:")
    for name in ("circuit_original", "circuit_sc_only", "circuit_greedy",
                  "circuit_lookahead", "circuit_global_ilp"):
        print(f"  {out / name}.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
