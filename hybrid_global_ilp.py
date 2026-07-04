"""Hybrid SC/NA Global ILP Transpiler Pass (Step 1 – Static Assignment).

This module implements a Qiskit ``TransformationPass`` that finds the
**globally optimal** modality assignment (Superconducting vs. Neutral-Atom)
for every qubit at every circuit layer, by formulating and solving a
Mixed-Integer Linear Program (MILP).

Unlike the greedy or look-ahead heuristics that make local decisions,
this pass considers the *entire* circuit simultaneously and minimises the
total cost — gate execution costs on each platform plus transduction
penalties for switching a qubit between modalities.

Key assumptions (Step 1)
------------------------
* **Free initial state** – the optimiser may freely choose the starting
  modality of each qubit without incurring a transduction penalty.
* **Static assignment** – physical qubit distances on the SC chip are
  computed from the ``initial_layout`` and ``coupling_map`` only; the
  optimiser does *not* re-route or re-map qubits.  SWAPs are still
  inserted in the output DAG when an SC gate requires non-adjacent
  qubits.

Hardware assumptions
--------------------
* **SC**: Restricted physical topology (``CouplingMap``).  Fast operations,
  but requires SWAP routing.
* **NA**: Idealised all-to-all connectivity.  Slower operations, no routing
  overhead.
* **Transduction**: Moving a qubit between SC and NA incurs a flat penalty.

Cost parameters
---------------
* ``c_sc``    – base cost of a 2-qubit gate on SC.
* ``c_na``    – base cost of a 2-qubit gate on NA.
* ``c_swap``  – cost of a SWAP gate on SC.
* ``c_trans`` – cost of moving **one** logical qubit between SC and NA.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import LinearConstraint, Bounds, milp
from scipy.sparse import lil_matrix

from qiskit.circuit import Qubit
from qiskit.circuit.library import SwapGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.basepasses import TransformationPass

from hybrid_greedy_router import TRANSDUCT


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

class HybridGlobalStaticILP(TransformationPass):
    """Find the globally optimal SC/NA modality assignment via MILP.

    For a circuit with *N* logical qubits partitioned into *T* layers the
    pass constructs a MILP with:

    * :math:`N \\times T` binary **x** variables (``x[i,t] = 1`` ⇔ qubit
      *i* is on SC at layer *t*).
    * :math:`N \\times (T-1)` continuous **z** auxiliary variables that
      linearise the absolute-value transduction cost
      :math:`z_{i,t} = |x_{i,t} - x_{i,t-1}|`.

    The MILP is solved with ``scipy.optimize.milp`` (HiGHS back-end).

    Parameters
    ----------
    coupling_map : CouplingMap
        The physical topology of the SC chip.
    initial_layout : Layout
        Initial mapping of logical qubits to physical SC node indices.
    c_sc : float
        Base cost of a 2-qubit gate on SC.
    c_na : float
        Base cost of a 2-qubit gate on NA.
    c_swap : float
        Cost of a single SWAP gate on SC.
    c_trans : float
        Cost of transducing **one** logical qubit between modalities.
    """

    def __init__(
        self,
        coupling_map: CouplingMap,
        initial_layout: Layout,
        c_sc: float = 1.0,
        c_na: float = 5.0,
        c_swap: float = 3.0,
        c_trans: float = 10.0,
    ) -> None:
        super().__init__()
        self.coupling_map = coupling_map
        self.initial_layout = initial_layout
        self.c_sc = c_sc
        self.c_na = c_na
        self.c_swap = c_swap
        self.c_trans = c_trans

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def run(self, dag: DAGCircuit, init_modality_sc = True) -> DAGCircuit:
        """Transform *dag* using the globally optimal modality assignment.

        Parameters
        ----------
        dag : DAGCircuit
            The input circuit DAG.  Only 1-qubit and 2-qubit operations
            are expected.

        Returns
        -------
        DAGCircuit
            A new DAG containing the original gates plus injected
            ``TRANSDUCT`` and ``SwapGate`` operations.
        """
        new_dag: DAGCircuit = dag.copy_empty_like()

        qubit_list = list(dag.qubits)
        qubit_to_idx = {q: i for i, q in enumerate(qubit_list)}
        N = len(qubit_list)

        # --- Build initial logical → physical mapping ---
        logical_to_physical_init: dict[Qubit, int] = {}
        for physical_idx in range(self.coupling_map.size()):
            qubit = self.initial_layout[physical_idx]
            logical_to_physical_init[qubit] = physical_idx

        # --- Extract layers ---
        layers_data: list[list[tuple]] = []
        for layer_dict in dag.layers():
            layer_ops = []
            for node in layer_dict["graph"].topological_op_nodes():
                layer_ops.append((node.op, node.qargs, node.cargs))
            layers_data.append(layer_ops)

        T = len(layers_data)
        if T == 0:
            return new_dag

        # --- Collect 2-qubit gate info for the MILP ---
        gate_info: list[tuple[int, int, int, int]] = []
        for t, layer_ops in enumerate(layers_data):
            for op, qargs, _cargs in layer_ops:
                if op.num_qubits == 2:
                    qi, qj = qargs[0], qargs[1]
                    i, j = qubit_to_idx[qi], qubit_to_idx[qj]
                    phys_i = logical_to_physical_init[qi]
                    phys_j = logical_to_physical_init[qj]
                    dist = self.coupling_map.distance(phys_i, phys_j)
                    gate_info.append((t, i, j, dist))

        if not gate_info:
            # No 2-qubit gates — just pass everything through unchanged.
            for layer_ops in layers_data:
                for op, qargs, cargs in layer_ops:
                    new_dag.apply_operation_back(op, qargs, cargs)
            return new_dag

        # --- Solve the MILP ---
        x_sol = self._solve_milp(N, T, gate_info, init_modality_sc)
                
        # --- Reconstruct DAG from the optimal solution ---
        out =  self._reconstruct_dag(
            new_dag, x_sol, N, T,
            qubit_list, qubit_to_idx,
            layers_data, logical_to_physical_init,
        )

        return out

    # ------------------------------------------------------------------
    # MILP construction and solving
    # ------------------------------------------------------------------

    def _solve_milp(
        self,
        N: int,
        T: int,
        gate_info: list[tuple[int, int, int, int]],
        init_modality_sc = True
    ) -> np.ndarray:
        """Build and solve the modality-assignment MILP.

        Parameters
        ----------
        N : int
            Number of logical qubits.
        T : int
            Number of circuit layers.
        gate_info : list of (t, i, j, dist)
            Each entry describes a 2-qubit gate: layer index *t*, qubit
            indices *i* and *j*, and the static coupling-map distance
            *dist* between their physical positions.

        Returns
        -------
        np.ndarray of shape (N, T)
            The optimal modality assignment.  ``x[i, t] == 1`` means
            qubit *i* is on SC at layer *t*; ``0`` means NA.
        """
        num_x = N * T
        num_z = N * (T - 1) if T > 1 else 0
        num_vars = num_x + num_z

        # --- Objective vector c ---
        c_vec = np.zeros(num_vars)

        for t, i, j, dist in gate_info:
            c_sc_ij = self.c_sc + self.c_swap * max(dist - 1, 0)
            alpha = (c_sc_ij - self.c_na) / 2.0
            c_vec[i * T + t] += alpha
            c_vec[j * T + t] += alpha

        for i in range(N):
            for t_off in range(T - 1):
                z_idx = num_x + i * (T - 1) + t_off
                c_vec[z_idx] = self.c_trans

        # --- Constraints ---
        num_gate_eq = len(gate_info)
        num_xor = 2 * N * (T - 1) if T > 1 else 0
        num_constraints = num_gate_eq + num_xor

        A = lil_matrix((num_constraints, num_vars))
        lb = np.zeros(num_constraints)
        ub = np.zeros(num_constraints)

        row = 0

        # Gate equality constraints: x_{i,t} - x_{j,t} = 0
        for t, i, j, _dist in gate_info:
            A[row, i * T + t] = 1.0
            A[row, j * T + t] = -1.0
            lb[row] = 0.0
            ub[row] = 0.0
            row += 1

        # XOR linearisation constraints (t >= 1 only)
        if T > 1:
            for i in range(N):
                for t in range(1, T):
                    x_it = i * T + t
                    x_it_prev = i * T + (t - 1)
                    z_it = num_x + i * (T - 1) + (t - 1)

                    # Row A: z_{i,t} - x_{i,t} + x_{i,t-1} >= 0
                    A[row, z_it] = 1.0
                    A[row, x_it] = -1.0
                    A[row, x_it_prev] = 1.0
                    lb[row] = 0.0
                    ub[row] = np.inf
                    row += 1

                    # Row B: z_{i,t} + x_{i,t} - x_{i,t-1} >= 0
                    A[row, z_it] = 1.0
                    A[row, x_it] = 1.0
                    A[row, x_it_prev] = -1.0
                    lb[row] = 0.0
                    ub[row] = np.inf
                    row += 1

        # --- Integrality and bounds ---
        integrality = np.zeros(num_vars)
        integrality[:num_x] = 1  # x variables are integer (binary)

        
        if init_modality_sc:
            # Force initalization to SC
            var_lb = np.zeros(num_vars)
            for i in range(N):
                var_lb[i * T] = 1.0
            bounds = Bounds(lb=var_lb, ub=1)
        else:
            # Free initialization
            bounds = Bounds(lb=0, ub=1)

        # --- Solve ---
        constraints = LinearConstraint(A.tocsc(), lb, ub)
        result = milp(
            c=c_vec,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
        )

        if not result.success:
            raise RuntimeError(
                f"MILP solver failed: {result.message}"
            )

        # Extract and round the x portion of the solution.
        x_sol = np.round(result.x[:num_x]).astype(int).reshape(N, T)
        return x_sol

    # ------------------------------------------------------------------
    # DAG reconstruction from the optimal solution
    # ------------------------------------------------------------------

    def _reconstruct_dag(
        self,
        new_dag: DAGCircuit,
        x_sol: np.ndarray,
        N: int,
        T: int,
        qubit_list: list[Qubit],
        qubit_to_idx: dict[Qubit, int],
        layers_data: list[list[tuple]],
        logical_to_physical_init: dict[Qubit, int],
    ) -> DAGCircuit:
        """Walk the layer schedule and emit gates into *new_dag*.

        For each layer, modality transitions are applied first (inserting
        ``TRANSDUCT`` markers), then gates are emitted with SWAP routing
        for SC 2-qubit gates.
        """
        # --- Live state trackers (same as greedy/lookahead) ---
        modality_state: dict[Qubit, str] = {}
        logical_to_physical: dict[Qubit, int] = {}
        physical_to_logical: dict[int, Optional[Qubit]] = {}

        for physical_idx in range(self.coupling_map.size()):
            qubit = self.initial_layout[physical_idx]
            logical_to_physical[qubit] = physical_idx
            physical_to_logical[physical_idx] = qubit

        # --- Set initial modalities from x_sol[:, 0] (free, no cost) ---
        for i, q in enumerate(qubit_list):
            if x_sol[i, 0] == 1:
                modality_state[q] = "SC"
            else:
                modality_state[q] = "NA"
                # Vacate the SC node (qubit starts on NA).
                phys = logical_to_physical[q]
                physical_to_logical[phys] = None
            
        # --- Process layers ---
        for t in range(T):
            # ── Inter-layer transitions (t ≥ 1) ──
            if t > 0:
                # Phase 1: SC → NA  (frees physical nodes first).
                for i in range(N):
                    if x_sol[i, t - 1] == 1 and x_sol[i, t] == 0:
                        q = qubit_list[i]
                        phys = logical_to_physical[q]
                        new_dag.apply_operation_back(TRANSDUCT, [new_dag.qubits[phys]], [])
                        physical_to_logical[phys] = None
                        modality_state[q] = "NA"

                # Phase 2: NA → SC  (uses newly freed nodes).
                na_to_sc = [
                    qubit_list[i]
                    for i in range(N)
                    if x_sol[i, t - 1] == 0 and x_sol[i, t] == 1
                ]
                if na_to_sc:
                    already_assigned: dict[Qubit, int] = {}
                    for q in na_to_sc:
                        target = self._find_sc_position(
                            q, t, layers_data, qubit_to_idx,
                            logical_to_physical, physical_to_logical,
                            logical_to_physical_init, already_assigned,
                        )
                        already_assigned[q] = target
                        new_dag.apply_operation_back(TRANSDUCT, [new_dag.qubits[logical_to_physical[q]]], [])
                        logical_to_physical[q] = target
                        physical_to_logical[target] = q
                        modality_state[q] = "SC"

            # ── Process gates in this layer ──
            for op, qargs, cargs in layers_data[t]:
                if op.num_qubits == 1:
                    new_dag.apply_operation_back(op, qargs, cargs)
                    continue

                assert op.num_qubits == 2, (
                    f"HybridGlobalStaticILP only handles 1- and 2-qubit "
                    f"gates, got {op.num_qubits}-qubit gate '{op.name}'."
                )

                qi, qj = qargs[0], qargs[1]
                i = qubit_to_idx[qi]

                if x_sol[i, t] == 1:
                    # ── Execute on SC: route SWAPs if needed ──
                    phys_i = logical_to_physical[qi]
                    phys_j = logical_to_physical[qj]
                    distance = self.coupling_map.distance(phys_i, phys_j)

                    if distance > 1:
                        path = list(
                            self.coupling_map.shortest_undirected_path(
                                phys_i, phys_j,
                            )
                        )
                        self._route_and_update_state(
                            qi, qj, path, new_dag,
                            logical_to_physical, physical_to_logical,
                        )

                # (NA gates need no routing — all-to-all connectivity.)
                new_dag.apply_operation_back(op, [new_dag.qubits[logical_to_physical[q]] for q in qargs], cargs)

        return new_dag

    # ------------------------------------------------------------------
    # SC position assignment for NA → SC transitions
    # ------------------------------------------------------------------

    def _find_sc_position(
        self,
        qubit: Qubit,
        t: int,
        layers_data: list[list[tuple]],
        qubit_to_idx: dict[Qubit, int],
        logical_to_physical: dict[Qubit, int],
        physical_to_logical: dict[int, Optional[Qubit]],
        logical_to_physical_init: dict[Qubit, int],
        already_assigned: dict[Qubit, int],
    ) -> int:
        """Choose a vacant SC node for a qubit transitioning NA → SC.

        Strategy:
        1. If the qubit participates in a 2-qubit gate this layer, place
           it near the partner qubit (``_find_closest_vacant_node``).
        2. Otherwise, prefer the qubit's original position from the
           initial layout; fall back to any vacant node.

        Returns the chosen physical node index.
        """
        # Look for a gate partner in the current layer.
        partner: Optional[Qubit] = None
        for op, qargs, _cargs in layers_data[t]:
            if op.num_qubits == 2 and qubit in qargs:
                partner = qargs[1] if qargs[0] is qubit else qargs[0]
                break

        claimed: set[int] = set(already_assigned.values())

        # Gather all vacant physical nodes.
        vacant_nodes = [
            node for node, occupant in physical_to_logical.items()
            if occupant is None and node not in claimed
        ]
        if not vacant_nodes:
            # Fallback: any node not already claimed this round.
            vacant_nodes = [
                node for node in physical_to_logical
                if node not in claimed
            ]
        if not vacant_nodes:
            vacant_nodes = list(physical_to_logical.keys())

        if partner is not None:
            # Place closest to partner's effective position.
            partner_phys = already_assigned.get(
                partner,
                logical_to_physical.get(partner),
            )
            if partner_phys is not None:
                return min(
                    vacant_nodes,
                    key=lambda n: self.coupling_map.distance(n, partner_phys),
                )

        # No partner — prefer original position if vacant.
        orig = logical_to_physical_init.get(qubit)
        if orig is not None and orig in vacant_nodes:
            return orig

        return vacant_nodes[0]

    # ------------------------------------------------------------------
    # SWAP / MOVE routing along a path (same as greedy/lookahead)
    # ------------------------------------------------------------------

    def _route_and_update_state(
        self,
        qi: Qubit,
        qj: Qubit,
        path: list[int],
        dag: DAGCircuit,
        logical_to_physical: dict[Qubit, int],
        physical_to_logical: dict[int, Optional[Qubit]],
    ) -> None:
        """Inject SWAPs (or cheaper MOVEs) along *path* to bring ``qi``
        and ``qj`` adjacent on the SC coupling map.

        A SWAP that involves a vacant physical node (``physical_to_logical
        [node] is None``) is treated as a logical MOVE: the maps are
        updated but no ``SwapGate`` is injected into the DAG.
        """
        for step_idx in range(len(path) - 2):
            node_a = path[step_idx]
            node_b = path[step_idx + 1]

            qubit_on_a = physical_to_logical[node_a]
            qubit_on_b = physical_to_logical[node_b]

            if qubit_on_a is None and qubit_on_b is None:
                continue

            if qubit_on_b is None:
                # MOVE: slide qubit into vacant node.
                logical_to_physical[qubit_on_a] = node_b
                physical_to_logical[node_b] = qubit_on_a
                physical_to_logical[node_a] = None
            elif qubit_on_a is None:
                # Mirror MOVE.
                logical_to_physical[qubit_on_b] = node_a
                physical_to_logical[node_a] = qubit_on_b
                physical_to_logical[node_b] = None
            else:
                # Full SWAP: both nodes occupied.
                dag.apply_operation_back(
                    SwapGate(), [dag.qubits[node_a], dag.qubits[node_b]], [],
                )
                logical_to_physical[qubit_on_a] = node_b
                logical_to_physical[qubit_on_b] = node_a
                physical_to_logical[node_a] = qubit_on_b
                physical_to_logical[node_b] = qubit_on_a
