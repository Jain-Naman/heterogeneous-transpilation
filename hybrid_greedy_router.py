"""Hybrid Superconducting / Neutral-Atom Greedy Router.

This module implements a custom Qiskit ``TransformationPass`` that greedily
partitions and routes a quantum circuit's DAG between a Superconducting (SC)
architecture and a Neutral Atom (NA) architecture, minimising the total
execution and transduction cost.

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

from qiskit.circuit import Instruction, Qubit
from qiskit.circuit.library import SwapGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.basepasses import TransformationPass


# ---------------------------------------------------------------------------
# Custom instruction representing qubit transduction between modalities.
# ---------------------------------------------------------------------------

class TransductGate(Instruction):
    """A 1-qubit, 0-clbit marker instruction representing the physical
    movement of a qubit between the SC and NA modalities.

    This gate carries no unitary meaning; it serves as an annotation in the
    output DAG so that downstream compilation stages can account for the
    transduction cost.
    """

    def __init__(self) -> None:
        super().__init__("transduct", 1, 0, [])


# Singleton instance shared across all invocations.
TRANSDUCT = TransductGate()


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

class HybridGreedyRouter(TransformationPass):
    """Greedily assign every 2-qubit gate to the cheapest modality.

    For each 2-qubit gate encountered in topological order the pass
    computes:

    * **Cost_NA** = ``c_na`` + ``c_trans`` × (number of operand qubits
      currently on SC).
    * **Cost_SC** = ``c_sc`` + ``c_swap`` × (``d`` − 1) + ``c_trans``
      × (number of operand qubits currently on NA), where *d* is the
      shortest-path distance between the two physical positions on the SC
      coupling map.

    The gate is then executed on whichever modality is cheaper, and the
    appropriate TRANSDUCT / SWAP gates are injected into the output DAG.

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

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Transform *dag* by greedily routing each 2-qubit gate.

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

        # --- state trackers ---
        modality_state: dict[Qubit, str] = {
            q: "SC" for q in dag.qubits
        }
        logical_to_physical: dict[Qubit, int] = {}
        physical_to_logical: dict[int, Optional[Qubit]] = {}

        # Initialise mappings from the provided layout.
        for physical_idx in range(self.coupling_map.size()):
            qubit = self.initial_layout[physical_idx]
            logical_to_physical[qubit] = physical_idx
            physical_to_logical[physical_idx] = qubit

        # --- iterate in topological order ---
        for node in dag.topological_op_nodes():
            num_qubits = node.op.num_qubits

            # Step 1 – single-qubit gates: pass through unchanged.
            if num_qubits == 1:
                new_dag.apply_operation_back(
                    node.op, [new_dag.qubits[logical_to_physical[q]] for q in node.qargs], node.cargs,
                )
                continue

            # Step 2 – two-qubit gates: evaluate costs.
            assert num_qubits == 2, (
                f"HybridGreedyRouter only handles 1- and 2-qubit gates, "
                f"got {num_qubits}-qubit gate '{node.op.name}'."
            )

            qi, qj = node.qargs[0], node.qargs[1]

            cost_na, cost_sc, sc_meta = self._evaluate_costs(
                qi, qj, modality_state,
                logical_to_physical, physical_to_logical,
            )

            # Step 3 – execute on the cheaper modality.
            if cost_na < cost_sc:
                # --- Branch A: execute on NA ---
                self._execute_on_na(
                    qi, qj, new_dag,
                    modality_state, logical_to_physical,
                    physical_to_logical,
                )
            else:
                # --- Branch B: execute on SC ---
                self._execute_on_sc(
                    qi, qj, new_dag, sc_meta,
                    modality_state, logical_to_physical,
                    physical_to_logical,
                )

            # Apply the original gate.
            new_dag.apply_operation_back(
                node.op, [new_dag.qubits[logical_to_physical[q]] for q in node.qargs], node.cargs,
            )

        return new_dag

    # ------------------------------------------------------------------
    # Cost evaluation
    # ------------------------------------------------------------------

    def _evaluate_costs(
        self,
        qi: Qubit,
        qj: Qubit,
        modality_state: dict[Qubit, str],
        logical_to_physical: dict[Qubit, int],
        physical_to_logical: dict[int, Optional[Qubit]],
    ) -> tuple[float, float, dict]:
        """Compute *Cost_NA* and *Cost_SC* for a 2-qubit gate on ``qi, qj``.

        Returns
        -------
        cost_na : float
            Total cost if the gate is executed on the NA chip.
        cost_sc : float
            Total cost if the gate is executed on the SC chip.
        sc_meta : dict
            Metadata needed by ``_execute_on_sc`` (shortest path,
            temporary physical assignments for NA-resident qubits, …).
        """
        # --- Cost_NA ---
        cost_na = self.c_na
        for q in (qi, qj):
            if modality_state[q] == "SC":
                cost_na += self.c_trans

        # --- Cost_SC (more involved) ---
        trans_penalty_sc = 0.0

        # Temporary physical assignments for qubits currently on NA.
        temp_assignments: dict[Qubit, int] = {}

        for q in (qi, qj):
            if modality_state[q] == "NA":
                trans_penalty_sc += self.c_trans
                # Temporarily assign to the closest vacant SC node (for
                # distance calculation).  The *other* qubit's physical
                # position is the reference point.
                best_node = self._find_closest_vacant_node(
                    q, qi, qj,
                    logical_to_physical, physical_to_logical,
                    temp_assignments,
                )
                temp_assignments[q] = best_node

        # Resolve effective physical positions.
        phys_i = temp_assignments.get(qi, logical_to_physical.get(qi))
        phys_j = temp_assignments.get(qj, logical_to_physical.get(qj))

        # Shortest-path distance on the coupling map.
        distance: int = self.coupling_map.distance(phys_i, phys_j)
        path: list[int] = list(
            self.coupling_map.shortest_undirected_path(phys_i, phys_j)
        )

        # Base + routing cost.
        routing_cost = self.c_swap * max(distance - 1, 0)
        cost_sc = self.c_sc + routing_cost + trans_penalty_sc

        sc_meta = {
            "distance": distance,
            "path": path,
            "temp_assignments": temp_assignments,
        }

        return cost_na, cost_sc, sc_meta

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _execute_on_na(
        self,
        qi: Qubit,
        qj: Qubit,
        dag: DAGCircuit,
        modality_state: dict[Qubit, str],
        logical_to_physical: dict[Qubit, int],
        physical_to_logical: dict[int, Optional[Qubit]],
    ) -> None:
        """Prepare the state for executing a gate on the NA chip.

        1. Transduct any SC-resident operand qubits to NA.
        2. Update the state trackers (modality, physical maps).

        The caller is responsible for appending the actual gate.
        """
        for q in (qi, qj):
            if modality_state[q] == "SC":
                phys = logical_to_physical[q]
                # Inject a TRANSDUCT marker.
                dag.apply_operation_back(TRANSDUCT, [dag.qubits[phys]], [])
                # Vacate the SC node.
                physical_to_logical[phys] = None
                modality_state[q] = "NA"

    def _execute_on_sc(
        self,
        qi: Qubit,
        qj: Qubit,
        dag: DAGCircuit,
        sc_meta: dict,
        modality_state: dict[Qubit, str],
        logical_to_physical: dict[Qubit, int],
        physical_to_logical: dict[int, Optional[Qubit]],
    ) -> None:
        """Prepare the state for executing a gate on the SC chip.

        1. Transduct any NA-resident operand qubits back to SC, placing
           them on optimal vacant physical nodes.
        2. SWAP-route the qubits so that they become adjacent on the
           coupling map.
        3. Update the state trackers.

        The caller is responsible for appending the actual gate.
        """
        temp_assignments: dict[Qubit, int] = sc_meta["temp_assignments"]

        # 1. Transduct NA → SC.
        for q in (qi, qj):
            if modality_state[q] == "NA":
                target_node = temp_assignments[q]
                dag.apply_operation_back(TRANSDUCT, [q], [])
                modality_state[q] = "SC"
                logical_to_physical[q] = target_node
                physical_to_logical[target_node] = q

        # 2. Route (SWAPs / MOVEs) along the shortest path.
        path: list[int] = sc_meta["path"]
        distance: int = sc_meta["distance"]

        if distance > 1:
            self._route_and_update_state(
                qi, qj, path, dag,
                logical_to_physical, physical_to_logical,
            )

    # ------------------------------------------------------------------
    # SWAP / MOVE routing along a path
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

        After this method returns, the two qubits will reside on the last
        two nodes of *path* (i.e. ``path[-2]`` and ``path[-1]``).

        A SWAP that involves a vacant physical node (``physical_to_logical
        [node] is None``) is treated as a logical MOVE: the maps are
        updated but no ``SwapGate`` is injected into the DAG (the qubit
        simply "slides" into the vacancy).

        Parameters
        ----------
        qi, qj : Qubit
            The two logical qubits that need to interact.
        path : list[int]
            The shortest path on the coupling map from the physical node
            of ``qi`` to the physical node of ``qj``.
        dag : DAGCircuit
            The output DAG to mutate.
        logical_to_physical, physical_to_logical : dicts
            The live mapping state.
        """
        # We SWAP qi along the path towards qj, one hop at a time.
        # After all SWAPs, qi will be on path[-2] (adjacent to qj on
        # path[-1]).
        for step_idx in range(len(path) - 2):
            node_a = path[step_idx]
            node_b = path[step_idx + 1]

            qubit_on_a = physical_to_logical[node_a]
            qubit_on_b = physical_to_logical[node_b]

            if qubit_on_a is None and qubit_on_b is None:
                # Both vacant – nothing to do (shouldn't normally happen
                # on a valid path, but handle defensively).
                continue

            if qubit_on_b is None:
                # MOVE optimisation: the target node is vacant, so we can
                # simply slide the qubit without a heavy SWAP gate.
                logical_to_physical[qubit_on_a] = node_b
                physical_to_logical[node_b] = qubit_on_a
                physical_to_logical[node_a] = None
            elif qubit_on_a is None:
                # Mirror MOVE: slide node_b's qubit into node_a.
                logical_to_physical[qubit_on_b] = node_a
                physical_to_logical[node_a] = qubit_on_b
                physical_to_logical[node_b] = None
            else:
                # Full SWAP: both nodes are occupied.
                dag.apply_operation_back(
                    SwapGate(), [dag.qubits[node_a], dag.qubits[node_b]], [],
                )
                # Update mappings for both qubits.
                logical_to_physical[qubit_on_a] = node_b
                logical_to_physical[qubit_on_b] = node_a
                physical_to_logical[node_a] = qubit_on_b
                physical_to_logical[node_b] = qubit_on_a

    # ------------------------------------------------------------------
    # Utility: find closest vacant node
    # ------------------------------------------------------------------

    def _find_closest_vacant_node(
        self,
        qubit: Qubit,
        qi: Qubit,
        qj: Qubit,
        logical_to_physical: dict[Qubit, int],
        physical_to_logical: dict[int, Optional[Qubit]],
        already_assigned: dict[Qubit, int],
    ) -> int:
        """Return the vacant SC node closest to the *other* operand qubit.

        When computing Cost_SC for a gate on ``(qi, qj)`` and one of them
        is on NA, we need to hypothetically place it somewhere on the SC
        chip.  The heuristic is to pick the closest vacant node to the
        partner's current (or already-assigned) physical position.

        Parameters
        ----------
        qubit : Qubit
            The qubit to place (currently on NA).
        qi, qj : Qubit
            The two operand qubits of the gate being evaluated.
        logical_to_physical : dict
            Current logical → physical mapping.
        physical_to_logical : dict
            Current physical → logical mapping.
        already_assigned : dict
            Temporary assignments already made during this cost evaluation
            (to avoid assigning two NA qubits to the same vacant node).

        Returns
        -------
        int
            The physical node index chosen for *qubit*.
        """
        # The partner is the other operand.
        partner = qj if qubit is qi else qi

        # Determine partner's effective physical position.
        if partner in already_assigned:
            partner_phys = already_assigned[partner]
        else:
            partner_phys = logical_to_physical.get(partner)

        # Nodes already claimed in this evaluation round.
        claimed: set[int] = set(already_assigned.values())

        # Gather all vacant physical nodes.
        vacant_nodes = [
            node for node, occupant in physical_to_logical.items()
            if occupant is None and node not in claimed
        ]

        if not vacant_nodes:
            # Fallback: if no vacant nodes exist (unlikely given balanced
            # qubit counts), pick the node closest to partner that is not
            # the partner itself.
            vacant_nodes = [
                node for node in physical_to_logical
                if node != partner_phys and node not in claimed
            ]

        if not vacant_nodes:
            # Last resort – use any node.
            vacant_nodes = list(physical_to_logical.keys())

        # Pick the vacant node with the smallest coupling-map distance to
        # the partner.
        best_node = min(
            vacant_nodes,
            key=lambda n: self.coupling_map.distance(n, partner_phys),
        )
        return best_node
