"""Hybrid Superconducting / Neutral-Atom Look-ahead Router.

This module implements a custom Qiskit ``TransformationPass`` that partitions
and routes a quantum circuit's DAG between a Superconducting (SC) architecture
and a Neutral-Atom (NA) architecture by inspecting a *window* of future 2-qubit
gates in the DAG before making each routing decision.

Compared to the greedy approach (``HybridGreedyRouter``) which evaluates each
gate in isolation, this pass anticipates future gate dependencies so that it
can avoid "thrashing" — repeatedly paying the transduction penalty to shuttle
qubits back and forth between modalities.

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

from qiskit.circuit import Qubit
from qiskit.circuit.library import SwapGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.basepasses import TransformationPass

from hybrid_greedy_router import TRANSDUCT


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

class HybridLookaheadRouter(TransformationPass):
    """Assign every 2-qubit gate to the cheapest modality using a windowed
    look-ahead heuristic.

    For each 2-qubit gate encountered in topological order the pass extracts
    a *window* of upcoming 2-qubit gates from the DAG and scores the total
    estimated cost of executing the current gate on NA vs. SC.  The gate is
    then executed on whichever modality yields the lower windowed cost.

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
    window_depth : int
        Number of future 2-qubit gates to inspect per logical qubit when
        making a routing decision.
    decay : float
        Discount factor γ ∈ (0, 1] applied to future gate costs.  Costs at
        depth *d* in the window are multiplied by γ^d so that immediate
        routing needs outweigh distant future dependencies.
    """

    def __init__(
        self,
        coupling_map: CouplingMap,
        initial_layout: Layout,
        c_sc: float = 1.0,
        c_na: float = 5.0,
        c_swap: float = 3.0,
        c_trans: float = 10.0,
        window_depth: int = 5,
        decay: float = 0.9,
    ) -> None:
        super().__init__()
        self.coupling_map = coupling_map
        self.initial_layout = initial_layout
        self.c_sc = c_sc
        self.c_na = c_na
        self.c_swap = c_swap
        self.c_trans = c_trans
        self.window_depth = window_depth
        self.decay = decay

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Transform *dag* by routing each 2-qubit gate using a look-ahead
        window heuristic.

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
                    node.op, node.qargs, node.cargs,
                )
                continue

            # Step 2 – two-qubit gates: evaluate windowed costs.
            assert num_qubits == 2, (
                f"HybridLookaheadRouter only handles 1- and 2-qubit gates, "
                f"got {num_qubits}-qubit gate '{node.op.name}'."
            )

            qi, qj = node.qargs[0], node.qargs[1]

            na_score, sc_score, sc_meta = self._score_window(
                node, dag, modality_state,
                logical_to_physical, physical_to_logical,
            )

            # Step 3 – execute on the cheaper modality.
            if na_score < sc_score:
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
                node.op, node.qargs, node.cargs,
            )

        return new_dag

    # ------------------------------------------------------------------
    # Look-ahead window extraction
    # ------------------------------------------------------------------

    def _extract_window(
        self,
        node: DAGOpNode,
        dag: DAGCircuit,
    ) -> list[tuple[DAGOpNode, int]]:
        """Extract future 2-qubit gates from the DAG reachable from *node*.

        Traverses DAG successors of both operand qubits up to
        ``self.window_depth`` 2-qubit gates per qubit, collecting future
        gates along with their depth (number of 2-qubit hops from *node*).

        Returns a deduplicated list of ``(gate_node, depth)`` pairs.
        """
        qi, qj = node.qargs[0], node.qargs[1]
        window: dict[int, tuple[DAGOpNode, int]] = {}  # node_id → (node, depth)

        for start_qubit in (qi, qj):
            self._traverse_successors(
                node, start_qubit, dag, window, depth=0, max_depth=self.window_depth,
            )

        # Return sorted by depth for determinism.
        return sorted(window.values(), key=lambda t: t[1])

    def _traverse_successors(
        self,
        current_node: DAGOpNode,
        qubit: Qubit,
        dag: DAGCircuit,
        window: dict[int, tuple[DAGOpNode, int]],
        depth: int,
        max_depth: int,
    ) -> None:
        """Recursively traverse DAG successors along *qubit*'s wire.

        Collects 2-qubit gate nodes into *window* (keyed by node id to
        avoid double-counting), incrementing *depth* for each 2-qubit gate
        encountered, and stopping when *max_depth* 2-qubit gates have been
        found along this wire.
        """
        if depth >= max_depth:
            return

        for successor in dag.successors(current_node):
            if not isinstance(successor, DAGOpNode):
                continue

            # Only follow successors that act on the qubit we're tracing.
            if qubit not in successor.qargs:
                continue

            if successor.op.num_qubits == 2:
                node_id = id(successor)
                # Record at the shallowest depth if seen from multiple paths.
                if node_id not in window or window[node_id][1] > depth + 1:
                    window[node_id] = (successor, depth + 1)
                # Continue searching along *both* qubits of this gate.
                for q in successor.qargs:
                    self._traverse_successors(
                        successor, q, dag, window, depth + 1, max_depth,
                    )
            elif successor.op.num_qubits == 1:
                # 1-qubit gates don't increment depth; pass through.
                self._traverse_successors(
                    successor, qubit, dag, window, depth, max_depth,
                )

    # ------------------------------------------------------------------
    # Windowed scoring
    # ------------------------------------------------------------------

    def _score_window(
        self,
        node: DAGOpNode,
        dag: DAGCircuit,
        modality_state: dict[Qubit, str],
        logical_to_physical: dict[Qubit, int],
        physical_to_logical: dict[int, Optional[Qubit]],
    ) -> tuple[float, float, dict]:
        """Score both NA and SC assignments for *node* using look-ahead.

        Returns
        -------
        na_score : float
            Total estimated cost (immediate + windowed future) for NA.
        sc_score : float
            Total estimated cost (immediate + windowed future) for SC.
        sc_meta : dict
            Metadata needed by ``_execute_on_sc`` (path, temp assignments).
        """
        qi, qj = node.qargs[0], node.qargs[1]

        # --- Extract the look-ahead window ---
        future_gates = self._extract_window(node, dag)

        # === Score NA assignment (x_{g0} = 1) ===
        na_score = self._score_na(
            qi, qj, modality_state, logical_to_physical,
            future_gates,
        )

        # === Score SC assignment (x_{g0} = 0) ===
        sc_score, sc_meta = self._score_sc(
            qi, qj, modality_state, logical_to_physical,
            physical_to_logical, future_gates,
        )

        return na_score, sc_score, sc_meta

    def _score_na(
        self,
        qi: Qubit,
        qj: Qubit,
        modality_state: dict[Qubit, str],
        logical_to_physical: dict[Qubit, int],
        future_gates: list[tuple[DAGOpNode, int]],
    ) -> float:
        """Compute the NA score: immediate cost + heuristic future cost.

        Assumes qi and qj will be in NA after executing the current gate.
        For future gates, if the other qubit is currently on SC, we estimate
        the cost of bringing it to NA as c_na + c_trans.
        """
        # --- Immediate cost ---
        score = self.c_na
        for q in (qi, qj):
            if modality_state[q] == "SC":
                score += self.c_trans

        # --- Future heuristic ---
        # Under the NA hypothesis, qi and qj are assumed to be in NA.
        assumed_na = {qi, qj}

        for future_node, depth in future_gates:
            discount = self.decay ** depth
            qm, qn = future_node.qargs[0], future_node.qargs[1]

            # Estimate cost of this future gate if we assume a NA-biased
            # world where qi and qj stay in NA.
            future_cost = self.c_na
            for q in (qm, qn):
                if q in assumed_na:
                    # Already assumed to be in NA, no transduction.
                    pass
                elif modality_state[q] == "SC":
                    # Would need to transduct from SC to NA.
                    future_cost += self.c_trans

            score += discount * future_cost

        return score

    def _score_sc(
        self,
        qi: Qubit,
        qj: Qubit,
        modality_state: dict[Qubit, str],
        logical_to_physical: dict[Qubit, int],
        physical_to_logical: dict[int, Optional[Qubit]],
        future_gates: list[tuple[DAGOpNode, int]],
    ) -> tuple[float, dict]:
        """Compute the SC score: immediate cost + heuristic future cost.

        Returns the score and sc_meta dict for execution.
        """
        # --- Immediate cost ---
        score = self.c_sc
        trans_penalty = 0.0
        temp_assignments: dict[Qubit, int] = {}

        for q in (qi, qj):
            if modality_state[q] == "NA":
                trans_penalty += self.c_trans
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

        routing_cost = self.c_swap * max(distance - 1, 0)
        score += routing_cost + trans_penalty

        sc_meta = {
            "distance": distance,
            "path": path,
            "temp_assignments": temp_assignments,
        }

        # --- Future heuristic ---
        # Under the SC hypothesis, qi and qj are assumed to stay on SC
        # at their current/assigned physical positions.  We use the
        # *current* layout as a static lower-bound (no intermediate SWAPs
        # simulated).
        for future_node, depth in future_gates:
            discount = self.decay ** depth
            qm, qn = future_node.qargs[0], future_node.qargs[1]

            # Estimate physical distance for this future gate.
            phys_m = logical_to_physical.get(qm)
            phys_n = logical_to_physical.get(qn)

            # If either qubit is on NA (and not one of qi/qj which we
            # assume will be on SC), we can't compute a real distance;
            # estimate with transduction penalty.
            future_trans = 0.0
            can_compute_distance = True

            for q in (qm, qn):
                if modality_state[q] == "NA" and q not in (qi, qj):
                    # This qubit is on NA and would need transduction.
                    future_trans += self.c_trans
                    can_compute_distance = False

            if can_compute_distance and phys_m is not None and phys_n is not None:
                d_mn = self.coupling_map.distance(phys_m, phys_n)
                future_cost = self.c_sc + self.c_swap * max(d_mn - 1, 0) + future_trans
            else:
                # Fallback: assume some routing cost using base SC cost.
                future_cost = self.c_sc + future_trans

            score += discount * future_cost

        return score, sc_meta

    # ------------------------------------------------------------------
    # Execution helpers (identical to HybridGreedyRouter)
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
                # Inject a TRANSDUCT marker.
                dag.apply_operation_back(TRANSDUCT, [q], [])
                # Vacate the SC node.
                phys = logical_to_physical[q]
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
                    SwapGate(), [qubit_on_a, qubit_on_b], [],
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
