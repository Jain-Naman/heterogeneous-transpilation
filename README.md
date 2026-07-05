# Heterogeneous Transpilation for Hybrid Quantum Architectures

This project explores routing and transpilation strategies for hybrid quantum computing architectures that combine **Superconducting (SC)** and **Neutral-Atom (NA)** qubits.

In a heterogeneous system, different gate operations are better suited for different physical modalities. For instance:
- **Superconducting (SC) Architectures** typically feature fast gates but have restricted topological connectivity (e.g., a heavy-hex grid), requiring costly `SWAP` routing overhead.
- **Neutral-Atom (NA) Architectures** allow for dynamic reconfiguration, offering idealized all-to-all connectivity, but suffer from slower 2-qubit gate times.

The core challenge addressed by this repository is deciding, for each 2-qubit gate in a quantum circuit, which modality it should execute on. Moving a logical qubit between the SC and NA platforms incurs a flat **transduction penalty**. A successful routing strategy must balance the benefits of NA's connectivity and SC's speed against the cost of continuously transducing qubits back and forth ("thrashing").

## Code Description & Core Functionalities

The repository implements custom Qiskit `TransformationPass` passes that route a quantum circuit's DAG (Directed Acyclic Graph) between the SC and NA architectures. 

There are three primary routing strategies implemented:

1. **Hybrid Greedy Router** (`hybrid_greedy_router.py`)
   A fast, local heuristic. It evaluates each 2-qubit gate sequentially in topological order. For each gate, it greedily assigns it to the cheapest modality (SC or NA), factoring in the base gate costs, required `SWAP` overhead on SC, and immediate transduction penalties. It does *not* look ahead, making it susceptible to transduction thrashing.

2. **Hybrid Lookahead Router** (`hybrid_lookahead_router.py`)
   An improvement over the greedy approach. When making a routing decision for a gate, this pass extracts a *window* of upcoming 2-qubit gates from the DAG. By anticipating future dependencies with a configurable discount factor (decay), it avoids short-sighted modality switches that would incur heavy transduction penalties later.

3. **Hybrid Global ILP Router** (`hybrid_global_ilp.py`)
   The mathematically optimal baseline. It formulates the entire circuit routing problem as a Mixed-Integer Linear Program (MILP) using `scipy.optimize`. It considers the complete circuit simultaneously to find a globally optimal static assignment that minimizes the sum of gate execution costs and transduction penalties.

## What to Expect

When compiling a circuit using these passes, 1-qubit gates are left unchanged (assumed modality-agnostic), while 2-qubit gates are tagged with their target execution platform. 

The transpiler injects:
- **`TRANSDUCT` markers**: Custom 1-qubit instructions denoting the movement of a logical qubit between the SC and NA chips.
- **`SWAP` gates**: Standard Qiskit SWAP operations inserted when an SC gate requires interacting two physical qubits that are not adjacent in the SC topology.

## Tools & Utilities

- **`benchmark_comparison.py`**: A benchmarking script that compares the three hybrid routers against a standard SC-only baseline. It generates random quantum circuits and computes tangible metrics across varying cost-parameter regimes (e.g., Total 2-qubit operations, transduction counts, and % overhead reduction).
- **`visualize_circuits.py`**: A plotting utility that produces color-coded matplotlib circuit diagrams showing which modality each gate executes on. It vividly highlights SC gates (blue), NA gates (amber), SWAP routing (purple), and transduction boundaries (red).

## Cost Parameter Regimes

The benchmarking and visualization tools operate across different cost-parameter regimes that dictate the routers' behavior. The base costs are generally `c_sc = 1.0` and `c_swap = 3.0` for superconducting operations. The regimes vary the neutral-atom gate cost (`c_na`) and transduction penalty (`c_trans`):
- **Regime A (High transduction cost)**: `c_na = 5.0`, `c_trans = 10.0`
- **Regime B (Moderate transduction cost)**: `c_na = 3.0`, `c_trans = 5.0`
- **Regime C (Low transduction cost)**: `c_na = 2.0`, `c_trans = 2.0`
- **Regime D (Ultra-low transduction cost)**: `c_na = 1.5`, `c_trans = 1.0`

## How to Run

### Setup Dependencies

The project uses Qiskit, NumPy, SciPy (for the ILP solver), and Matplotlib (for visualization). 
You can install them via pip (or if using a virtual environment, ensure your dependencies are installed):

```bash
pip install qiskit numpy scipy matplotlib
```

### Benchmarking
Run the benchmark script to compare the transpilation overhead of the different routers across multiple randomly generated circuits.

```bash
python benchmark_comparison.py --num-trials 10 --seed 42
```

### Visualizing Circuits
Generate color-coded circuit diagrams to visually inspect the routing decisions made by the greedy and lookahead heuristics.

```bash
python visualize_circuits.py --output-dir ./visualizations --regime A
```
(Supported regimes correspond to different penalty parameters: `A`, `B`, `C`, or `D`).

### Running Unit Tests

The repository contains a comprehensive suite of unit tests verifying the routing logic for the Greedy, Lookahead, and Global ILP routers. You can run these tests using `pytest`:

```bash
pip install pytest
pytest tests/
```
