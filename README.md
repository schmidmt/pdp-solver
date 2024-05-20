# Heuristic Solver for the Pickup and Drop-off Problem

This PDP solver uses the Clarke and Wright Algorithm to find an approximately optimal solution to a given PDP.

## Running the solver

1. Create a virtual environment and activate it.
```bash
python -m venv .venv
source .venv/bin/activate
```
2. Install the dependencies.
```bash
python -m pip install -r requirements.txt
```

3. Run the solver on a problem file.
```bash
python solver.py 'Training Problems'/problem1.txt
```

## Assumptions
1. A single depot at location (0, 0).
2. An unbound number of vehicles.
3. A cost of 500 per vehicle and 1 cost per unit distance.
4. Distances are determined from Euclidean metric.

## References
* Clarke, Geoff, and John W. Wright. "Scheduling of vehicles from a central depot to a number of delivery points." Operations research 12.4 (1964): 568-581.
* Fikejz, Jan, Markéta Brázdová, and Ľudmila Jánošíková. "Modification of the Clarke and Wright Algorithm with a Dynamic Savings Matrix." Journal of Advanced Transportation 2024 (2024).
