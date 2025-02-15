"""Run a solver for the Pickup and Delivery Problem (PDP)
for a single depot and unbounded vehicle count.
"""

from typing import Self, List, Optional
import argparse
from pathlib import Path
from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray

DEFAULT_MAX_DISTANCE: float = 12 * 60
DEFAULT_DEPLOYMENT_COST: float = 500
DEFAULT_COST_PER_DISANCE: float = 1.0


@dataclass
class Config:
    """Configuration options"""

    max_distance: float
    deployment_cost: float
    cost_per_distance: float


@dataclass
class Point:
    """A 2D cartesian point"""

    x: float
    y: float

    def dist(self, other: Self) -> float:
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    @classmethod
    def parse(cls, s: str) -> Self:
        s = s.lstrip("( ").rstrip(" )\n")
        x_s, y_s = s.split(",", maxsplit=2)
        return cls(float(x_s), float(y_s))


@dataclass
class PickupDropff:
    """Pickup and dropoff locations and ids for a customer order"""

    id: int
    pickup: Point
    dropoff: Point

    @classmethod
    def parse(cls, s: str) -> Self:
        id_s, pickup_s, dropoff_s = s.split(" ", maxsplit=3)
        id = int(id_s)
        pickup = Point.parse(pickup_s)
        dropoff = Point.parse(dropoff_s)

        return cls(id, pickup, dropoff)


class Route:
    """A route container which caches the distance"""

    sequence: List[int]
    distance: float
    limit: float

    def __init__(
        self,
        limit: float,
        sequence: Optional[List[int]] = None,
        distances: Optional[NDArray] = None,
    ):
        self.limit = limit
        self.distance = 0.0
        if sequence is None or len(sequence) == 0:
            self.sequence = list()
        else:
            self.sequence = sequence
            if distances is None:
                raise ValueError("distances cannot be none if sequence is given")
            self.distance += (
                distances[0, self.sequence[0]] + distances[self.sequence[-1], 0]
            )
            for a in self.sequence:
                self.distance += distances[a, a]
            for a, b in zip(self.sequence, self.sequence[1:]):
                self.distance += distances[a, b]

    def to_id_list(self, pds: List[PickupDropff]) -> List[int]:
        """Output a list of customer ids from the pickup and drop off list"""
        return [pds[i - 1].id for i in self.sequence]

    def join_if_feasible(self, other: Self, distances: NDArray) -> bool:
        """Join two routes if it's possible"""
        new_distance = (
            self.distance
            + other.distance
            - distances[self.last, 0]
            - distances[0, other.first]
            + distances[self.last, other.first]
        )
        if new_distance < self.limit:
            self.sequence.extend(other.sequence)
            self.distance = new_distance
            return True
        else:
            return False

    def append_if_feasible(self, pd: int, distances: NDArray) -> bool:
        """Append a point to this route if feasible"""
        return self.add_if_feasible(len(self.sequence), pd, distances)

    def add_if_feasible(self, index: int, pd: int, distances: NDArray) -> bool:
        """Add an pickup and deliver at the given index if feasible"""
        # Choose the original start of the intermediate route at index
        if index <= 0:
            left = 0  # Use the depot if we're at the start
        else:
            left = self.sequence[index - 1]

        # Choose the original stop of the intermediate route at index
        if index >= len(self.sequence):
            right = 0  # Use the depot as the return location
        else:
            right = self.sequence[index]

        new_distance = (
            self.distance
            + distances[left, pd]
            + distances[pd, pd]
            + distances[pd, right]
            - distances[left, right]
        )

        # Add if the new distance is feasible
        if new_distance <= self.limit:
            self.distance = new_distance
            self.sequence.insert(index, pd)
            return True
        else:
            return False

    @property
    def last(self) -> int:
        if self.sequence:
            return self.sequence[-1]
        else:
            return 0

    @property
    def first(self) -> int:
        if self.sequence:
            return self.sequence[0]
        else:
            return 0

    def __repr__(self) -> str:
        return f"{{Route dist={self.distance} s={self.sequence}}}"

    def __len__(self) -> int:
        return len(self.sequence)


def elementry_routes(distances, config: Config) -> List[Route]:
    """Create an initial feasible solution with one vehicle per stop."""
    routes = []

    for i in range(1, distances.shape[0]):
        route = Route(limit=config.max_distance)
        assert route.add_if_feasible(
            0, i, distances
        ), "Adding a single destination to a route should be feasible"
        routes.append(route)

    return routes


def save_matrix(distances: NDArray, implementation_cost: float) -> NDArray:
    """Compute the save matrix"""
    return (
        -distances
        + distances[0, :][np.newaxis, :]
        + distances[:, 0][:, np.newaxis]
        + implementation_cost
    )


def compute_distances(customers: List[PickupDropff]) -> NDArray:
    """Compute the distance matrix for the given customer pickup and drop offs.

    The diagonal components store the distance between the pickup and drop
    off locations.
    """

    n = len(customers) + 1
    distances = np.zeros((n, n))
    depot = Point(0, 0)

    for i, a in enumerate(customers, start=1):
        distances[0, i] = depot.dist(a.pickup)
        distances[i, 0] = depot.dist(a.dropoff)
        for j, b in enumerate(customers, start=1):
            if i == j:
                distances[i, i] = a.dropoff.dist(a.pickup)
            else:
                distances[i, j] = a.dropoff.dist(b.pickup)

    return distances


def clarke_wright_solver(distances: NDArray, config: Config) -> List[Route]:
    """Use the Clarke and Write Algorithm to find a near-optimal solution to the PDP.

    References:
        Clarke, Geoff, and John W. Wright. "Scheduling of vehicles from a
        central depot to a number of delivery points." Operations research 12.4
        (1964): 568-581.

        Fikejz, Jan, Markéta Brázdová, and Ľudmila Jánošíková. "Modification
        of the Clarke and Wright Algorithm with a Dynamic Savings Matrix."
        Journal of Advanced Transportation 2024 (2024).
    """
    n = distances.shape[0]
    save = save_matrix(distances, 500)
    ordered_saves = sorted(
        ((i, j, save[i, j]) for i in range(1, n) for j in range(1, n) if i != j),
        key=lambda x: x[2],
        reverse=True,
    )

    routes: List[Route] = elementry_routes(distances, config)
    # Create lookup tables to keep search times low
    ends: List[Optional[int]] = [None] * n
    starts: List[Optional[int]] = [None] * n

    for i, route in enumerate(routes):
        ends[route.last] = i
        starts[route.first] = i

    for i, j, save in ordered_saves:
        # Skip this save if the elements are inaccessible
        if ends[i] is None or starts[j] is None or ends[i] == starts[j]:
            continue

        route_to_extend = routes[ends[i]]
        route_to_consume = routes[starts[j]]

        if route_to_extend.join_if_feasible(route_to_consume, distances):
            # Since the join was feasible, we need to update the lookup tables
            ends[route_to_extend.last] = ends[i]

            # The route that was consumed can be removed.
            routes[starts[j]] = None

            # i and j are no longer accessible
            ends[i] = None
            starts[j] = None

    return [r for r in routes if r is not None]


def main():
    """Parse arguments, load the customer data, and run a solver."""

    parser = argparse.ArgumentParser(
        description="Find a fast, near-optimal solution to the 1-1 PDP problem."
    )

    parser.add_argument("PROBLEM_FILE", type=Path, help="Problem file location")
    parser.add_argument(
        "-m",
        "--max-distance",
        type=float,
        help="Maximum distance for any driver.",
        default=DEFAULT_MAX_DISTANCE,
    )
    parser.add_argument(
        "-c",
        "--cost-per-distance",
        type=float,
        help="Cost per unit distance for all routes.",
        default=DEFAULT_COST_PER_DISANCE,
    )
    parser.add_argument(
        "-d",
        "--deployment-cost",
        type=float,
        help="Cost per deployment",
        default=DEFAULT_COST_PER_DISANCE,
    )

    args = parser.parse_args()

    config = Config(
        max_distance=args.max_distance,
        deployment_cost=args.deployment_cost,
        cost_per_distance=args.cost_per_distance,
    )

    # Read the pickup and dropoffs from the external file
    customers: List[PickupDropff] = []
    with open(args.PROBLEM_FILE) as fin:
        line_iter = iter(fin.readlines())
        # Read, and drop, the header
        _ = next(line_iter)
        for line in line_iter:
            customers.append(PickupDropff.parse(line))

    # Compute the distances for each pair of customers
    distances = compute_distances(customers)

    # Use the CW solver
    routes = clarke_wright_solver(distances, config)

    # Output in the expected format
    for route in routes:
        print(route.to_id_list(customers))


if __name__ == "__main__":
    main()
