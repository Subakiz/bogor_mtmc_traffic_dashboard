#!/usr/bin/env python3
"""
Tier 5 Adversarial Stress Tests: Graph Traversal, Invariants & Routing
BSW Kota Bogor Multi-Camera Tracking ("Follow That Car") Traffic Intelligence Dashboard

Verifies:
- 1,000+ random walk traversals without one-way invariant violations.
- SSA counter-clockwise circulation invariant.
- Suryakencana southbound circulation invariant.
- Graph diameter and average path lengths are physically realistic for Kota Bogor.
- Robust handling of edge cases (invalid camera IDs, missing directions, boundary numbers).
"""

import unittest
import random
import math
import numpy as np
from backend.topology_engine import (
    TopologyEngine,
    get_camera_info,
    get_all_cameras,
    get_downstream_candidates,
    is_valid_transition,
    calculate_arrival_window
)


class TestGraphTraversalAdversarial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = TopologyEngine()
        cls.node_ids = sorted(list(cls.engine.cameras.keys()))
        cls.ssa_forbidden_reverse = set(cls.engine.SSA_FORBIDDEN_REVERSE_EDGES)
        cls.surken_forbidden_reverse = set(cls.engine.SURYAKENCANA_FORBIDDEN_REVERSE_EDGES)
        cls.surken_additional_reverse = {
            (167, 75), (188, 75),
            (75, 86), (86, 96), (75, 96), (86, 100), (86, 89),
            (89, 91), (91, 101), (100, 83), (101, 83), (96, 83)
        }
        cls.all_surken_forbidden = cls.surken_forbidden_reverse | cls.surken_additional_reverse

    def test_random_walks_zero_invariant_violations(self):
        """Execute 1,000+ random walks and ensure zero invariant violations."""
        random.seed(1337)
        total_walks = 1500
        walk_length = 30
        
        ssa_violations = []
        surken_violations = []
        
        for walk_i in range(total_walks):
            curr = random.choice(self.node_ids)
            for step in range(walk_length):
                outgoing = self.engine.outgoing_edges.get(curr, [])
                self.assertTrue(len(outgoing) > 0, f"Dead-end node encountered: {curr}")
                edge = random.choice(outgoing)
                nxt = edge["target_id"]
                pair = (curr, nxt)
                
                if pair in self.ssa_forbidden_reverse:
                    ssa_violations.append((walk_i, step, pair))
                if pair in self.all_surken_forbidden:
                    surken_violations.append((walk_i, step, pair))
                    
                curr = nxt
                
        self.assertEqual(len(ssa_violations), 0, f"Encountered SSA violations: {ssa_violations}")
        self.assertEqual(len(surken_violations), 0, f"Encountered Surken violations: {surken_violations}")

    def test_bogor_graph_diameter_and_physical_realism(self):
        """Verify graph diameter and average path lengths are physically realistic for Kota Bogor."""
        N = len(self.node_ids)
        hop_matrix = np.full((N, N), np.inf)
        dist_matrix = np.full((N, N), np.inf)
        time_matrix = np.full((N, N), np.inf)

        for i, src in enumerate(self.node_ids):
            for j, tgt in enumerate(self.node_ids):
                if src == tgt:
                    hop_matrix[i, j] = 0
                    dist_matrix[i, j] = 0
                    time_matrix[i, j] = 0
                else:
                    path_info = self.engine.find_shortest_path(src, tgt)
                    if path_info is not None:
                        path, t_min, _ = path_info
                        hop_matrix[i, j] = len(path) - 1
                        time_matrix[i, j] = t_min
                        d = 0.0
                        for u, v in zip(path[:-1], path[1:]):
                            edges = self.engine.edge_by_pair.get((u, v), [])
                            if edges:
                                d += edges[0]["road_distance_m"]
                        dist_matrix[i, j] = d

        # Strongly connected: all pairs reachable
        self.assertTrue(np.all(~np.isinf(hop_matrix)), "Graph is not strongly connected!")

        finite_hops = hop_matrix[hop_matrix > 0]
        finite_dists = dist_matrix[dist_matrix > 0]
        finite_times = time_matrix[time_matrix > 0]

        hop_diameter = np.max(finite_hops)
        dist_diameter_m = np.max(finite_dists)
        avg_dist_m = np.mean(finite_dists)
        time_diameter_s = np.max(finite_times)

        # Bogor municipal bounds check
        self.assertGreaterEqual(hop_diameter, 15, "Hop diameter too small for 94 camera network")
        self.assertLessEqual(hop_diameter, 30, "Hop diameter excessively large")
        self.assertGreaterEqual(dist_diameter_m, 12000.0, "Road diameter < 12 km; unrealistic for Kota Bogor")
        self.assertLessEqual(dist_diameter_m, 22000.0, "Road diameter > 22 km; exceeds Kota Bogor municipal boundary")
        self.assertGreaterEqual(avg_dist_m, 3000.0, "Average road trip distance too small")
        self.assertLessEqual(avg_dist_m, 9000.0, "Average road trip distance too large")
        self.assertGreaterEqual(time_diameter_s, 900.0, "Travel time diameter < 15 min")
        self.assertLessEqual(time_diameter_s, 2400.0, "Travel time diameter > 40 min")

    def test_ssa_ring_cycle_intact_and_unidirectional(self):
        """Verify the counter-clockwise SSA ring loop is complete and has zero reverse edges."""
        ssa_cycle = [79, 87, 83, 81, 85, 133, 134, 92, 120, 122, 132, 79]
        edges_set = set((e["source_id"], e["target_id"]) for e in self.engine.edges)

        # Every step in counter-clockwise cycle must exist
        for u, v in zip(ssa_cycle[:-1], ssa_cycle[1:]):
            self.assertIn((u, v), edges_set, f"Missing forward SSA edge: {u} -> {v}")
            # The exact reverse must NOT exist (one-way enforcement)
            self.assertNotIn((v, u), edges_set, f"Illegal reverse SSA edge exists: {v} -> {u}")

    def test_suryakencana_strict_dag(self):
        """Verify that the Suryakencana corridor forms a strictly acyclic south-flowing DAG."""
        surken_nodes = set(self.engine.get_corridor_cameras("SURKEN_PASAR_BOGOR"))
        surken_edges = [
            (e["source_id"], e["target_id"]) for e in self.engine.edges
            if e["source_id"] in surken_nodes and e["target_id"] in surken_nodes
        ]
        
        in_degree = {cid: 0 for cid in surken_nodes}
        adj = {cid: [] for cid in surken_nodes}
        for u, v in surken_edges:
            adj[u].append(v)
            in_degree[v] += 1
            
        queue = [cid for cid, deg in in_degree.items() if deg == 0]
        visited = 0
        while queue:
            curr = queue.pop(0)
            visited += 1
            for nxt in adj[curr]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)
                    
        self.assertEqual(visited, len(surken_nodes), "Suryakencana contains a cycle or circular feedback loop!")

    def test_routing_edge_cases_and_invalid_inputs(self):
        """Verify edge case handling on invalid IDs, directions, and boundary values."""
        # Invalid source on downstream candidates raises KeyError
        with self.assertRaises(KeyError):
            get_downstream_candidates(-1, "NORTH")
        with self.assertRaises(KeyError):
            get_downstream_candidates(99999, "NORTH")

        # Wildcard queries return all candidates
        all_cands = get_downstream_candidates(79, "*")
        self.assertEqual(len(all_cands), 7)   # 80, 87, 103, 199 + Lawang 9 (204, 205) and Otista 207

        # Unknown directions return empty list
        self.assertEqual(get_downstream_candidates(79, "NONEXISTENT_DIR"), [])

        # Non-finite and negative elapsed times return (False, 0.0) safely
        self.assertEqual(is_valid_transition(79, 87, -50.0), (False, 0.0))
        self.assertEqual(is_valid_transition(79, 87, float("inf")), (False, 0.0))
        self.assertEqual(is_valid_transition(79, 87, float("-inf")), (False, 0.0))
        self.assertEqual(is_valid_transition(79, 87, float("nan")), (False, 0.0))

        # Reverse SSA query returns (False, 0.0) via module-level function and direct_only engine method
        self.assertEqual(is_valid_transition(87, 79, 50.0), (False, 0.0))
        self.assertEqual(self.engine.is_valid_transition(87, 79, 50.0, direct_only=True), (False, 0.0))


if __name__ == "__main__":
    unittest.main()
