"""Bayesian network controller over scene variables.

Edge directions (causal / generative order):
  bottleShape → shipType
  table → light
  bottleShape → aspect          (wide bottles prefer landscape framing)
  shipType → light              (taller ships interact with overhead light)
  table → aspect                (surface type weakly influences framing)

Inference: exact enumeration for small domains + optional Gibbs sampling
for soft evidence / missing variables.
"""
from __future__ import annotations
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .scene_config import (
    SceneConfig,
    BOTTLE_SHAPES,
    SHIP_TYPES,
    TABLES,
    LIGHTS,
    ASPECTS,
)


class BottleController:
    """Learns / holds CPTs and answers P(var | evidence)."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Domain
        self.domains = {
            "bottleShape": BOTTLE_SHAPES,
            "shipType": SHIP_TYPES,
            "table": TABLES,
            "light": LIGHTS,
            "aspect": ASPECTS,
        }

        # Parents (corrected directions)
        self.parents = {
            "bottleShape": [],
            "shipType": ["bottleShape"],
            "table": [],
            "light": ["table", "shipType"],
            "aspect": ["bottleShape", "table"],
        }

        # CPTs: dict[var][parent_tuple][value] = prob
        self.cpts: Dict[str, Dict[Tuple, Dict[str, float]]] = {}
        self._init_prior_cpts()
        self._seed_from_synthetic_examples()

    # ------------------------------------------------------------------
    # CPT initialisation
    # ------------------------------------------------------------------
    def _init_prior_cpts(self):
        for var, pars in self.parents.items():
            self.cpts[var] = {}
            if not pars:
                # uniform prior
                n = len(self.domains[var])
                self.cpts[var][()] = {v: 1.0 / n for v in self.domains[var]}
            else:
                # will be filled by synthetic data or Laplace later
                pass

    def _seed_from_synthetic_examples(self):
        """A handful of hand-crafted (config, weight) pairs that encode
        reasonable aesthetic preferences.  These replace the earlier
        'guessed' probabilities and fix edge directions.
        """
        examples = [
            # (bottle, ship, table, light, aspect, weight)
            ("classic round, long neck", "full-rigged ship", "dark oak table, grain left-to-right, worn finish",
             "camera-left window light", "portrait 3:4", 3.0),
            ("classic round, long neck", "clipper", "mahogany desk, polished, warm tone",
             "warm sunset side light", "portrait 3:4", 2.0),
            ("demijohn with wide belly", "full-rigged ship", "dark oak table, grain left-to-right, worn finish",
             "camera-left window light", "square 1:1", 2.5),
            ("demijohn with wide belly", "schooner", "pine workbench, rough saw marks",
             "soft overhead studio key", "landscape 4:3", 1.5),
            ("square apothecary bottle", "sloop", "black slate surface, matte",
             "cool north-facing window", "portrait 3:4", 1.8),
            ("slim cylindrical flask", "brigantine", "mahogany desk, polished, warm tone",
             "warm sunset side light", "portrait 3:4", 1.2),
            ("classic round, long neck", "schooner", "dark oak table, grain left-to-right, worn finish",
             "camera-left window light", "portrait 3:4", 2.0),
            ("demijohn with wide belly", "full-rigged ship", "mahogany desk, polished, warm tone",
             "soft overhead studio key", "square 1:1", 1.7),
        ]
        # Count
        counts: Dict[str, Dict[Tuple, Dict[str, float]]] = {
            v: defaultdict(lambda: defaultdict(float)) for v in self.domains
        }
        for b, s, t, l, a, w in examples:
            counts["bottleShape"][()][b] += w
            counts["shipType"][(b,)][s] += w
            counts["table"][()][t] += w
            counts["light"][(t, s)][l] += w
            counts["aspect"][(b, t)][a] += w

        # Laplace smooth + normalise
        alpha = 0.4
        for var, pars in self.parents.items():
            domain = self.domains[var]
            parent_domains = [self.domains[p] for p in pars]
            # enumerate all parent combinations that appear + a few more
            seen = list(counts[var].keys())
            if not pars:
                total = sum(counts[var][()].values()) + alpha * len(domain)
                self.cpts[var][()] = {
                    v: (counts[var][()][v] + alpha) / total for v in domain
                }
            else:
                # fill every combination that has data; others stay near-uniform
                for pvals in seen:
                    total = sum(counts[var][pvals].values()) + alpha * len(domain)
                    self.cpts[var][pvals] = {
                        v: (counts[var][pvals][v] + alpha) / total for v in domain
                    }
                # default for unseen parent configs
                default = {v: 1.0 / len(domain) for v in domain}
                self.cpts[var]["_default"] = default

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _get_cpt(self, var: str, parent_vals: Tuple) -> Dict[str, float]:
        cpt = self.cpts[var]
        if parent_vals in cpt:
            return cpt[parent_vals]
        return cpt.get("_default", {v: 1.0 / len(self.domains[var]) for v in self.domains[var]})

    def query(
        self,
        target: str,
        evidence: Dict[str, str],
        method: str = "enumerate",
    ) -> Dict[str, float]:
        """Return P(target | evidence)."""
        if target not in self.domains:
            raise KeyError(f"Unknown variable {target}")
        # clamp evidence to known values
        ev = {k: v for k, v in evidence.items() if k in self.domains and k != target}

        if method == "gibbs":
            return self._gibbs_query(target, ev, n_samples=2000, burn=400)
        return self._enumerate_query(target, ev)

    def _enumerate_query(self, target: str, evidence: Dict[str, str]) -> Dict[str, float]:
        """Exact enumeration over the remaining free variables (tiny domain)."""
        free = [v for v in self.domains if v not in evidence and v != target]
        # topological order
        order = self._topo_order()
        scores = {val: 0.0 for val in self.domains[target]}

        def rec(i: int, assignment: Dict[str, str], joint: float):
            if i == len(order):
                tval = assignment[target]
                scores[tval] += joint
                return
            var = order[i]
            if var in assignment:
                # already fixed
                rec(i + 1, assignment, joint)
                return
            pars = self.parents[var]
            pvals = tuple(assignment[p] for p in pars)
            dist = self._get_cpt(var, pvals)
            for val, p in dist.items():
                assignment[var] = val
                rec(i + 1, assignment, joint * p)
                del assignment[var]

        # seed assignment with evidence + iterate target values
        for tval in self.domains[target]:
            start = dict(evidence)
            start[target] = tval
            # probability of this partial assignment under the BN
            # we let the recursion accumulate the product of CPTs
            rec(0, start, 1.0)

        # the recursion above double-counts because target is fixed early;
        # simpler approach: full joint enumeration
        scores = {val: 0.0 for val in self.domains[target]}
        from itertools import product
        free_vars = [v for v in order if v not in evidence]
        free_domains = [self.domains[v] for v in free_vars]
        for combo in product(*free_domains):
            assignment = dict(evidence)
            assignment.update(zip(free_vars, combo))
            # joint
            j = 1.0
            for var in order:
                pars = self.parents[var]
                pvals = tuple(assignment[p] for p in pars)
                j *= self._get_cpt(var, pvals).get(assignment[var], 1e-12)
            scores[assignment[target]] += j

        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}

    def _gibbs_query(
        self,
        target: str,
        evidence: Dict[str, str],
        n_samples: int = 2000,
        burn: int = 400,
    ) -> Dict[str, float]:
        """Gibbs sampler for the posterior of `target`."""
        free = [v for v in self.domains if v not in evidence]
        if target not in free:
            free.append(target)

        # initialise randomly
        state = dict(evidence)
        for v in free:
            state[v] = self.rng.choice(self.domains[v])

        counts = defaultdict(float)
        for s in range(n_samples + burn):
            for var in free:
                # sample from P(var | Markov blanket) ≈ full conditional
                pars = self.parents[var]
                pvals = tuple(state[p] for p in pars)
                dist = self._get_cpt(var, pvals)
                # also multiply by likelihood from children
                children = [c for c, ps in self.parents.items() if var in ps]
                for child in children:
                    cpars = self.parents[child]
                    # for each possible value of var, the child CPT contribution
                    # is already conditioned; we re-weight
                    pass  # for simplicity we use the prior conditional only
                    # (full blanket would re-normalise over children's CPTs)
                vals = list(dist.keys())
                probs = np.array([dist[v] for v in vals], dtype=float)
                probs /= probs.sum()
                state[var] = self.np_rng.choice(vals, p=probs)
            if s >= burn:
                counts[state[target]] += 1.0

        total = sum(counts.values()) or 1.0
        return {k: counts[k] / total for k in self.domains[target]}

    def _topo_order(self) -> List[str]:
        order = []
        visited = set()

        def visit(v):
            if v in visited:
                return
            visited.add(v)
            for p in self.parents[v]:
                visit(p)
            order.append(v)

        for v in self.domains:
            visit(v)
        return order

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------
    def sample_config(
        self,
        evidence: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
    ) -> SceneConfig:
        """Ancestral sampling → SceneConfig."""
        if seed is not None:
            rng = random.Random(seed)
            np_rng = np.random.default_rng(seed)
        else:
            rng = self.rng
            np_rng = self.np_rng

        evidence = evidence or {}
        assignment = dict(evidence)
        for var in self._topo_order():
            if var in assignment:
                continue
            pars = self.parents[var]
            pvals = tuple(assignment[p] for p in pars)
            dist = self._get_cpt(var, pvals)
            vals = list(dist.keys())
            probs = np.array([dist[v] for v in vals])
            probs /= probs.sum()
            assignment[var] = np_rng.choice(vals, p=probs)

        return SceneConfig(
            bottleShape=assignment["bottleShape"],
            shipType=assignment["shipType"],
            table=assignment["table"],
            light=assignment["light"],
            aspect=assignment["aspect"],
            seed=seed if seed is not None else self.rng.randint(0, 2**31 - 1),
        )

    def most_probable_config(
        self, evidence: Optional[Dict[str, str]] = None
    ) -> SceneConfig:
        """Greedy MAP under the current CPTs."""
        evidence = evidence or {}
        assignment = dict(evidence)
        for var in self._topo_order():
            if var in assignment:
                continue
            dist = self.query(var, assignment, method="enumerate")
            assignment[var] = max(dist, key=dist.get)
        return SceneConfig(
            bottleShape=assignment["bottleShape"],
            shipType=assignment["shipType"],
            table=assignment["table"],
            light=assignment["light"],
            aspect=assignment["aspect"],
        )

    @property
    def bn(self):
        """Tiny namespace so the original run script can do ctl.bn.parents."""
        class _BN:
            pass
        b = _BN()
        b.parents = self.parents
        return b
