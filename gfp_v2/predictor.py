"""Measured avGFP predictors. No archive labels enter an estimator except via fit()."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Callable, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from threadpoolctl import threadpool_limits

from gfp_data import FLUORESCENT_THRESHOLD, GFP_WT, Variant, fit_predictor, make_variant


BASELINES = ("ridge", "rf")
ENHANCED = ("physchem_ridge", "pair_ridge", "hist_boost", "hurdle", "blend")
MODELS = (*BASELINES, *ENHANCED)

# Kyte & Doolittle, J Mol Biol 157 (1982), 105–132, doi:10.1016/0022-2836(82)90515-0.
HYDROPATHY = dict(zip("ACDEFGHIKLMNPQRSTVWY", (
    1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
    1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3,
), strict=True))
CHEMISTRY = {aa: group for group, amino_acids in (
    ("aliphatic", "AVILM"), ("aromatic", "FWY"), ("polar", "STNQ"),
    ("acidic", "DE"), ("basic", "HKR"), ("special", "CGP"),
) for aa in amino_acids}
CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1}  # Neutral histidine approximation, not a pKa model.


@lru_cache(maxsize=4096)
def _features(variant: Variant, pairwise: bool) -> dict[str, float]:
    if make_variant(variant.mutant) != variant:
        raise ValueError("Predictors require canonical variants of the pinned GFP reference")
    count = len(variant.edits)
    features = {"edit_count": float(count), f"degree:{count}": 1.0}
    hydropathy, charge = [], []
    changes = []
    for token in variant.edits:
        source, position, target = token[0], int(token[1:-1]), token[-1]
        chemical_change = f"{CHEMISTRY[source]}>{CHEMISTRY[target]}"
        changes.append((position, chemical_change))
        delta_hydro = (HYDROPATHY[target] - HYDROPATHY[source]) / 9.0
        delta_charge = (CHARGE.get(target, 0) - CHARGE.get(source, 0)) / 2.0
        hydropathy.append(delta_hydro)
        charge.append(delta_charge)
        features.update({f"position:{position}": 1.0, f"allele:{token}": 1.0})
        for key, amount in (
            (f"substitution:{source}>{target}", 0.5),
            (f"chemistry:{chemical_change}", 0.5),
            (f"position_chemistry:{position}:{CHEMISTRY[target]}", 0.5),
            (f"target:{target}", 0.5),
            (f"segment:{(position - 1) // 24}", 0.5),
            (f"site_hydropathy:{position}", delta_hydro),
            (f"site_charge:{position}", delta_charge),
        ):
            if amount:
                features[key] = features.get(key, 0.0) + amount
    if count:
        features.update({
            "hydropathy_delta_total": sum(hydropathy),
            "hydropathy_delta_absolute": sum(map(abs, hydropathy)),
            "hydropathy_delta_largest": max(map(abs, hydropathy)),
            "charge_delta_total": sum(charge),
            "charge_delta_absolute": sum(map(abs, charge)),
        })
    if pairwise:
        for first, second in combinations(variant.edits, 2):
            features[f"pair_alleles:{first}|{second}"] = 0.5
            features[f"pair_positions:{first[1:-1]}|{second[1:-1]}"] = 0.5
        for (left, left_change), (right, right_change) in combinations(changes, 2):
            distance = right - left  # Sequence distance, not a structural contact.
            distance_bin = next(i for i, edge in enumerate((5, 20, 60, len(GFP_WT))) if distance <= edge)
            key = f"pair_chemistry:{distance_bin}:{'|'.join(sorted((left_change, right_change)))}"
            features[key] = features.get(key, 0.0) + 0.5
    return {key: value for key, value in features.items() if value}


def _ridge():
    return Ridge(alpha=4.0, solver="lsqr")


def _hurdle(matrix, targets: np.ndarray) -> Callable:
    is_bright = targets > FLUORESCENT_THRESHOLD
    if np.all(is_bright) or not np.any(is_bright):
        return _ridge().fit(matrix, targets).predict
    classifier = LogisticRegression(C=1.0, solver="liblinear", max_iter=500, random_state=0)
    classifier.fit(matrix, is_bright)
    bright = _ridge().fit(matrix[is_bright], targets[is_bright])
    dark = _ridge().fit(matrix[~is_bright], targets[~is_bright])

    def predict(query):
        chance = classifier.predict_proba(query)[:, 1]
        return chance * bright.predict(query) + (1 - chance) * dark.predict(query)

    return predict


@dataclass
class Predictor:
    name: str
    _predict: Callable
    _risk: Callable

    def predict(self, variants: Sequence[Variant]) -> np.ndarray:
        return np.asarray(self._predict(variants), dtype=float) if len(variants) else np.array([], dtype=float)

    def uncertainty(self, variants: Sequence[Variant]) -> np.ndarray:
        return np.asarray(self._risk(variants), dtype=float) if len(variants) else np.array([], dtype=float)

    def estimate(self, variants: Sequence[Variant]) -> tuple[np.ndarray, np.ndarray]:
        return self.predict(variants), self.uncertainty(variants)


def _support_risk(variants: Sequence[Variant], targets: np.ndarray) -> Callable:
    allele_count = Counter(edit for record in variants for edit in record.edits)
    site_count = Counter(edit[1:-1] for record in variants for edit in record.edits)
    pair_count = Counter(pair for record in variants for pair in combinations(record.edits, 2))
    scale = max(float(np.std(targets)), 1e-6)

    def risk(query):
        values = []
        for variant in query:
            alleles = variant.edits
            if not alleles:
                values.append(0.0)
                continue
            sites = [edit[1:-1] for edit in alleles]
            components = [1 / (1 + allele_count[edit]) for edit in alleles]
            components.extend(1 / (1 + site_count[site]) for site in sites)
            components.extend(1 / (1 + pair_count[pair]) for pair in combinations(alleles, 2))
            values.append(scale * np.sqrt(np.mean(components)))
        return values

    return risk


def fit(variants: Sequence[Variant], observed: Sequence[float], seed: int, model: str) -> Predictor:
    """Fit one fixed recipe with no access to sequence labels except observed."""
    if model not in MODELS:
        raise ValueError(f"Unknown GFP V2 model: {model!r}")
    targets = np.asarray(observed, dtype=float)
    if (
        targets.ndim != 1 or len(targets) < 2 or len(variants) != len(targets)
        or not np.isfinite(targets).all() or len({variant.id for variant in variants}) != len(variants)
    ):
        raise ValueError("Training needs at least two distinct canonical variants and finite observations")
    if model in BASELINES:
        legacy = fit_predictor(variants, targets, seed, model)
        risk = legacy.uncertainty if model == "rf" else _support_risk(variants, targets)
        return Predictor(model, legacy.predict, risk)

    with threadpool_limits(limits=1):
        pairwise = model != "physchem_ridge"
        encoder = DictVectorizer(sparse=True, sort=True)
        matrix = encoder.fit_transform([_features(variant, pairwise) for variant in variants])
        if model == "hist_boost":
            # Limit dimensionality by training support only. Equal prevalence preserves feature-name order.
            supported = np.argsort(-matrix.getnnz(axis=0), kind="stable")[:192]
            estimator = HistGradientBoostingRegressor(
                max_iter=120, max_leaf_nodes=15, learning_rate=0.06,
                min_samples_leaf=10, l2_regularization=1.0, random_state=seed, early_stopping=False,
            ).fit(matrix[:, supported].toarray(), targets)
            transform = lambda query: estimator.predict(query[:, supported].toarray())
        elif model == "hurdle":
            transform = _hurdle(matrix, targets)
        else:
            ridge = _ridge().fit(matrix, targets)
            transform = ridge.predict
            if model == "blend":
                hurdle = _hurdle(matrix, targets)
                transform = lambda query: 0.5 * ridge.predict(query) + 0.5 * hurdle(query)

    def predict(query):
        matrix_query = encoder.transform([_features(variant, pairwise) for variant in query])
        with threadpool_limits(limits=1):
            return transform(matrix_query)

    return Predictor(model, predict, _support_risk(variants, targets))
