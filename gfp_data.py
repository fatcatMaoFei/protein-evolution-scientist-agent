"""Pinned avGFP fluorescence data, scoped sequence handling, and predictors."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import os
import re
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Mapping, Sequence
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_squared_error


PROJECT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = PROJECT / ".cache" / "gfp"
ASSAY = "GFP_AEQVI_Sarkisyan_2016"
GFP_WT = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLS"
    "YGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDF"
    "KEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLL"
    "PDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)
WT_ID = "WT"
MAX_MUTATIONS = 4
FLUORESCENT_THRESHOLD = 2.5
SCORE_SCALE = "Sarkisyan log10 fluorescence (ProteinGym DMS_score)"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SUBSTITUTION = re.compile(rf"([{AMINO_ACIDS}])([1-9][0-9]{{0,2}})([{AMINO_ACIDS}])")
ARCHIVE_URL = "https://zenodo.org/records/15293562/files/DMS_ProteinGym_substitutions.zip"
ARCHIVE_MIRROR = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip"
ARCHIVE_SHA256 = "3a83766254ac9ac9984ec25cb73c6e010ea4418f5e35f143933e6b6e6473b921"
ASSAY_MEMBER = f"DMS_ProteinGym_substitutions/{ASSAY}.csv"
ASSAY_SHA256 = "05b97d36a23363331b271edc56f1496b9a539109fd006e65532f91e9903a0838"
REFERENCE_URL = "https://zenodo.org/records/15293562/files/DMS_substitutions.csv"
REFERENCE_SHA256 = "a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308"
ORIGINAL_URL = "https://ndownloader.figshare.com/files/4820647"
ORIGINAL_SHA256 = "5581230b9888f9960708a4769ce1c9b4453561f0b5f0374525445b4a984d54d5"


@dataclass(frozen=True, slots=True)
class Variant:
    id: str
    mutant: str
    sequence: str
    edits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GFPDataset:
    wild_type: str
    variants: Mapping[str, Variant]
    scores: Mapping[str, float]
    binary: Mapping[str, int]
    provenance: dict

    @property
    def wild_type_id(self) -> str:
        return WT_ID


def _parse_edits(mutant: str, limit: int) -> tuple[tuple[int, str, str], ...]:
    if mutant == WT_ID:
        return ()
    if not isinstance(mutant, str) or not mutant:
        raise ValueError("Use WT or colon-delimited substitutions such as K3R:V55A")
    edits = []
    for token in mutant.split(":"):
        match = SUBSTITUTION.fullmatch(token)
        if match is None:
            raise ValueError(f"Invalid substitution syntax: {token!r}")
        origin, site_text, alternate = match.groups()
        site = int(site_text)
        if site > len(GFP_WT) or GFP_WT[site - 1] != origin or alternate == origin:
            raise ValueError(f"Substitution is incompatible with the GFP reference: {token}")
        edits.append((site, origin, alternate))
    if len(edits) > limit:
        raise ValueError(f"At most {limit} substitutions allowed")
    if len(edits) != len({site for site, _, _ in edits}):
        raise ValueError("Each position may be substituted only once")
    return tuple(sorted(edits))


def _variant(edits: tuple[tuple[int, str, str], ...]) -> Variant:
    result = list(GFP_WT)
    for site, _, alternate in edits:
        result[site - 1] = alternate
    texts = tuple(f"{origin}{site}{alternate}" for site, origin, alternate in edits)
    identity = ":".join(texts) if texts else WT_ID
    return Variant(identity, identity, "".join(result), texts)


def make_variant(
    mutant: str, wild_type: str | None = None, max_mutations: int = MAX_MUTATIONS
) -> Variant:
    """Construct one substitution-only variant of the pinned isolated avGFP."""
    if wild_type is not None and wild_type != GFP_WT:
        raise ValueError("This workflow accepts only the pinned avGFP reporter reference")
    if not isinstance(max_mutations, int) or not 0 <= max_mutations <= MAX_MUTATIONS:
        raise ValueError(f"The GFP edit budget must be between 0 and {MAX_MUTATIONS}")
    return _variant(_parse_edits(mutant, max_mutations))


def parse_mutations(
    mutant: str, wild_type: str | None = None, max_mutations: int = MAX_MUTATIONS
) -> tuple[str, ...]:
    """Validate mutations and return canonical position-sorted tokens."""
    return make_variant(mutant, wild_type, max_mutations).edits


def _check_bytes(content: bytes, expected: str, name: str) -> bytes:
    if sha256(content).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {name}")
    return content


def _download(url: str) -> bytes:
    with urlopen(url, timeout=90) as response:
        return response.read()


def _cache_verified(path: Path, producer, expected: str) -> bytes:
    if path.exists():
        return _check_bytes(path.read_bytes(), expected, path.name)
    content = _check_bytes(producer(), expected, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(dir=path.parent, prefix=".gfp-", delete=False)
    pending = Path(stream.name)
    try:
        with stream:
            stream.write(content)
        try:
            os.link(pending, path)
        except FileExistsError:
            return _check_bytes(path.read_bytes(), expected, path.name)
    finally:
        pending.unlink()
    return content


def _assay_from_release() -> bytes:
    last_error = None
    for url in (ARCHIVE_URL, ARCHIVE_MIRROR):
        try:
            archive = _check_bytes(_download(url), ARCHIVE_SHA256, url)
            with ZipFile(BytesIO(archive)) as handle:
                return handle.read(ASSAY_MEMBER)
        except (OSError, ValueError) as error:
            last_error = error
    raise RuntimeError("Could not retrieve the pinned ProteinGym v1.3 archive") from last_error


def _wt_measurement(original: bytes) -> float:
    table = pd.read_csv(BytesIO(original), sep="\t", usecols=["aaMutations", "medianBrightness", "uniqueBarcodes"])
    rows = table.loc[table["aaMutations"].isna()]
    if len(rows) != 1:
        raise ValueError("The original GFP archive must contain one aggregate wild-type row")
    measurement = float(rows.iloc[0]["medianBrightness"])
    if not np.isfinite(measurement):
        raise ValueError("The archived wild-type measurement must be finite")
    return measurement


@lru_cache(maxsize=1)
def load_dataset(cache_dir: Path | None = None) -> GFPDataset:
    """Load the official measurement release. Only 1–4-edit variants and WT are eligible."""
    cache_dir = DEFAULT_CACHE_DIR if cache_dir is None else Path(cache_dir)
    content = _cache_verified(cache_dir / f"{ASSAY}.csv", _assay_from_release, ASSAY_SHA256)
    original = _cache_verified(cache_dir / "sarkisyan-brightness.tsv", lambda: _download(ORIGINAL_URL), ORIGINAL_SHA256)
    frame = pd.read_csv(BytesIO(content))
    expected = ["mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"]
    if list(frame.columns) != expected or len(frame) != 51_714 or frame.isna().any().any():
        raise ValueError("ProteinGym GFP assay schema, row count, or completeness does not match v1.3")
    wt = make_variant(WT_ID)
    wt_score = _wt_measurement(original)
    variants = {WT_ID: wt}
    scores = {WT_ID: wt_score}
    binary = {WT_ID: int(wt_score > FLUORESCENT_THRESHOLD)}
    all_sequences = set()
    for notation, sequence, measured, is_fluorescent in frame.itertuples(index=False, name=None):
        edits = _parse_edits(notation, len(GFP_WT))
        variant = _variant(edits)
        if not edits or variant.sequence != sequence or sequence in all_sequences:
            raise ValueError(f"Invalid or duplicated ProteinGym GFP sequence: {notation}")
        all_sequences.add(sequence)
        if not np.isfinite(measured) or is_fluorescent != int(measured > FLUORESCENT_THRESHOLD):
            raise ValueError(f"Inconsistent fluorescence measurement: {notation}")
        if len(edits) <= MAX_MUTATIONS:
            variants[variant.id] = variant
            scores[variant.id] = float(measured)
            binary[variant.id] = int(is_fluorescent)
    provenance = {
        "assay": ASSAY,
        "name": "Isolated Aequorea victoria green fluorescent protein",
        "score_scale": SCORE_SCALE,
        "higher_is_better": True,
        "fluorescent_threshold": FLUORESCENT_THRESHOLD,
        "fluorescent_rule": "DMS_score > 2.5",
        "release": "ProteinGym 1.3",
        "release_doi": "10.5281/zenodo.15293562",
        "release_license": "MIT",
        "reference_metadata_url": REFERENCE_URL,
        "reference_metadata_sha256": REFERENCE_SHA256,
        "archive_sha256": ARCHIVE_SHA256,
        "archive_url": ARCHIVE_URL,
        "assay_member": ASSAY_MEMBER,
        "assay_sha256": ASSAY_SHA256,
        "assay_rows": len(frame),
        "assay_duplicates": 0,
        "assay_missing": 0,
        "original_data_doi": "10.6084/m9.figshare.3102154.v1",
        "original_data_license": "CC BY 4.0",
        "original_file_url": ORIGINAL_URL,
        "original_file_sha256": ORIGINAL_SHA256,
        "wild_type_id": WT_ID,
        "wild_type_sequence": GFP_WT,
        "wild_type_measurement": wt_score,
        "wild_type_source": "Original Figshare archive, empty aaMutations row and medianBrightness",
        "eligible_measured_variants_excluding_wild_type": len(variants) - 1,
        "maximum_substitutions": MAX_MUTATIONS,
    }
    return GFPDataset(
        GFP_WT, MappingProxyType(variants), MappingProxyType(scores), MappingProxyType(binary), provenance
    )


def make_split(
    dataset: GFPDataset, seed: int, initial: int = 96, validation: int = 120,
    pool: int = 384, test: int = 120
) -> dict[str, list[str]]:
    """Create a label-independent per-seed split, with measured WT in the initial set."""
    if initial < 2 or validation < 2 or test < 2 or pool < 1:
        raise ValueError("Need initial/validation/test counts of at least two and a nonempty candidate pool")
    population = sorted(key for key in dataset.variants if key != WT_ID)
    if initial - 1 + validation + pool + test > len(population):
        raise ValueError("Requested split exceeds the number of unique measured GFP variants")
    ordered = np.random.default_rng(seed).permutation(population).tolist()
    start_validation = initial - 1
    start_pool = start_validation + validation
    start_test = start_pool + pool
    return {
        "initial": [WT_ID, *ordered[:start_validation]],
        "validation": ordered[start_validation:start_pool],
        "pool": ordered[start_pool:start_test],
        "test": ordered[start_test:start_test + test],
    }


def _features(variants: Sequence[Variant]) -> list[dict]:
    items = []
    for variant in variants:
        if make_variant(variant.mutant) != variant:
            raise ValueError("Predictors require valid variants of the pinned GFP reference")
        entry = {"edit_count": float(len(variant.edits))}
        for token in variant.edits:
            entry[f"position:{token[1:-1]}"] = 1.0
            entry[f"allele:{token}"] = 1.0
        items.append(entry)
    return items


@dataclass(slots=True)
class GFPRegressor:
    encoder: DictVectorizer
    estimator: Ridge | RandomForestRegressor
    name: str

    def predict(self, variants: Sequence[Variant]) -> np.ndarray:
        return np.asarray(self.estimator.predict(self.encoder.transform(_features(variants))), dtype=float)

    def uncertainty(self, variants: Sequence[Variant]) -> np.ndarray:
        if self.name != "rf":
            return np.zeros(len(variants))
        matrix = self.encoder.transform(_features(variants))
        trees = self.estimator.estimators_
        return np.std([tree.predict(matrix) for tree in trees], axis=0)


def fit_predictor(
    variants: Sequence[Variant], observed: Sequence[float], seed: int, model: str = "rf"
) -> GFPRegressor:
    """Fit using feature records and only the caller's observed labels."""
    targets = np.asarray(observed, dtype=float)
    if (
        targets.ndim != 1
        or len(variants) != len(targets)
        or len(variants) < 2
        or not np.isfinite(targets).all()
        or len({variant.id for variant in variants}) != len(variants)
    ):
        raise ValueError("Training needs at least two unique variants and matching observed finite scores")
    if model == "rf":
        estimator = RandomForestRegressor(
            n_estimators=128, min_samples_leaf=2, max_features=1.0, random_state=seed, n_jobs=1
        )
    elif model == "ridge":
        estimator = Ridge(alpha=8.0, solver="lsqr")
    else:
        raise ValueError(f"Unknown GFP predictor: {model!r}")
    encoder = DictVectorizer(sparse=True, sort=True)
    estimator.fit(encoder.fit_transform(_features(variants)), targets)
    return GFPRegressor(encoder, estimator, model)


def _top_k_hits(scores: np.ndarray, positive: np.ndarray, k: int) -> float:
    cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
    greater = scores > cutoff
    ties = scores == cutoff
    return float(positive[greater].sum() + (k - greater.sum()) * positive[ties].mean())


def metrics(
    y_true: Sequence[float], y_pred: Sequence[float], *, top_k: int = 5,
    threshold: float | None = None
) -> dict:
    """Aggregate measures for a sealed evaluation split, with tie-aware ranking at k."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or not len(actual):
        raise ValueError("Predictions and measured labels must be nonempty vectors of the same size")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all() or top_k < 1:
        raise ValueError("Measurements, predictions, and top-k must be valid")
    k = min(top_k, len(actual))
    quantile = float(np.quantile(actual, 0.9))
    high = actual >= quantile if threshold is None else actual > threshold
    fluorescent = actual > FLUORESCENT_THRESHOLD
    expected_hits = _top_k_hits(predicted, high, k)
    varying = len(actual) > 1 and np.ptp(actual) > 0 and np.ptp(predicted) > 0
    return {
        "n": len(actual),
        "spearman": float(spearmanr(actual, predicted).statistic) if varying else None,
        "pearson": float(pearsonr(actual, predicted).statistic) if varying else None,
        "mse": float(mean_squared_error(actual, predicted)),
        "top_k": k,
        "high_threshold": quantile if threshold is None else float(threshold),
        "high_threshold_rule": "true >= split 90th percentile" if threshold is None else "true > supplied threshold",
        "high_prevalence": float(high.mean()),
        "top_k_precision": expected_hits / k,
        "top_k_recall": None if not high.any() else expected_hits / float(high.sum()),
        "ap_high": float(average_precision_score(high, predicted)) if high.any() else None,
        "ap_fluorescent": float(average_precision_score(fluorescent, predicted)) if fluorescent.any() else None,
    }
