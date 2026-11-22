"""Check FLIP partition parsing, feature blocks, and aggregated metrics."""

import csv
from io import StringIO

import numpy as np
import pytest

from gb1_audit import CONSTRUCT_LENGTH, POSITIONS, audit_table, encode, precision_at_k
from gb1_audit import single_effect_transfer


def split_fixture(duplicate=False, mark_test_as_validation=False):
    background = "A" * CONSTRUCT_LENGTH
    combo = list(background)
    for position in POSITIONS:
        combo[position - 1] = "C"
    training = list(background)
    training[POSITIONS[0] - 1] = "C"
    rows = [
        [background, 1, "train", ""],
        ["".join(training), 0.4, "train", "True"],
        ["".join(combo), 1.2, "test", "True" if mark_test_as_validation else ""],
    ]
    if duplicate:
        rows.append(rows[1])
    handle = StringIO()
    writer = csv.writer(handle)
    writer.writerow(["sequence", "target", "set", "validation"])
    writer.writerows(rows)
    return handle.getvalue().encode()


def test_official_flags_keep_validation_out_of_training_and_test():
    _, _, _, _, masks, audit = audit_table(split_fixture())
    assert np.flatnonzero(masks["train"]).tolist() == [0]
    assert np.flatnonzero(masks["validation"]).tolist() == [1]
    assert np.flatnonzero(masks["test"]).tolist() == [2]
    assert audit["partitions"]["test"]["edits"] == {4: 1}
    assert audit["partition_intersection_count"] == 0


@pytest.mark.parametrize(
    ("violation", "message"),
    [
        ({"duplicate": True}, "sequence or target audit failed"),
        ({"mark_test_as_validation": True}, "also marked as test"),
    ],
)
def test_audit_rejects_duplicate_or_overlapping_rows(violation, message):
    with pytest.raises(ValueError, match=message):
        audit_table(split_fixture(**violation))


def test_pair_encoding_adds_one_indicator_per_position_pair():
    residues = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
    single = encode(residues).toarray()
    paired = encode(residues, pairwise=True).toarray()
    assert single.shape == (2, 80)
    assert paired.shape == (2, 2480)
    np.testing.assert_array_equal(paired[:, :80], single)
    np.testing.assert_array_equal(single.sum(axis=1), [4, 4])
    np.testing.assert_array_equal(paired.sum(axis=1), [10, 10])


def test_top_k_metrics_average_across_the_cutoff_tie():
    positive = np.array([True, False, True, False])
    estimates = np.array([3.0, 2.0, 2.0, 2.0])
    precision, hits = precision_at_k(positive, estimates, 2)
    assert hits == pytest.approx(1 + 1 / 3)
    assert precision == pytest.approx(2 / 3)


def test_single_effect_transfer_requires_constituent_training_measurements():
    features = np.array([[1, 0], [0, 1], [1, 1]])
    target = np.array([0.5, 0.5, 2.0])
    depth = np.array([1, 1, 2])
    wt = np.array([0, 0])
    query = np.array([False, False, True])
    complete = single_effect_transfer(
        features, target, depth, wt, np.array([True, True, False]), query
    )
    assert complete["covered_by_training_singles"] == 1
    assert complete["observed_above_wt_expected_not_above_wt"] == 1
    assert complete["mean_abs_observed_minus_multiplicative"] == 1.75

    missing = single_effect_transfer(
        features, target, depth, wt, np.array([True, False, False]), query
    )
    assert missing["covered_by_training_singles"] == 0
    assert missing["mean_abs_observed_minus_multiplicative"] is None
    assert missing["observed_above_wt_expected_not_above_wt"] is None
