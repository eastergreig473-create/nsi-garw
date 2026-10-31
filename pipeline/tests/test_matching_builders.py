import numpy as np

from nsi_garw.adapters import LocalFeatures
from nsi_garw.matching_builders import pair_matching_record, summarize_pair_matching


def test_exact_matching_and_zero_match_pairs_remain_in_mean() -> None:
    coordinates = np.array([[2., 4.], [8., 10.], [20., 6.]])
    first = LocalFeatures(coordinates, np.eye(3), np.ones(3), {})
    translated = LocalFeatures(coordinates + [3., -1.], np.eye(3), np.ones(3), {})
    homography = np.array([[1., 0., 3.], [0., 1., -1.], [0., 0., 1.]])
    correct = pair_matching_record(first, translated, homography,
        pair_id="s1_1_2", sequence="s1", split="test")
    empty = LocalFeatures(np.empty((0, 2)), np.empty((0, 3)), np.empty(0), {})
    failure = pair_matching_record(first, empty, homography,
        pair_id="s2_1_2", sequence="s2", split="test")
    assert correct["mma_at_3"] == 1.
    assert failure["mma_at_3"] == 0.
    summary = summarize_pair_matching([correct, failure], bootstrap_replicates=100)
    test = summary["held_out_test_sequences"]
    assert test["sequence_count"] == 2
    assert test["zero_match_pairs"] == 1
    assert test["metrics"]["mma_at_3"]["estimate"] == 0.5
