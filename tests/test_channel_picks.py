import numpy as np

from scripts.evaluate_from_npz import parse_picks as parse_npz_eval_picks
from scripts.evaluate_trained_model import parse_picks as parse_trained_eval_picks
from scripts.real_time_classifier import resolve_numeric_picks as resolve_realtime_numeric_picks
from scripts.record_trials_lsl import resolve_numeric_picks as resolve_record_numeric_picks


def test_source_channel_picks_are_used_when_available():
    requested = [2, 3, 4]

    assert resolve_record_numeric_picks(requested, 8) == requested
    assert resolve_realtime_numeric_picks(requested, 8) == requested
    np.testing.assert_array_equal(parse_trained_eval_picks("2,3,4", 8), np.asarray(requested))
    np.testing.assert_array_equal(parse_npz_eval_picks("2,3,4", 8), np.asarray(requested))


def test_source_channel_picks_fall_back_when_data_is_already_narrowed():
    requested = [2, 3, 4]
    narrowed = [0, 1, 2]

    assert resolve_record_numeric_picks(requested, 3) == narrowed
    assert resolve_realtime_numeric_picks(requested, 3) == narrowed
    np.testing.assert_array_equal(parse_trained_eval_picks("2,3,4", 3), np.asarray(narrowed))
    np.testing.assert_array_equal(parse_npz_eval_picks("2,3,4", 3), np.asarray(narrowed))