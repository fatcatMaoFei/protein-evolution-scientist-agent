from pathlib import Path

import pandas as pd

from gfp_v2.cases import WT, decision_cases, extract


def test_archive_cases_distinguish_real_high_predictions_and_signed_errors():
    result = extract(Path("evidence/gfp-v2"))
    prediction = result["prediction"]
    assert prediction["accurate_above_wt"]["measured"] > WT
    assert prediction["accurate_above_wt"]["predicted"] > WT
    assert prediction["largest_overestimate"]["observed_minus_prediction"] < 0
    assert prediction["largest_underestimate"]["observed_minus_prediction"] > 0
    assert result["decision"]["agent_beats_direct"]["new_best"] > 0
    assert result["decision"]["agent_loses_to_direct"]["new_best"] < 0


def test_absent_agent_win_has_no_substitute():
    pairs = pd.DataFrame([{"control": "B_predictor", "treatment": "D_agent", "block": 1, "new_best": 0.0}])
    assert decision_cases(pairs, pd.DataFrame()) == {"agent_beats_direct": None, "agent_loses_to_direct": None}
