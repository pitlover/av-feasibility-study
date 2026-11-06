import json
import numpy as np
import pytest
from cosmos3eval.predictions import load_pred


def test_official_content_action(tmp_path):
    clip = "kitti_09_000000"
    p = tmp_path / clip / "sample_outputs.json"
    p.parent.mkdir(parents=True)
    a = np.zeros((60, 9), dtype=np.float64)
    a[:, 3] = 1.0
    a[:, 7] = 1.0
    p.write_text(json.dumps({"args": {"model_mode":"inverse_dynamics"},
                             "outputs": [{"content": {"action":a.tolist()}, "files":[]}]}), encoding="utf8")
    got, src = load_pred(tmp_path, clip)
    assert src == p
    assert got.shape == (60, 9)
    np.testing.assert_allclose(got, a)


def test_missing_action_is_not_silently_fabricated(tmp_path):
    clip = "kitti_09_000000"
    p = tmp_path / clip / "sample_outputs.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"args": {}, "outputs": [{"content": {}, "files": []}]}),encoding="utf8")
    with pytest.raises(ValueError, match="Action not found"):
        load_pred(tmp_path,clip)


def test_bad_shape_is_rejected(tmp_path):
    clip="kitti_09_000000"
    p=tmp_path/clip/"sample_outputs.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"outputs":[{"content":{"action": [[0]*9]*59},"files":[]}]}),encoding="utf8")
    with pytest.raises(ValueError,match="action must be finite"):
        load_pred(tmp_path,clip)
