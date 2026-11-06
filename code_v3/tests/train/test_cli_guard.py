"""No-GPU regression test for Edge-only experiment selection."""
import json
from pathlib import Path
import pytest
from cosmos3av.cli import _toml, main


def test_plan_template_targets_edge():
    text = _toml("pretrain", "edge_test", 10, 1e-5, 10, 4)
    assert 'experiment = "action_inverse_dynamics_av_edge"' in text
    assert 'experiment = "action_inverse_dynamics_av_nano"' not in text


def test_launch_rejects_stale_nano_toml(tmp_path):
    framework = tmp_path / "cosmos-framework"
    script = framework / "cosmos_framework" / "scripts" / "train.py"
    script.parent.mkdir(parents=True)
    script.write_text("# fake; guard must reject before launch")
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"gpus": 4}))
    (run / "sft_config.toml").write_text('experiment = "action_inverse_dynamics_av_nano"\n')
    with pytest.raises(RuntimeError, match="action_inverse_dynamics_av_edge"):
        main(["launch", "--run-dir", str(run), "--framework-root", str(framework)])
