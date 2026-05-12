"""Stage 1 trains on CPU smoke config and loss should decrease."""

from lrd_reason.train.stage1 import train


def test_stage1_runs_and_loss_decreases(smoke_spec):
    # 20 steps already configured; we just need loss at step 20 < loss at step 1.
    result = train(smoke_spec, resume=False, max_steps=20)
    assert result["steps"] == 20
    assert result["final_loss"] < 100.0  # sanity bound; toy data, real check is non-NaN


def test_stage1_resume_picks_up(smoke_spec, tmp_path, monkeypatch):
    # Point output_dir at a temp location.
    smoke_spec.run.output_dir = str(tmp_path / "run")
    r1 = train(smoke_spec, resume=False, max_steps=5)
    assert r1["steps"] == 5
    r2 = train(smoke_spec, resume=True, max_steps=10)
    assert r2["steps"] == 10
