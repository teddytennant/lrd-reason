"""Inference runner smoke test (no checkpoints, untrained pipeline)."""

from lrd_reason.infer.pipeline import InferenceRunner


def test_runner_chat_two_turns(smoke_spec, tmp_path):
    runner = InferenceRunner(spec=smoke_spec, sessions_dir=tmp_path / "sessions")
    r1 = runner.chat("sess-1", "Hello")
    r2 = runner.chat("sess-1", "Again")
    assert isinstance(r1, str)
    assert isinstance(r2, str)
    # Second-turn state file should exist.
    assert (tmp_path / "sessions" / "sess-1.pt").exists()
