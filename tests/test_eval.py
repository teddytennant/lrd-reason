"""Eval harness smoke (small fixture, StubLLM)."""

from lrd_reason.eval import gsm8k, math_bench, multi_turn
from lrd_reason.eval.metrics import accuracy, extract_final_number
from lrd_reason.infer.pipeline import InferenceRunner


def test_extract_final_number():
    assert extract_final_number("the answer is 42") == "42"
    assert extract_final_number("3 cats and 7 dogs, total 10") == "10"
    assert extract_final_number("no number here") is None


def test_accuracy():
    assert accuracy(["The answer is 5"], ["5"]) == 1.0
    assert accuracy(["The answer is 6"], ["5"]) == 0.0


def test_gsm8k_evaluate_runs(smoke_spec, tmp_path):
    runner = InferenceRunner(spec=smoke_spec, sessions_dir=tmp_path / "s")
    res = gsm8k.evaluate(runner, n=2, num_diffusion_steps=1)
    assert "accuracy" in res
    assert res["n"] >= 1


def test_math_evaluate_runs(smoke_spec, tmp_path):
    runner = InferenceRunner(spec=smoke_spec, sessions_dir=tmp_path / "s")
    res = math_bench.evaluate(runner, n=2, num_diffusion_steps=1)
    assert "accuracy" in res


def test_multi_turn_evaluate_runs(smoke_spec, tmp_path):
    runner = InferenceRunner(spec=smoke_spec, sessions_dir=tmp_path / "s")
    res = multi_turn.evaluate(runner, n_scenarios=2, num_diffusion_steps=1)
    assert "consistency" in res
    assert "contradiction_rate" in res
