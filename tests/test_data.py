from pathlib import Path

import torch

from lrd_reason.data.collate import collate_pairs
from lrd_reason.data.cot_generator import CoTGenerator, split_cot_answer
from lrd_reason.data.dataset import LatentPairDataset


def test_dataset_iter_yields_expected_keys(smoke_fixture_path):
    ds = LatentPairDataset(
        jsonl_path=smoke_fixture_path,
        latent_dim=16,
        latents_path=None,
        task_embed_dim=8,
    )
    items = list(ds)
    assert len(items) == 8
    for it in items:
        assert set(it.keys()) >= {
            "prompt", "gold_cot", "gold_answer", "session_id", "turn_idx",
            "target_latent", "task_embed",
        }
        assert it["target_latent"].shape == (16,)
        assert it["task_embed"].shape == (8,)


def test_collate_stacks(smoke_fixture_path):
    ds = LatentPairDataset(
        jsonl_path=smoke_fixture_path, latent_dim=16, task_embed_dim=8,
    )
    items = list(ds)[:4]
    batch = collate_pairs(items)
    assert batch["target_latents"].shape == (4, 16)
    assert batch["task_embeds"].shape == (4, 8)
    assert len(batch["prompts"]) == 4
    assert batch["turn_idxs"].shape == (4,)


def test_target_latent_deterministic(smoke_fixture_path):
    ds1 = LatentPairDataset(jsonl_path=smoke_fixture_path, latent_dim=8, task_embed_dim=0)
    ds2 = LatentPairDataset(jsonl_path=smoke_fixture_path, latent_dim=8, task_embed_dim=0)
    a = list(ds1)[0]["target_latent"]
    b = list(ds2)[0]["target_latent"]
    assert torch.allclose(a, b)


def test_split_cot_answer_with_marker():
    cot, ans = split_cot_answer("Step 1: do x.\nStep 2: do y.\nFinal answer: 42")
    assert "Step 1" in cot
    assert ans == "42"


def test_split_cot_answer_no_marker():
    cot, ans = split_cot_answer("Step 1: do x.\nStep 2: do y.\n42")
    assert ans == "42"


class _FakeEngine:
    def chat(self, messages_list, **kwargs):
        # Each output mirrors the user message with a fake answer line.
        out = []
        for msgs in messages_list:
            user = next(m["content"] for m in msgs if m["role"] == "user")
            out.append({"text": f"Reasoning about: {user}\nAnswer: 42"})
        return out


def test_cot_generator_e2e(tmp_path):
    gen = CoTGenerator(engine=_FakeEngine())
    rows = gen.generate(["What is 6 * 7?"], session_ids=["s0"], turn_idxs=[0])
    assert len(rows) == 1
    r = rows[0]
    assert r["prompt"] == "What is 6 * 7?"
    assert "Reasoning about" in r["gold_cot"]
    assert r["gold_answer"] == "42"
    assert r["session_id"] == "s0"

    out = tmp_path / "out.jsonl"
    n = gen.generate_to_jsonl(["q1", "q2"], out_path=out, append=False)
    assert n == 2
    assert sum(1 for _ in out.open()) == 2


def test_encode_targets_script_smoke(tmp_path, smoke_fixture_path, smoke_spec):
    """Exercise the new encode_targets script (implements the missing LAUNCH step)."""
    import subprocess
    import sys

    out_pt = tmp_path / "latents.pt"
    script = Path(__file__).parents[1] / "scripts" / "encode_targets.py"
    cmd = [
        sys.executable,
        str(script),
        "--config",
        str(Path(__file__).parents[1] / "configs" / "smoke.yaml"),
        "--inputs",
        str(smoke_fixture_path),
        "--out",
        str(out_pt),
        "--batch-size",
        "4",
        "--device",
        "cpu",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert out_pt.exists()

    blob = torch.load(out_pt, map_location="cpu", weights_only=False)
    assert isinstance(blob, dict)
    assert len(blob) >= 1
    for v in blob.values():
        assert v.shape == (smoke_spec.model.latent_dim,)
        break
