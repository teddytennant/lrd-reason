"""End-to-end CPU smoke test of LRDPipeline.

Uses StubEncoder + GRU recurrent + tiny diffusion + StubLLM. Goes through every
component boundary without touching the network or HF weights.
"""

import torch

from lrd_reason.models.pipeline import LRDPipeline


def test_pipeline_forward_no_nans(smoke_spec):
    pipe = LRDPipeline(
        model_cfg=smoke_spec.model,
        ablation=smoke_spec.ablation,
        task_dim=smoke_spec.data.task_embed_dim,
        device="cpu",
        dtype=torch.float32,
    )
    targets = torch.randn(2, smoke_spec.model.latent_dim)
    task_embeds = torch.randn(2, smoke_spec.data.task_embed_dim)
    out = pipe.training_step(
        prompts=["What is 2+2?", "Hello world."],
        target_latents=targets,
        prev_states=None,
        task_embed=task_embeds,
        next_target_latents=targets.roll(-1, dims=0),
        contrastive_weight=0.1,
    )
    assert "diffusion" in out.losses
    assert "contrastive" in out.losses
    for v in out.losses.values():
        assert torch.isfinite(v)


def test_pipeline_generate(smoke_spec):
    pipe = LRDPipeline(
        model_cfg=smoke_spec.model,
        ablation=smoke_spec.ablation,
        task_dim=smoke_spec.data.task_embed_dim,
        device="cpu",
        dtype=torch.float32,
    )
    input_ids = torch.zeros(1, 4, dtype=torch.long)
    out_ids, state, plan = pipe.generate(
        prompts=["Hello"],
        input_ids=input_ids,
        prev_state=None,
        task_embed=None,
        num_diffusion_steps=2,
        max_new_tokens=4,
    )
    assert out_ids.shape[0] == 1
    assert out_ids.shape[1] == 4 + 4  # original + generated
    assert state.shape == (1, smoke_spec.model.recurrent.hidden_dim)
    assert plan.shape == (1, smoke_spec.model.latent_dim)


def test_pipeline_persistent_state(smoke_spec):
    pipe = LRDPipeline(
        model_cfg=smoke_spec.model,
        ablation=smoke_spec.ablation,
        task_dim=smoke_spec.data.task_embed_dim,
        device="cpu",
        dtype=torch.float32,
    )
    input_ids = torch.zeros(1, 4, dtype=torch.long)
    _, state1, _ = pipe.generate(
        prompts=["First"], input_ids=input_ids, prev_state=None, task_embed=None,
        num_diffusion_steps=1, max_new_tokens=2,
    )
    _, state2, _ = pipe.generate(
        prompts=["Second"], input_ids=input_ids, prev_state=state1, task_embed=None,
        num_diffusion_steps=1, max_new_tokens=2,
    )
    # State carries forward (different from a fresh start with the same prompt).
    _, fresh, _ = pipe.generate(
        prompts=["Second"], input_ids=input_ids, prev_state=None, task_embed=None,
        num_diffusion_steps=1, max_new_tokens=2,
    )
    assert not torch.allclose(state2, fresh)


def test_pipeline_ablation_disables_components(smoke_spec):
    from lrd_reason.config import AblationConfig

    # No recurrent: state should be zeros.
    no_recur = AblationConfig(use_recurrent=False, use_diffusion=True, zero_prefix=False)
    pipe = LRDPipeline(
        model_cfg=smoke_spec.model, ablation=no_recur,
        task_dim=smoke_spec.data.task_embed_dim, device="cpu",
    )
    z = torch.randn(2, smoke_spec.model.latent_dim)
    s = pipe.step_recurrent(z, None, None)
    assert torch.allclose(s, torch.zeros_like(s))

    # No diffusion: refine returns z unchanged.
    no_diff = AblationConfig(use_recurrent=True, use_diffusion=False, zero_prefix=False)
    pipe2 = LRDPipeline(
        model_cfg=smoke_spec.model, ablation=no_diff,
        task_dim=smoke_spec.data.task_embed_dim, device="cpu",
    )
    s2 = torch.randn(2, smoke_spec.model.recurrent.hidden_dim)
    refined = pipe2.refine(z, s2, None, num_steps=4)
    assert torch.allclose(refined, z)
