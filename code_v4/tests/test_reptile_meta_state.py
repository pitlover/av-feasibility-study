"""CPU tests of the Reptile theta bookkeeping (torch, no framework, no GPU)."""
from __future__ import annotations

import math
import unittest

import reptile_testutil  # noqa: F401 - puts the code root on sys.path

try:
    import torch
    from torch import nn

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, 'torch not installed')
class MetaStateTests(unittest.TestCase):
    def _net(self):
        torch.manual_seed(0)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.moe_gen = nn.Linear(3, 3, bias=False)
                self.action2llm = nn.Linear(2, 2, bias=False)
                self.frozen = nn.Linear(2, 2)
                self.vae2llm = nn.Linear(2, 1)

        net = Net()
        for p in net.frozen.parameters():
            p.requires_grad_(False)
        return net

    def _perturb(self, state, scale):
        deltas = {}
        with torch.no_grad():
            for n, p in state.params.items():
                d = torch.full_like(p.data, scale) * (1 + torch.arange(p.numel(), dtype=p.dtype).reshape(p.shape))
                p.data.add_(d)
                deltas[n] = d
        return deltas

    def test_theta_covers_trainable_set_and_groups(self):
        from cosmos3av.reptile_meta import ReptileMetaState

        net = self._net()
        state = ReptileMetaState(net)
        self.assertEqual(sorted(state.params), ['action2llm.weight', 'moe_gen.weight', 'vae2llm.bias', 'vae2llm.weight'])
        self.assertEqual(state.groups['moe_gen.weight'], 'moe_gen')
        self.assertEqual(state.groups['action2llm.weight'], 'action_heads')
        self.assertEqual(state.groups['vae2llm.weight'], 'vae2llm')
        self.assertEqual(state.numel_by_group(), {'action_heads': 4, 'moe_gen': 9, 'vae2llm': 3})
        self.assertEqual(state.numel, 16)
        self.assertEqual(state.metadata()['algorithm'], 'reptile')

    def test_meta_step_sgd_and_restore(self):
        from cosmos3av.reptile_meta import ReptileMetaState

        net = self._net()
        state = ReptileMetaState(net)
        theta0 = {n: t.clone() for n, t in state.theta.items()}
        deltas = self._perturb(state, 0.1)
        disp = state.displacement_norms_by_group()
        for g in ('moe_gen', 'action_heads', 'vae2llm'):
            expect = math.sqrt(sum(float((d * d).sum()) for n, d in deltas.items() if state.groups[n] == g))
            self.assertAlmostEqual(disp[g], expect, places=5)
        total = math.sqrt(sum(float((d * d).sum()) for d in deltas.values()))
        norm = state.meta_step(0.5)
        self.assertAlmostEqual(norm, 0.5 * total, places=5)
        for n, p in state.params.items():
            torch.testing.assert_close(state.theta[n], theta0[n] + 0.5 * deltas[n])
            torch.testing.assert_close(p.data, state.theta[n])  # model restored to theta
        # frozen parameter untouched
        self.assertFalse(any('frozen' in n for n in state.params))
        # restore after a perturbation without a meta step
        self._perturb(state, 1.0)
        state.restore_to_model()
        for n, p in state.params.items():
            torch.testing.assert_close(p.data, state.theta[n])

    def test_weighted_accumulation(self):
        from cosmos3av.reptile_meta import ReptileMetaState

        net = self._net()
        state = ReptileMetaState(net)
        theta0 = {n: t.clone() for n, t in state.theta.items()}
        d1 = self._perturb(state, 0.1)
        state.accumulate_displacement(0.75)
        state.restore_to_model()
        d2 = self._perturb(state, -0.3)
        state.accumulate_displacement(0.25)
        state.meta_step(1.0)
        for n in state.params:
            torch.testing.assert_close(state.theta[n], theta0[n] + 0.75 * d1[n] + 0.25 * d2[n])
        self.assertIsNone(state.accum)
        self.assertEqual(state.accum_weight, 0.0)

    def test_adam_meta_optimizer_runs(self):
        from cosmos3av.reptile_meta import ReptileMetaState

        state = ReptileMetaState(self._net(), meta_optimizer='adam')
        self._perturb(state, 0.1)
        norm = state.meta_step(0.1)
        self.assertTrue(norm > 0)
        with self.assertRaises(ValueError):
            ReptileMetaState(self._net(), meta_optimizer='rmsprop')

    def test_optimizer_helpers(self):
        from cosmos3av.reptile_meta import capture_base_lrs, reset_optimizer_state, set_lr_scale

        net = self._net()
        params = [p for p in net.parameters() if p.requires_grad]
        opt = torch.optim.Adam([{'params': params[:1], 'lr': 1e-3}, {'params': params[1:], 'lr': 5e-3}])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1e-6)  # LambdaLR rewrites lr but keeps initial_lr
        self.assertEqual(capture_base_lrs(opt), [[1e-3, 5e-3]])
        set_lr_scale(opt, capture_base_lrs(opt), 0.5)
        self.assertEqual([g['lr'] for g in opt.param_groups], [5e-4, 2.5e-3])
        loss = sum((p * p).sum() for p in params)
        loss.backward()
        opt.step()
        self.assertTrue(len(opt.state) > 0)
        opt.param_groups[0]['step'] = 5
        reset_optimizer_state(opt)
        self.assertEqual(len(opt.state), 0)
        self.assertNotIn('step', opt.param_groups[0])
        del sched

    def test_names_and_groups(self):
        from cosmos3av.reptile_meta import canonical_param_name, param_group

        self.assertEqual(canonical_param_name('net._orig_mod.blocks.0._checkpoint_wrapped_module.moe_gen.w'), 'net.blocks.0.moe_gen.w')
        self.assertEqual(param_group('blocks.3.mlp_moe_gen.up_proj.weight'), 'moe_gen')
        self.assertEqual(param_group('time_embedder.linear_1.weight'), 'time_embedder')
        self.assertEqual(param_group('action_modality_embed.weight'), 'action_heads')
        self.assertEqual(param_group('llm2action.linear.weight'), 'action_heads')
        self.assertEqual(param_group('k_norm_und_for_gen.weight'), 'k_norm_und_for_gen')
        self.assertEqual(param_group('blocks.0.attn.q_proj.lora_A.weight'), 'lora')
        self.assertEqual(param_group('something.else'), 'other')

    def test_mean_or_nan(self):
        from cosmos3av.reptile_meta import mean_or_nan

        self.assertTrue(math.isnan(mean_or_nan([])))
        self.assertTrue(math.isnan(mean_or_nan([float('nan')])))
        self.assertAlmostEqual(mean_or_nan([1.0, float('nan'), 3.0]), 2.0)


if __name__ == '__main__':
    unittest.main()
