"""CPU tests of the Reptile config and episode sampler (numpy only; no framework, no GPU)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import reptile_testutil as tu
from cosmos3av.reptile import (
    MIXED_NAME,
    ROLE_QUERY,
    ROLE_SUPPORT,
    ClipTables,
    ReptileConfig,
    ReptileEpisodeSampler,
    eps_at,
    unique_rows,
)


def make_cfg(**over) -> ReptileConfig:
    base = dict(datasets=['kitti', 'waymo'], dataset_weights={'kitti': 1.0, 'waymo': 1.0}, inner_steps=3,
                clips_per_rank=1, query_clips_per_rank=1, seed=42)
    base.update(over)
    return ReptileConfig(**base)


class ConfigTests(unittest.TestCase):
    def test_roundtrip_and_defaults(self):
        cfg = make_cfg()
        again = ReptileConfig.from_dict(cfg.to_dict())
        self.assertEqual(again.to_dict(), cfg.to_dict())
        self.assertEqual(cfg.support_size(4), 3 * 4 * 1)
        self.assertEqual(cfg.query_size(4), 4)
        self.assertEqual(cfg.items_per_rank_per_task(), 1 + 3)
        with tempfile.TemporaryDirectory() as d:
            p = cfg.save(Path(d) / 'r.json')
            self.assertEqual(ReptileConfig.load(p).to_dict(), cfg.to_dict())
            self.assertIn('"episode_mode"', p.read_text())

    def test_validation(self):
        for bad in (dict(episode_mode='bogus'), dict(dataset_schedule='bogus'), dict(loss_mode='bogus'),
                    dict(inner_steps=0), dict(inner_warmup_steps=4), dict(meta_lr=1.5), dict(meta_lr=0.0),
                    dict(meta_lr_min_ratio=0.0), dict(clips_per_rank=2, query_clips_per_rank=3),
                    dict(dataset_weights={'kitti': 1.0}), dict(dataset_weights={'kitti': 1.0, 'waymo': 0.0}),
                    dict(episode_mode='per_dataset', dataset_schedule='alternate', dataset_weights={'kitti': 2.0, 'waymo': 1.0})):
            with self.assertRaises(ValueError, msg=str(bad)):
                make_cfg(**bad)
        with self.assertRaises(ValueError):
            ReptileConfig.from_dict({'enabled': True})  # not a ReptileConfig field
        self.assertIsNone(make_cfg(inner_grad_clip=0).inner_grad_clip)  # 0 -> disabled
        make_cfg(episode_mode='per_dataset', dataset_schedule='random', dataset_weights={'kitti': 2.0, 'waymo': 1.0})

    def test_eps_schedule(self):
        cfg = make_cfg(meta_lr=0.5, meta_lr_min_ratio=0.1)
        self.assertAlmostEqual(eps_at(cfg, 0, 100), 0.5)
        self.assertAlmostEqual(eps_at(cfg, 50, 100), 0.5 * (1 - 0.9 * 0.5))
        self.assertAlmostEqual(eps_at(cfg, 100, 100), 0.05)
        self.assertAlmostEqual(eps_at(cfg, 1000, 100), 0.05)  # clamped
        vals = [eps_at(cfg, i, 100) for i in range(101)]
        self.assertTrue(all(a >= b for a, b in zip(vals, vals[1:])))
        warm = make_cfg(meta_lr=1.0, meta_lr_warmup_iters=4)
        self.assertAlmostEqual(eps_at(warm, 0, 100), 0.25)
        self.assertAlmostEqual(eps_at(warm, 3, 100), 1.0)
        self.assertAlmostEqual(eps_at(warm, 4, 100), 1.0)


class TableTests(unittest.TestCase):
    def test_unique_rows_sorted_and_deduplicated(self):
        rows = tu.synthetic_rows(4, 2)
        dup = rows + rows[:3]
        uniq = unique_rows(reversed(dup))
        self.assertEqual([r['clip_id'] for r in uniq], sorted(r['clip_id'] for r in rows))
        tables = ClipTables(uniq)
        self.assertEqual(tables.summary(), {'kitti': {'clips': 4, 'sequences': 4}, 'waymo': {'clips': 2, 'sequences': 2}})
        with self.assertRaises(ValueError):
            ClipTables(dup)


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.tables = ClipTables(unique_rows(tu.synthetic_rows(20, 30, seqs_kitti=4, seqs_waymo=10)))

    def _check_partition(self, sampler: ReptileEpisodeSampler, spec):
        """Every rank's plan has the same shape; the ranks' support slots partition the support list (fresh clips)
        and their query slots partition the query list."""
        cfg = sampler.cfg
        per_rank = sampler.cfg.items_per_rank_per_task() * len(spec.tasks)
        plans = [sampler.plan_for_rank(spec, r) for r in range(sampler.world_size)]
        for plan in plans:
            self.assertEqual(len(plan), per_rank)
        for ti, task in enumerate(spec.tasks):
            support_seen, query_seen = [], []
            for plan in plans:
                items = [it for it in plan if it.task == ti]
                roles = [it.role for it in items]
                # query first, then the inner steps in order
                self.assertEqual(roles, [ROLE_QUERY] * cfg.query_clips_per_rank + [ROLE_SUPPORT] * (cfg.inner_steps * cfg.clips_per_rank))
                steps = [it.step for it in items if it.role == ROLE_SUPPORT]
                self.assertEqual(steps, [s for s in range(cfg.inner_steps) for _ in range(cfg.clips_per_rank)])
                support_seen += [it.clip_index for it in items if it.role == ROLE_SUPPORT]
                query_seen += [it.clip_index for it in items if it.role == ROLE_QUERY]
            if cfg.support_clips == 0:
                self.assertEqual(sorted(support_seen), sorted(task.support))
            self.assertEqual(sorted(query_seen), sorted(task.query))

    def test_per_dataset_alternate(self):
        cfg = make_cfg(episode_mode='per_dataset', dataset_schedule='alternate')
        sampler = ReptileEpisodeSampler(self.tables, cfg, world_size=4)
        self.assertEqual(sampler.support_size, 12)
        self.assertEqual(sampler.query_size, 4)
        names = [sampler.spec(j).tasks[0].name for j in range(6)]
        self.assertEqual(names, ['kitti', 'waymo'] * 3)
        for j in range(4):
            spec = sampler.spec(j)
            self.assertEqual(len(spec.tasks), 1)
            task = spec.tasks[0]
            ds = self.tables.datasets_of
            self.assertTrue(all(ds[i] == task.name for i in task.support + task.query))
            self.assertEqual(len(set(task.support)), 12)  # 12 <= 20 clips -> no repeats
            self.assertTrue(set(task.query).isdisjoint(task.support))
            self.assertFalse(task.query_overlaps_support)
            if task.query_sequence_disjoint:
                support_seqs = {self.tables.sequences_of[i] for i in task.support}
                self.assertTrue(all(self.tables.sequences_of[i] not in support_seqs for i in task.query))
            self.assertEqual(task.weight, 1.0)
            self._check_partition(sampler, spec)
        # kitti has only 4 sequences and the support set spans them all -> query cannot be sequence-disjoint
        self.assertFalse(sampler.spec(0).tasks[0].query_sequence_disjoint)
        # one clip per drive: 12 support clips leave 18 untouched drives -> the query comes from other drives
        many_drives = ClipTables(unique_rows(tu.synthetic_rows(20, 30, seqs_kitti=4, seqs_waymo=30)))
        task = ReptileEpisodeSampler(many_drives, cfg, world_size=4).spec(1).tasks[0]
        self.assertEqual(task.name, 'waymo')
        self.assertTrue(task.query_sequence_disjoint)
        support_seqs = {many_drives.sequences_of[i] for i in task.support}
        self.assertTrue(all(many_drives.sequences_of[i] not in support_seqs for i in task.query))
        # the knob off -> plain clip-disjoint sampling, flag never set
        off = ReptileEpisodeSampler(many_drives, make_cfg(query_sequence_disjoint=False), world_size=4).spec(1).tasks[0]
        self.assertFalse(off.query_sequence_disjoint)
        self.assertTrue(set(off.query).isdisjoint(off.support))

    def test_determinism(self):
        cfg = make_cfg()
        a = ReptileEpisodeSampler(self.tables, cfg, 4)
        b = ReptileEpisodeSampler(ClipTables(unique_rows(tu.synthetic_rows(20, 30, 4, 10))), cfg, 4)
        for j in (0, 7, 123):
            self.assertEqual(a.spec(j).summary(), b.spec(j).summary())
            self.assertEqual(a.spec(j).tasks[0].support, b.spec(j).tasks[0].support)
        other = ReptileEpisodeSampler(self.tables, make_cfg(seed=7), 4)
        self.assertNotEqual(a.spec(1).tasks[0].support, other.spec(1).tasks[0].support)
        self.assertNotEqual(a.spec(1).tasks[0].support, a.spec(3).tasks[0].support)

    def test_random_schedule_follows_weights(self):
        cfg = make_cfg(episode_mode='per_dataset', dataset_schedule='random', dataset_weights={'kitti': 3.0, 'waymo': 1.0})
        sampler = ReptileEpisodeSampler(self.tables, cfg, 2)
        names = [sampler.spec(j).tasks[0].name for j in range(600)]
        frac = names.count('kitti') / len(names)
        self.assertTrue(0.68 < frac < 0.82, frac)

    def test_all_datasets(self):
        cfg = make_cfg(episode_mode='all_datasets', dataset_weights={'kitti': 3.0, 'waymo': 1.0})
        sampler = ReptileEpisodeSampler(self.tables, cfg, 4)
        spec = sampler.spec(5)
        self.assertEqual([t.name for t in spec.tasks], ['kitti', 'waymo'])
        self.assertAlmostEqual(spec.tasks[0].weight, 0.75)
        self.assertAlmostEqual(spec.tasks[1].weight, 0.25)
        for t in spec.tasks:
            self.assertTrue(all(self.tables.datasets_of[i] == t.name for i in t.support + t.query))
        self.assertEqual(sampler.items_per_episode(spec), 2 * (1 + 3))
        self._check_partition(sampler, spec)

    def test_mixed_quota(self):
        for weights, expect in (({'kitti': 1.0, 'waymo': 1.0}, {'kitti': 6, 'waymo': 6}),
                                ({'kitti': 3.0, 'waymo': 1.0}, {'kitti': 9, 'waymo': 3})):
            cfg = make_cfg(episode_mode='mixed', dataset_weights=weights)
            sampler = ReptileEpisodeSampler(self.tables, cfg, 4)
            spec = sampler.spec(2)
            self.assertEqual(len(spec.tasks), 1)
            task = spec.tasks[0]
            self.assertEqual(task.name, MIXED_NAME)
            self.assertEqual(task.support_datasets, expect)
            self.assertEqual(len(task.support), 12)
            self.assertTrue(set(task.query).isdisjoint(task.support))
            # a mixture is shuffled, not grouped by dataset
            ds_order = [self.tables.datasets_of[i] for i in task.support]
            self.assertNotEqual(ds_order, sorted(ds_order))
            self._check_partition(sampler, spec)

    def test_fixed_support_cycles(self):
        cfg = make_cfg(support_clips=4, inner_steps=3)
        sampler = ReptileEpisodeSampler(self.tables, cfg, 2)
        spec = sampler.spec(0)
        task = spec.tasks[0]
        self.assertEqual(len(task.support), 4)
        for r in range(2):
            items = [it for it in sampler.plan_for_rank(spec, r) if it.role == ROLE_SUPPORT]
            self.assertEqual([it.clip_index for it in items], [task.support[(s * 2 + r) % 4] for s in range(3)])
        with self.assertRaises(ValueError):
            ReptileEpisodeSampler(self.tables, make_cfg(support_clips=3, clips_per_rank=2), 2)

    def test_clips_per_rank_gt_1(self):
        cfg = make_cfg(inner_steps=2, clips_per_rank=2, query_clips_per_rank=2)
        sampler = ReptileEpisodeSampler(self.tables, cfg, 2)
        self.assertEqual(sampler.support_size, 8)
        self.assertEqual(sampler.query_size, 4)
        spec = sampler.spec(1)
        task = spec.tasks[0]
        for r in range(2):
            items = [it for it in sampler.plan_for_rank(spec, r) if it.role == ROLE_SUPPORT]
            expect = [task.support[s * 4 + r * 2 + c] for s in range(2) for c in range(2)]
            self.assertEqual([it.clip_index for it in items], expect)
            queries = [it.clip_index for it in sampler.plan_for_rank(spec, r) if it.role == ROLE_QUERY]
            self.assertEqual(queries, task.query[r * 2:(r + 1) * 2])
        self._check_partition(sampler, spec)

    def test_tiny_pool_fallbacks(self):
        tables = ClipTables(unique_rows(tu.synthetic_rows(3, 3, seqs_kitti=1, seqs_waymo=1)))
        cfg = make_cfg(inner_steps=2)  # support 8 > 3 clips
        sampler = ReptileEpisodeSampler(tables, cfg, 4)
        spec = sampler.spec(0)
        task = spec.tasks[0]
        self.assertEqual(len(task.support), 8)
        self.assertEqual(len(set(task.support[:3])), 3)  # every clip once before repeats
        self.assertTrue(task.query_overlaps_support)
        self.assertFalse(task.query_sequence_disjoint)
        sampler.plan_for_rank(spec, 3)

    def test_bad_geometry(self):
        with self.assertRaises(ValueError):
            ReptileEpisodeSampler(self.tables, make_cfg(datasets=['kitti', 'av2'], dataset_weights={'kitti': 1.0, 'av2': 1.0}), 2)
        with self.assertRaises(ValueError):
            ReptileEpisodeSampler(self.tables, make_cfg(), 0)
        sampler = ReptileEpisodeSampler(self.tables, make_cfg(), 2)
        with self.assertRaises(ValueError):
            sampler.plan_for_rank(sampler.spec(0), 2)

    def test_summary_is_json(self):
        sampler = ReptileEpisodeSampler(self.tables, make_cfg(episode_mode='all_datasets'), 2)
        text = json.dumps(sampler.spec(0).summary())
        self.assertIn('"support_datasets"', text)
        self.assertTrue(np.isfinite(len(text)))


if __name__ == '__main__':
    unittest.main()
