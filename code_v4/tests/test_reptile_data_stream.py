"""The Reptile clip stream: worker interleaving (torch only) and the real Cosmos packing-loader stack on the bundled
sample clips (needs cosmos_framework + cv2; single-process gloo world of 1; CPU, no GPU)."""
from __future__ import annotations

import importlib.util
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import reptile_testutil as tu

HAVE_TORCH = importlib.util.find_spec('torch') is not None
HAVE_FRAMEWORK = HAVE_TORCH and importlib.util.find_spec('cosmos_framework') is not None and importlib.util.find_spec('cv2') is not None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@unittest.skipUnless(HAVE_TORCH, 'torch not installed')
class StreamOrderTests(unittest.TestCase):
    def test_workers_interleave_into_the_rank_plan(self):
        from cosmos3av import reptile_data
        from cosmos3av.reptile import ClipTables, ReptileConfig, ReptileEpisodeSampler, unique_rows

        tables = ClipTables(unique_rows(tu.synthetic_rows(20, 30, 4, 10)))
        cfg = ReptileConfig(episode_mode='all_datasets', inner_steps=3, clips_per_rank=1, query_clips_per_rank=1,
                            datasets=['kitti', 'waymo'], dataset_weights={'kitti': 1.0, 'waymo': 1.0}, seed=3)
        world, rank, offset = 2, 1, 5
        sampler = ReptileEpisodeSampler(tables, cfg, world)

        class FakeDataset:
            def __len__(self):
                return len(tables)

            def __getitem__(self, i):
                return {'clip': tables.clip_ids[i]}

        ds = reptile_data.ReptileClipStreamDataset(FakeDataset(), sampler, spec_offset=offset)
        ds.shard_rank, ds.shard_world_size = rank, world
        expected = []
        for j in (offset, offset + 1, offset + 2):
            expected += sampler.plan_for_rank(sampler.spec(j), rank)
        n = len(expected)
        per_worker = []
        for w in range(3):
            with mock.patch.object(reptile_data, 'get_worker_info', return_value=SimpleNamespace(id=w, num_workers=3)):
                it = iter(ds)
                per_worker.append([next(it) for _ in range((n + 2 - w) // 3)])
        merged = []
        for p in range(n):
            merged.append(per_worker[p % 3][p // 3])
        for got, item in zip(merged, expected):
            self.assertEqual(got['clip'], tables.clip_ids[item.clip_index])
            self.assertEqual(got['_reptile_spec_id'], item.spec_id)
            self.assertEqual(got['_reptile_task'], item.task)
            self.assertEqual(got['_reptile_role'], item.role)
            self.assertEqual(got['_reptile_step'], item.step)
            self.assertEqual(got['_reptile_clip_index'], item.clip_index)
        self.assertEqual(merged[0]['_reptile_spec_id'], offset)  # resume: the stream starts at spec_offset

    def test_world_size_mismatch_is_detected(self):
        from cosmos3av import reptile_data
        from cosmos3av.reptile import ClipTables, ReptileConfig, ReptileEpisodeSampler, unique_rows

        tables = ClipTables(unique_rows(tu.synthetic_rows(8, 8, 2, 2)))
        cfg = ReptileConfig(datasets=['kitti', 'waymo'], dataset_weights={'kitti': 1.0, 'waymo': 1.0}, inner_steps=1, inner_warmup_steps=0)
        ds = reptile_data.ReptileClipStreamDataset(object(), ReptileEpisodeSampler(tables, cfg, 2), 0)
        ds.shard_world_size = 4
        with self.assertRaises(RuntimeError):
            next(iter(ds))


@unittest.skipUnless(HAVE_FRAMEWORK and tu.have_sample_clips(), 'cosmos_framework / cv2 / sample clips not available')
class FrameworkLoaderTests(unittest.TestCase):
    """Builds exactly the loader stack the trainer builds (world size 1) and pulls one full episode."""

    def test_packed_batches_follow_the_plan(self):
        import torch
        import torch.distributed as dist

        from cosmos3av.reptile import ClipTables, ReptileConfig, ReptileEpisodeSampler, ROLE_SUPPORT, unique_rows
        from cosmos3av.reptile_data import META_KEYS, build_reptile_stream_dataset
        from cosmos3av.reptile_train import check_batch
        from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
        from cosmos_framework.utils.lazy_config import LazyCall as L

        rows = [r for r in tu.sample_rows() if r['split'] == 'train' and r['dataset'] in ('kitti', 'waymo')]
        unique = unique_rows(rows)
        with tempfile.TemporaryDirectory() as d:
            manifest = Path(d) / 'train_unique.jsonl'
            manifest.write_text(''.join(__import__('json').dumps(r) + '\n' for r in unique))
            cfg = ReptileConfig(episode_mode='all_datasets', inner_steps=2, clips_per_rank=1, query_clips_per_rank=1,
                                datasets=['kitti', 'waymo'], dataset_weights={'kitti': 1.0, 'waymo': 1.0}, manifest=str(manifest), seed=42)
            cfg_path = cfg.save(Path(d) / 'reptile_config.json')
            dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{_free_port()}', rank=0, world_size=1)
            try:
                # tokenizer_config=None keeps the test offline (no VLM tokenizer files); the JSON prompt would then stay a
                # dict that the packing loader cannot split, so use the plain-string prompt here. A real run passes the
                # experiment's tokenizer and format_prompt_as_json=True, exactly like the joint baseline.
                dataset_cfg = L(build_reptile_stream_dataset)(
                    reptile_config_path=str(cfg_path), world_size=1, spec_offset=0,
                    tokenizer_config=None, max_action_dim=64, resolution='480', cfg_dropout_rate=0.1, format_prompt_as_json=False,
                )
                rank_loader = RankPartitionedDataLoader(datasets={'av': {'ratio': 1, 'dataset': dataset_cfg}}, batch_size=1,
                                                        num_workers=0, in_order=True)
                loader = PackingDataLoader(dataloader=rank_loader, max_samples_per_batch=1, max_sequence_length=None, patch_spatial=2,
                                           tokenizer_spatial_compression_factor=16, tokenizer_temporal_compression_factor=4,
                                           sound_latent_fps=0, audio_sample_rate=48000, dataset_name='action_libero')
                sampler = ReptileEpisodeSampler(ClipTables(unique), cfg, 1)
                spec = sampler.spec(0)
                plan = sampler.plan_for_rank(spec, 0)
                self.assertEqual(len(plan), 2 * (1 + 2))
                it = iter(loader)
                for item in plan:
                    batch = next(it)
                    meta = check_batch(batch, spec_id=0, task=item.task, role=item.role, step=item.step, expected_samples=1)
                    self.assertEqual(meta['_reptile_clip_index'], [item.clip_index])
                    for k in META_KEYS:
                        self.assertIn(k, batch)
                    # the baseline's packed-batch format: list-of-per-sample lists for the multi-item keys
                    self.assertIsInstance(batch['video'], list)
                    self.assertIsInstance(batch['video'][0], list)
                    video = batch['video'][0][0]
                    self.assertEqual(video.shape[1], 61)
                    self.assertEqual(batch['action'][0][0].shape[-2:], (60, 64))
                    self.assertEqual(len(batch['sequence_plan']), 1)
                    self.assertEqual(len(batch['domain_id']), 1)
                    self.assertIn('_num_tokens', batch)
                    self.assertEqual(batch['dataset_name'], ['av'])  # the rank-partition name, as in the baseline
                    clip = unique[item.clip_index]
                    self.assertEqual(clip['dataset'], spec.tasks[item.task].name)
                # the next batch belongs to episode 1
                nxt = next(it)
                self.assertEqual(int(torch.as_tensor(nxt['_reptile_spec_id'][0]).reshape(-1)[0]), 1)
                # a wrong expectation is caught
                with self.assertRaises(RuntimeError):
                    check_batch(nxt, spec_id=1, task=0, role=ROLE_SUPPORT, step=0, expected_samples=1)
            finally:
                dist.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
