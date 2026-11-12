"""Standalone CPU/stdlib tests for 4-dataset index merging, FPS, variants and source safety."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PKG=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('source_index', PKG/'scripts'/'stage1_source_index.py')
si=importlib.util.module_from_spec(spec);spec.loader.exec_module(si)


def write_clip(root:Path, dataset:str, fps:int, split='train', mode='virtual', *, stale=False):
    clip_id=f'{dataset}_scene_{split}_000000'
    folder=root/dataset/split/clip_id
    folder.mkdir(parents=True)
    (folder/'video.mp4').write_bytes(b'not a real video (index test only)')
    (folder/'actions_raw_9d.npy').write_bytes(b'fake actions')
    meta=dict(clip_id=clip_id,dataset=dataset,split=split,sequence=f'{dataset}_scene_{split}',
              frame_count=61,action_count=60,fps=fps,image_mode=mode,action_normalization='none')
    (folder/'metadata.json').write_text(json.dumps(meta))
    prefix=Path('/stale/output/location') if stale else folder
    row={'clip_id':clip_id,'dataset':dataset,'split':split,'sequence':meta['sequence'],
         'video':str(prefix/'video.mp4'),'action_file':str(prefix/'actions_raw_9d.npy'),
         'meta':str(prefix/'metadata.json')}
    with (root/'index.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

class SourceIndexTest(unittest.TestCase):
    def test_four_datasets_three_fps_and_relocated_paths(self):
        with tempfile.TemporaryDirectory() as t:
            temp=Path(t)
            virtual10=temp/'processed_virtual_native_v2'
            rawhz=temp/'processed_raw_rawhz'/'original_native'
            virtual_raw=temp/'processed_virtual_rawhz'/'virtual_native'
            write_clip(virtual10,'kitti',10,stale=True)
            write_clip(virtual10,'waymo',10)
            write_clip(rawhz,'av2',20,mode='original')
            write_clip(virtual_raw,'nuscenes',12)
            spec={'base':str(temp),'processed':str(virtual10),
                  'datasets':['kitti','waymo','av2','nuscenes'],
                  'sources':{'av2':str(rawhz.parent),'nuscenes':str(virtual_raw.parent)},
                  'weights':{x:1.0 for x in ('kitti','waymo','av2','nuscenes')}}
            output=temp/'run'/'input_sources'
            result=si.merge(spec,output)
            self.assertEqual(result['fps'],{'kitti':{'10':1},'waymo':{'10':1},'av2':{'20':1},'nuscenes':{'12':1}})
            rows=[json.loads(line) for line in (output/'index.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),4)
            self.assertTrue(all(Path(row['video']).exists() for row in rows))
            self.assertEqual(json.loads((output/'source_spec.json').read_text())['datasets'],spec['datasets'])

    def test_no_nuscenes_10hz(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'original_10hz'
            write_clip(root,'nuscenes',10)
            spec={'base':t,'processed':str(root),'datasets':['nuscenes'],'sources':{},'weights':{'nuscenes':1.0}}
            with self.assertRaisesRegex(ValueError,'nuScenes prepared clips must have source-native 12Hz'):
                si.merge(spec,Path(t)/'out')
            self.assertFalse((Path(t)/'out'/'index.jsonl').exists())

    def test_ambiguous_variants_require_subfolder(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'processed_rawhz'
            write_clip(root/'virtual_native','kitti',10)
            write_clip(root/'original_native','kitti',10,mode='original')
            with self.assertRaisesRegex(ValueError,'ambiguous source'):
                si.locate_source(str(root),'kitti',t)

if __name__=='__main__':unittest.main()
