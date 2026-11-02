import json
from pathlib import Path
import numpy as np
import cv2
import torch

from camera import K_VIRTUAL, WIDTH, HEIGHT
from geometry import kitti_cam2_from_pose_cam0, waymo_optical_pose, verify_trajectory, relative_poses
from build_index import index_kitti, write_json
from datasets import PairDataset, load_manifest, load_virtual_bgr, rgb_tensor
from materialize import main as dummy_materialize_main  # validate import
from model import PoseResNet18


def test_gt_pose_conversion():
    K = np.array([[700.,0,640.],[0,700.,192.],[0,0,1.]])
    P0 = np.column_stack([K, np.zeros(3)])
    P2 = np.column_stack([K, K @ np.array([-0.54,0,0])])
    cam2 = kitti_cam2_from_pose_cam0(np.eye(4), P0, P2)
    assert np.allclose(cam2[:3,3], [0.54,0,0])
    T = waymo_optical_pose(np.eye(4), np.eye(4))
    assert np.allclose(T[:3,:3] @ [0,0,1], [1,0,0])
    P = np.tile(np.eye(4), (3,1,1))
    P[:,0,3] = [0, 1, 2]
    verify_trajectory(P)
    assert np.allclose(relative_poses(P)[:, 0, 3], [1,1])


def test_prepared_online_and_model(tmp_path):
    import sys
    from argparse import Namespace
    from materialize import main
    source = tmp_path / 'dataset'
    path = source / 'sequences' / '00'
    (path / 'image_2').mkdir(parents=True)
    pose_dir = tmp_path / 'poses'
    pose_dir.mkdir()
    K = np.array([[700.,0,640.],[0,700.,192.],[0,0,1.]])
    P0 = np.column_stack([K, np.zeros(3)])
    P2 = np.column_stack([K, K @ np.array([-0.54,0,0])])
    (path / 'calib.txt').write_text('P0: ' + ' '.join(map(str, P0.flatten())) + '\nP2: ' + ' '.join(map(str, P2.flatten())) + '\n')
    np.savetxt(path / 'times.txt', np.arange(4)*.1)
    poses = []
    for i in range(4):
        T = np.eye(4)
        T[2,3] = float(i)*.1
        poses.append(T[:3,:].reshape(-1))
        image = np.zeros((384,1280,3), dtype=np.uint8)
        image[:] = [i*15,70,110]
        assert cv2.imwrite(str(path / 'image_2' / f'{i:06d}.png'),image)
    np.savetxt(pose_dir/'00.txt', np.stack(poses))
    index = dict(format_version=3, stage='raw', camera={
        'K_virtual':K_VIRTUAL.tolist(), 'virtual_size':[WIDTH,HEIGHT],
        'virtual_hfov_deg':45.0
    }, sequences=index_kitti(Namespace(root=str(source),poses_root=str(pose_dir),val_seqs='09,10',max_seqs=1)))
    ix = tmp_path/'index.json'
    write_json(ix,index)
    raw = PairDataset(ix, mode='online',split='train')
    assert len(raw) == 3
    x, Kstar, y = raw[0]
    assert x.shape == (6, HEIGHT, WIDTH)
    assert Kstar.shape == (3,3) and y.shape == (6,)
    assert abs(float(y[2])-.1)<1e-6
    monkey_argv = ['materialize.py','--index',str(ix),'--out-dir',str(tmp_path/'processed'),
                   '--manifest',str(tmp_path/'prepared.json')]
    old = sys.argv
    try:
        sys.argv=monkey_argv
        main()
    finally:
        sys.argv=old
    prepared = PairDataset(tmp_path/'prepared.json', mode='prepared',split='train')
    a,k2,y2=prepared[0]
    assert a.shape == x.shape
    assert torch.allclose(y,y2) and torch.allclose(Kstar,k2)
    assert (a-x).abs().mean().item() < 0.02
    with torch.no_grad():
        # one 2-frame pair through ResNet18
        prediction = PoseResNet18()(a.unsqueeze(0), k2.unsqueeze(0))
    assert prediction.shape==(1,6)


def test_waymo_raw_jpeg_online(tmp_path):
    from camera import camera_metadata
    raw_dir = tmp_path / 'waymo_raw'
    raw_dir.mkdir()
    seq_frames = []
    K = [[800.,0.,640.],[0.,800.,192.],[0.,0.,1.]]
    for i in range(3):
        img = np.full((384,1280,3), 60+i*20, np.uint8)
        path = raw_dir / f'{i:06d}.jpg'
        assert cv2.imwrite(str(path),img)
        T = np.eye(4)
        T[0,3] = .15*i
        seq_frames.append(dict(timestamp_s=.1*i, pose=T.tolist(),raw_path=str(path)))
    seq = dict(name='waymo_fake_0',source='waymo',split='train',K_src=K,
               distortion=[0.,0.,0.,0.,0.],frames=seq_frames)
    manifest = dict(format_version=3,stage='raw',camera=camera_metadata(),sequences=[seq])
    path=tmp_path/'waymo.json'
    write_json(path, manifest)
    ds=PairDataset(path,split='train',mode='online')
    x,Kstar,y=ds[0]
    assert x.shape==(6,HEIGHT,WIDTH)
    assert abs(float(y[0])-.15)<1e-6
    assert torch.allclose(Kstar,torch.tensor(K_VIRTUAL,dtype=torch.float32))


def test_tfrecord_offsets(tmp_path):
    import struct
    from tfrecord_io import each_record, read_record
    path=tmp_path/'fake.tfrecord'
    payloads=[b'abc',b'mockprotobuf']
    with open(path,'wb') as out:
        for p in payloads:
            out.write(struct.pack('<Q',len(p)))
            out.write(b'\x00'*4)
            out.write(p)
            out.write(b'\x00'*4)
    records=list(each_record(path))
    assert [x[1] for x in records]==payloads
    for off,payload in records:
        assert read_record(path,off)==payload
