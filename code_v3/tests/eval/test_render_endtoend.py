import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cosmos3eval.geometry import encode_framewise
from cosmos3eval.render import _encode_h264, render_clip


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='FFmpeg missing')
def test_render_full_clip(tmp_path):
    cid='kitti_09_000000'
    clip=tmp_path/'clip'
    clip.mkdir()
    poses=np.tile(np.eye(4),(61,1,1))
    poses[:,0,3]=np.arange(61)*.2
    actions=encode_framewise(poses)
    assert actions.shape==(60,9)
    np.save(clip/'poses_local_ego.npy', poses)
    np.save(clip/'actions_raw_9d.npy', actions)
    np.save(clip/'timestamps_s.npy', np.arange(61)*.1)
    meta={'clip_id':cid,'dataset':'kitti','split':'val',
          'pose_convention':'backward_framewise','action_normalization':'none',
          'action_encoding':'translation3 + rot6d_column_based',
          'fps':10,'frame_count':61,'action_count':60,'pose_frame':'velodyne'}
    (clip/'metadata.json').write_text(json.dumps(meta))
    frames=(np.full((240,426,3),i,dtype=np.uint8) for i in range(61))
    _encode_h264(frames,clip/'video.mp4',10,426,240,23)
    out=tmp_path/'prediction'
    (out/cid).mkdir(parents=True)
    (out/cid/'sample_outputs.json').write_text(json.dumps({'outputs':[{'content':{'action':actions.tolist()},'files':[]}]}))
    row={'clip_id':cid,'dataset':'kitti','split':'val','sequence':'09',
         'video':str(clip/'video.mp4'),'meta':str(clip/'metadata.json'),
         'action_file':str(clip/'actions_raw_9d.npy')}
    rendered=tmp_path/'rendered.mp4'
    render_clip(row, out, rendered, fps=10)
    assert rendered.exists() and rendered.stat().st_size>1000
    if shutil.which('ffprobe'):
        args=['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,nb_frames',
              '-of','default=noprint_wrappers=1',str(rendered)]
        info=subprocess.run(args,text=True,capture_output=True,check=True).stdout
        assert 'codec_name=h264' in info
        assert 'nb_frames=61' in info
