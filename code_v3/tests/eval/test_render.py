from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest

from cosmos3eval.render import _bev_points, _draw_bev, _encode_h264


def test_bev_points_and_image():
    gt=np.tile(np.eye(4),(61,1,1));pr=gt.copy()
    gt[:,0,3]=np.arange(61)*0.1
    pr[:,0,3]=np.arange(61)*0.11
    p,q=_bev_points(gt,pr,250)
    assert p.shape==(61,2) and q.shape==(61,2)
    assert _draw_bev(p,q,15,250).shape==(250,250,3)

@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='FFmpeg missing')
def test_h264_writes_video(tmp_path):
    frames=(np.full((240,320,3),i,dtype=np.uint8) for i in range(5))
    out=tmp_path/'video.mp4'
    _encode_h264(frames,out,10,320,240,18)
    assert out.exists() and out.stat().st_size>100
    if shutil.which('ffprobe'):
        cmd=['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,pix_fmt',
             '-of','default=noprint_wrappers=1',str(out)]
        result=subprocess.run(cmd,capture_output=True,text=True,check=True)
        assert 'codec_name=h264' in result.stdout
        assert 'pix_fmt=yuv420p' in result.stdout
