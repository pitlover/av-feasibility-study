"""Parse genuine NVIDIA sample_outputs.json actions; no missing-output oracle fallback."""
import json
from pathlib import Path
import numpy as np
from .geometry import check_actions


def _get_action(obj):
    if isinstance(obj,list) and len(obj)==1:return _get_action(obj[0])
    if not isinstance(obj,dict):raise ValueError('Prediction JSON must be an object')
    if obj.get('status') not in (None,'success'):
        raise ValueError(f'Inference failed: {obj.get("status")}')
    if obj.get('action') is not None:return obj['action']
    if isinstance(obj.get('content'),dict) and obj['content'].get('action') is not None:
        return obj['content']['action']
    if isinstance(obj.get('outputs'),dict) and obj['outputs'].get('action') is not None:
        return obj['outputs']['action']
    for k in ('outputs','samples'):
        if isinstance(obj.get(k),list) and len(obj[k])==1:return _get_action(obj[k][0])
    raise ValueError('Action not found in model output')


def load_pred(pred_dir,clip_id,allow_padded=False):
    root=Path(pred_dir)
    paths=(root/clip_id/'sample_outputs.json',root/(clip_id+'.npy'),root/(clip_id+'.json'))
    found=next((p for p in paths if p.is_file()),None)
    if found is None:raise FileNotFoundError(f'{clip_id}: prediction file absent under {root}')
    a=np.load(found,allow_pickle=False) if found.suffix=='.npy' else np.asarray(_get_action(json.loads(found.read_text())),np.float64)
    if a.ndim==3 and a.shape[0]==1:a=a[0]
    if a.shape==(60,64):
        if not allow_padded or not np.allclose(a[:,9:],0,atol=1e-5):
            raise ValueError('Padded 64D actions require explicit verified zero padding')
        a=a[:,:9]
    return check_actions(a),found
