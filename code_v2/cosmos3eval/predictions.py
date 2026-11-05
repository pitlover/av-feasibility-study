"""Strictly parse real action predictions. Never substitute GT for missing model output."""
import json
from pathlib import Path
import numpy as np
from .geometry import check_actions


def _get_action(payload):
    if isinstance(payload, list) and len(payload)==1 and isinstance(payload[0],dict):
        return _get_action(payload[0])
    if not isinstance(payload, dict):
        raise ValueError('Inference JSON must be an object')
    if payload.get('status') not in (None,'success'):
        raise ValueError(f'Inference did not succeed: {payload.get("status")}: {payload.get("message")}')
    # Accommodates common action serialization versions, but refuses ambiguous shapes.
    if payload.get('action') is not None:
        return payload['action']
    if isinstance(payload.get('outputs'),dict) and payload['outputs'].get('action') is not None:
        return payload['outputs']['action']
    if isinstance(payload.get('outputs'),list) and len(payload['outputs'])==1:
        return _get_action(payload['outputs'][0])
    if isinstance(payload.get('samples'),list) and len(payload['samples'])==1:
        return _get_action(payload['samples'][0])
    raise ValueError('Action not found. JSON keys='+str(list(payload))+
                     '; inspect sample_outputs.json and adapt the parser to actual model schema')


def load_pred(pred_dir, clip_id, allow_padded=False):
    base=Path(pred_dir)
    candidates=(base/clip_id/'sample_outputs.json',base/f'{clip_id}.npy',base/f'{clip_id}.json')
    found=next((p for p in candidates if p.is_file()),None)
    if found is None:
        raise FileNotFoundError(f'{clip_id}: no sample_outputs.json or .npy under {base}')
    if found.suffix=='.npy':
        a=np.load(found,allow_pickle=False)
    else:
        payload=json.loads(found.read_text(encoding='utf8'))
        a=np.asarray(_get_action(payload),dtype=np.float64)
    if a.ndim==3 and a.shape[0]==1: a=a[0]
    if a.shape==(60,64):
        if not allow_padded:
            raise ValueError(f'{clip_id}: 64D action; pass --allow-padded-output if first 9D are confirmed raw AV')
        if not np.allclose(a[:,9:],0,atol=1e-5):
            raise ValueError(f'{clip_id}: 64D action has nonzero padding; refusing automatic truncation')
        a=a[:,:9]
    return check_actions(a),found
