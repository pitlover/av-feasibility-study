"""Official dataset readers -> virtual MP4 + camera-relative GT in a NEW root."""
from __future__ import annotations
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import cv2

from cosmos3prep.core import frame_runs_kitti_native, frame_runs_10hz, safe_id
from cosmos3prep.geometry import localize, encode_actions, decode_local, validate_pose_array
from .camera import VirtualSpec, optical_from_source, render_virtual
from .temporal import SourcePair, blend_virtual, exact_grid_clips


def _read_image(image):
    if isinstance(image, bytes):
        img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Cannot read {image if not isinstance(image, bytes) else "JPEG bytes"}')
    return img


def _json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def _encode_video(images, path, width, height, fps=10, crf=18):
    exe = shutil.which('ffmpeg')
    if not exe:
        raise RuntimeError('H.264 encoding requires ffmpeg with libx264')
    cmd = [exe, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
           '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s:v', f'{width}x{height}',
           '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-crf', str(crf),
           '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    with tempfile.TemporaryFile() as stderr:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=stderr, stdout=subprocess.DEVNULL)
        try:
            for img in images:
                if img.shape != (height,width,3):
                    raise ValueError('Incorrect virtual frame size')
                proc.stdin.write(img.tobytes())
            proc.stdin.close()
            exitcode = proc.wait()
        except BaseException:
            if proc.poll() is None: proc.kill()
            proc.wait()
            raise
        stderr.seek(0)
        if exitcode or not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f'FFmpeg failed: {stderr.read().decode(errors="replace")}')


def make_clip(frames, *, dataset, sequence, split, first_idx, spec: VirtualSpec,
              out_root, convention='backward_framewise', encoder='local',
              min_valid_fraction=.97, crf=18, overwrite=False, camera_name=None,
              time_mode='native', rgb_interpolation='flow'):
    if len(frames) != 61:
        raise ValueError('Exactly 61 source frames required')
    cid = f'{safe_id(dataset)}_{safe_id(sequence)}_{first_idx:06d}'
    dst = Path(out_root) / dataset / split / cid
    if dst.exists():
        if overwrite:
            shutil.rmtree(dst)
        elif (dst/'metadata.json').is_file() and (dst/'video.mp4').is_file():
            print('[skip exists]', cid)
            return False
        else:
            raise RuntimeError(f'Incomplete output {dst}; use --overwrite if safe')
    dst.mkdir(parents=True, exist_ok=False)
    try:
        ts = np.asarray([f.t_s for f in frames], dtype=np.float64)
        if not np.all(np.diff(ts) > 0):
            raise ValueError('Non-monotonic timestamp')
        world_ego = np.stack([f.world_ego for f in frames])
        T_ego_source = np.stack([f.ego_camera for f in frames])
        validate_pose_array(world_ego)
        validate_pose_array(T_ego_source)
        # Translate Waymo's x-forward camera sensor convention to pinhole optical.
        T_source_virtual = optical_from_source(dataset)
        T_ego_virtual = T_ego_source @ T_source_virtual
        world_virtual = world_ego @ T_ego_virtual
        virtual_local = localize(world_virtual)
        actions, encoder_used = encode_actions(virtual_local, convention, encoder)
        rec = decode_local(actions, convention)
        t_err = np.linalg.norm(virtual_local[:, :3, 3] - rec[:, :3, 3], axis=1).max()
        dR = np.swapaxes(rec[:, :3, :3], -1, -2) @ virtual_local[:, :3, :3]
        cos = np.clip((np.trace(dR, axis1=1, axis2=2)-1)/2.,-1,1)
        r_err = np.rad2deg(np.arccos(cos)).max()
        if t_err > .002 or r_err > .05:
            raise ValueError(f'GT roundtrip fail: {t_err}m, {r_err}deg')

        Kvirtual = spec.K
        # Remap original RGB to virtual intrinsics; native mode NEVER synthesizes frames.
        # Exact (explicit opt-in) may interpolate after optical rectification.
        def virtual_frame(f):
            def render_source(src):
                dist=src.distortion if src.needs_undistort else np.zeros(5)
                return render_virtual(_read_image(src.image),src.K,dist,
                                      Kvirtual,(spec.width,spec.height),
                                      min_valid_fraction=min_valid_fraction)
            pair=f.image
            if isinstance(pair,SourcePair):
                left,valid0=render_source(pair.left)
                if pair.left is pair.right:
                    return left,valid0
                right,valid1=render_source(pair.right)
                result=blend_virtual(left,right,pair.alpha,method=rgb_interpolation)
                return result,min(valid0,valid1)
            dist=f.distortion if f.needs_undistort else np.zeros(5)
            return render_virtual(_read_image(f.image),f.K,dist,Kvirtual,
                                  (spec.width,spec.height),
                                  min_valid_fraction=min_valid_fraction)

        first_im, valid_fraction = virtual_frame(frames[0])
        def images():
            yield first_im
            for f in frames[1:]:
                image,coverage=virtual_frame(f)
                yield image
        _encode_video(images(), dst/'video.mp4', spec.width, spec.height, 10, crf)
        interpolated = [isinstance(f.image,SourcePair) and f.image.left is not f.image.right
                        for f in frames]
        original_brackets=np.array([[f.image.left.t_s,f.image.right.t_s] if isinstance(f.image,SourcePair)
                                   else [f.t_s,f.t_s] for f in frames],dtype=np.float64)
        original_alphas=np.array([f.image.alpha if isinstance(f.image,SourcePair) else 0.
                                  for f in frames],dtype=np.float64)
        np.save(dst/'source_time_brackets_s.npy',original_brackets)
        np.save(dst/'source_interpolation_alpha.npy',original_alphas)
        max_dt_err=float(np.max(np.abs(np.diff(ts)-.1)))
        if time_mode=='exact' and max_dt_err > 1e-5:
            raise ValueError('Exactly sampled 10Hz clip has non-100ms timestamps')
        np.save(dst/'timestamps_s.npy', ts)
        np.save(dst/'poses_world_ego.npy', world_ego)
        np.save(dst/'poses_local_ego.npy', localize(world_ego))
        np.save(dst/'poses_local_virtual_camera.npy', virtual_local)
        np.save(dst/'poses_local_camera.npy', virtual_local)  # compatibility alias
        np.save(dst/'T_ego_virtual_camera.npy', T_ego_virtual)
        np.save(dst/'T_ego_source_camera.npy', T_ego_source)
        np.save(dst/'K_source.npy', np.stack([f.K for f in frames]))
        np.save(dst/'distortion_source.npy', np.stack([f.distortion for f in frames]))
        np.save(dst/'T_ego_camera.npy', T_ego_virtual)  # compatibility alias
        np.save(dst/'K_virtual.npy', Kvirtual)
        np.save(dst/'actions_raw_9d.npy', actions)
        meta = dict(format='cosmos3_virtual_camera_v2', clip_id=cid, dataset=dataset,
                    split=split, sequence=sequence, camera=camera_name or ('image_2' if dataset=='kitti' else 'FRONT' if dataset=='waymo' else 'stereo_front_left'),
                    fps=10, frame_count=61, action_count=60, video_codec='h264',
                    video_size_wh=[spec.width,spec.height], virtual_hfov_deg=spec.hfov_deg,
                    K_virtual=Kvirtual.tolist(), K_video=Kvirtual.tolist(),
                    virtual_camera_definition='optical +X right, +Y down, +Z forward; rotation_only; source_optical_center_preserved',
                    virtual_valid_fraction_first_frame=valid_fraction,
                    pose_frame='virtual_camera', pose_local_file='poses_local_virtual_camera.npy',
                    evaluation_pose_file='poses_local_virtual_camera.npy',
                    action_file='actions_raw_9d.npy', action_normalization='none',
                    action_encoding='translation3 + rot6d_column_based',
                    pose_convention=convention, action_encoder=encoder_used,
                    T_source_virtual_optical=T_source_virtual.tolist(),
                    T_ego_virtual_camera_first=T_ego_virtual[0].tolist(),
                    GT_roundtrip_max_translation_m=float(t_err),
                    GT_roundtrip_max_rotation_deg=float(r_err),
                    time_mode=time_mode,
                    exact_gt_10hz=(time_mode=='exact'),
                    rgb_interpolation=(rgb_interpolation if time_mode=='exact' else 'none'),
                    synthetic_rgb_frames=sum(interpolated),
                    synthetic_rgb_fraction=float(np.mean(interpolated)),
                    timestamp_max_dt_error_s=max_dt_err,
                    source_time_brackets_file='source_time_brackets_s.npy',
                    source_interpolation_alpha_file='source_interpolation_alpha.npy',
                    note_rgb_time='Intermediate images are estimated, not physical exposures' if any(interpolated) else 'Original measured exposures',
                    actual_capture_duration_s=float(ts[-1]-ts[0]),
                    actual_effective_fps=float(60/(ts[-1]-ts[0])),
                    clip_timebase='exact_100ms_with_rgb_temporal_interpolation' if time_mode=='exact' else 'nominal_10fps_native_timestamps',
                    first_timestamp_s=float(ts[0]), last_timestamp_s=float(ts[-1]),
                    video_path=str((dst/'video.mp4').resolve()),
                    source_frame_ids=[f.source_id for f in frames])
        _json(dst/'metadata.json',meta)
        _json(dst/'cosmos_inference.json',dict(name=cid, model_mode='inverse_dynamics',
               domain_name='av', action_chunk_size=60, fps=10, image_size=480,
               view_point='ego_view', prompt='You are an autonomous vehicle planning system.',
               vision_path=str((dst/'video.mp4').resolve()),seed=0))
        return True
    except BaseException:
        shutil.rmtree(dst, ignore_errors=True)
        raise


def rebuild_virtual_index(root):
    root=Path(root)
    index=root/'index.jsonl'
    rows=[]
    for p in sorted(root.glob('*/**/metadata.json')):
        folder=p.parent
        if not (folder/'video.mp4').is_file() or not (folder/'actions_raw_9d.npy').is_file():continue
        m=json.loads(p.read_text())
        rows.append(dict(clip_id=m['clip_id'], dataset=m['dataset'], sequence=m['sequence'], split=m['split'],
                         video=str((folder/'video.mp4').resolve()),meta=str(p.resolve()),
                         action_file=str((folder/'actions_raw_9d.npy').resolve())))
    root.mkdir(parents=True, exist_ok=True)
    tmp=index.with_suffix('.jsonl.tmp')
    tmp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    tmp.replace(index)
    return len(rows)


def process_log(frames, *, dataset, sequence, split, spec, args, remaining):
    if not frames:return 0
    if args.time_mode=='exact':
        candidates_iter=exact_grid_clips(frames,dataset,stride=args.stride,
                                         max_gap_s=args.max_source_gap_ms/1000.)
        print(f'[time] {dataset}/{sequence}: exact 100ms targets, SE3 SLERP, '
              f'RGB={args.rgb_interpolation}; source frames={len(frames)}')
    else:
        if dataset=='kitti' and args.kitti_timing=='native':
            runs=frame_runs_kitti_native(frames)
        else:
            runs=frame_runs_10hz(frames,fps=10,tolerance_s=args.tolerance_ms/1000.)
        indices={id(f):i for i,f in enumerate(frames)}
        def legacy_candidates():
            for run in runs:
                for off in range(0,len(run)-60,args.stride):
                    yield indices[id(run[off])],run[off:off+61]
        candidates_iter=legacy_candidates()
        print(f'[time] {dataset}/{sequence}: native exposures + matching GT at recorded timestamps; nominal MP4 10fps (not physically exact 10Hz)')
    n=0
    for first_idx,chunk in candidates_iter:
        if n>=remaining:return n
        try:
            created=make_clip(chunk,dataset=dataset,sequence=sequence,split=split,
                              first_idx=first_idx,spec=spec,out_root=args.out,
                              convention=args.convention,encoder=args.encoder,
                              min_valid_fraction=args.min_valid_fraction,crf=args.crf,
                              overwrite=args.overwrite,camera_name=args.camera,
                              time_mode=args.time_mode,rgb_interpolation=args.rgb_interpolation)
        except Exception as e:
            if args.strict:raise
            print(f'[clip error] {dataset}/{sequence}/{first_idx}: {e}')
            continue
        if created:
            n+=1
            print(f'  [ok] {dataset}_{safe_id(sequence)}_{first_idx:06d}')
    return n
