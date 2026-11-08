# V2/V4 synchronization

This archive accepts historical V2 virtual camera dataset metadata alongside
new V4 metadata. Stage1 uses KITTI + Waymo, Stage2 uses AV2; evaluation
compares camera-relative GT against 60-step 9D prediction actions.

`cosmos3prep/contracts.py` supplies the canonical interpretation for V2:
`action_frame=cam`, `pose_local_file=poses_local_virtual_camera.npy`,
`format=cosmos3_virtual_camera_v2`. It rejects ambiguities rather than
silently assuming an ego frame. Source dataset and checkpoints are never
modified by validation.

The original Cosmos3 GPU execution is an external dependency. Regression tests
validate local geometry and contracts, not the CUDA runtime.
