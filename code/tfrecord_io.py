"""Read uncompressed TFRecord records by offset, without TensorFlow runtime.

Requires Waymo dataset_pb2 to *decode* records (and only in data indexing /
on-the-fly reading). CRC is not verified; use trusted, intact local TFRecords.
"""
import struct


def each_record(path):
    with open(path, 'rb') as f:
        while True:
            pos = f.tell()
            header = f.read(12)
            if not header:
                break
            if len(header) != 12:
                raise IOError(f"Truncated TFRecord header: {path} at {pos}")
            length = struct.unpack('<Q', header[:8])[0]
            if length > 512 * 1024 * 1024:
                raise IOError(f"Unreasonable TFRecord length at {pos}: {length}")
            payload = f.read(length)
            crc = f.read(4)
            if len(payload) != length or len(crc) != 4:
                raise IOError(f"Truncated record: {path} at {pos}")
            yield pos, payload


def read_record(path, offset):
    with open(path, 'rb') as f:
        f.seek(int(offset))
        header = f.read(12)
        if len(header) != 12:
            raise IOError(f"Bad TFRecord header at {path}:{offset}")
        length = struct.unpack('<Q', header[:8])[0]
        if length > 512 * 1024 * 1024:
            raise IOError(f"Unreasonable TFRecord length at {path}:{offset}")
        payload = f.read(length)
        if len(payload) != length or len(f.read(4)) != 4:
            raise IOError(f"Truncated TFRecord at {path}:{offset}")
        return payload


def decode_waymo_frame(payload):
    try:
        from waymo_open_dataset import dataset_pb2
    except ImportError as exc:
        raise RuntimeError("Waymo TFRecord decoding requires waymo_open_dataset.dataset_pb2. "
                           "Use the 'waymo-prep' environment or index with --raw-jpeg-cache.") from exc
    frame = dataset_pb2.Frame()
    frame.ParseFromString(payload)
    return frame


def get_front(frame):
    from waymo_open_dataset import dataset_pb2
    name = dataset_pb2.CameraName.FRONT
    img = next((im for im in frame.images if im.name == name), None)
    cam = next((c for c in frame.context.camera_calibrations if c.name == name), None)
    if img is None or cam is None:
        raise ValueError('FRONT image or calibration missing')
    return img, cam
