"""Real-codec check of exact-PTS selection for the fresh-source reader."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import av
import numpy as np
from pipeline.common import ROOT


def test_fresh_decoder_selects_exact_frames_and_rejects_missing_pts():
    with tempfile.TemporaryDirectory(dir=ROOT / 'work/track1_fmr') as directory:
        folder = Path(directory)
        path = folder / 'synthetic.mp4'
        with av.open(str(path), 'w') as output:
            stream = output.add_stream('libx264', rate=30)
            stream.width, stream.height, stream.pix_fmt = 1920, 1080, 'yuv420p'
            stream.codec_context.thread_count = 1
            for level in [32, 96, 192]:
                frame = av.VideoFrame.from_ndarray(np.full((1080, 1920, 3), level, np.uint8), format='rgb24')
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        with av.open(str(path)) as source:
            pts = sorted(p.pts for p in source.demux(video=0) if p.pts is not None)
        request = folder / 'request.json'
        request.write_text(json.dumps(dict(source_path=str(path.relative_to(ROOT)), frames=[dict(pts_raw=p) for p in [pts[0], pts[2]]])))
        command = [sys.executable, '-m', 'track1_fmr.source_stream', str(request)]
        result = subprocess.run(command, capture_output=True, check=True, cwd=ROOT)
        data, offset = result.stdout, 0
        for index, level in enumerate([32, 192]):
            selected, size = struct.unpack_from('<II', data, offset)
            assert selected == index and size == 1920 * 1080 * 3
            offset += 8
            rgb = np.frombuffer(data, np.uint8, count=size, offset=offset)
            assert abs(float(rgb.mean()) - level) < 3
            offset += size
        assert offset == len(data)
        request.write_text(json.dumps(dict(source_path=str(path.relative_to(ROOT)), frames=[dict(pts_raw=pts[-1] + 100_000)])))
        failed = subprocess.run(command, capture_output=True, cwd=ROOT)
        assert failed.returncode != 0 and b'Exact PTS missing' in failed.stderr
