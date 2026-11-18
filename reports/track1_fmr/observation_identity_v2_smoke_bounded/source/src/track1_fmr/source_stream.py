"""Isolated exact-PTS RGB decoder for project-local newly acquired sources."""
import json
import struct
import sys
from pathlib import Path

import av

ROOT = Path(__file__).resolve().parents[2]


def main():
    request_path = Path(sys.argv[1]).resolve()
    if not request_path.is_relative_to(ROOT):
        raise ValueError('Request outside project')
    request = json.loads(request_path.read_text())
    source = (ROOT / request['source_path']).resolve()
    if not source.is_relative_to(ROOT / 'work/track1_fmr') or not source.is_file():
        raise ValueError('Fresh source must reside in project work/track1_fmr')
    wanted = {int(f['pts_raw']): i for i, f in enumerate(request['frames'])}
    if not wanted or len(wanted) != len(request['frames']):
        raise ValueError('Unique exact source PTS required')
    seen = set()
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        container.seek(min(wanted), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts in wanted:
                if (frame.width, frame.height) != (1920, 1080):
                    raise ValueError('Unexpected native frame dimensions')
                raw = frame.to_ndarray(format='rgb24').tobytes()
                sys.stdout.buffer.write(struct.pack('<II', wanted[frame.pts], len(raw)))
                sys.stdout.buffer.write(raw)
                seen.add(frame.pts)
            if seen == set(wanted):
                break
    if seen != set(wanted):
        raise ValueError('Exact PTS missing; nearby frames are never substituted')


if __name__ == '__main__':
    main()
