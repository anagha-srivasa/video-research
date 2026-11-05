"""Sequential hardware decoder process; deliberately never imports OpenCV."""
import sys,json
from pathlib import Path
import av
from av.codec.hwaccel import HWAccel

def main():
    request=json.loads(Path(sys.argv[1]).read_text());wanted={int(k):v for k,v in request['pts'].items()};seen=set()
    with av.open(request['path'],hwaccel=HWAccel('videotoolbox',allow_software_fallback=False)) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=4
        for frame in container.decode(stream):
            if frame.pts not in wanted:continue
            rgb=frame.reformat(width=request['width'],height=request['height'],format='rgb24').to_ndarray()
            payload=rgb.tobytes();header=dict(timestamp_us=wanted[frame.pts],shape=list(rgb.shape),bytes=len(payload))
            sys.stdout.buffer.write(json.dumps(header).encode()+b'\n');sys.stdout.buffer.write(payload);sys.stdout.buffer.flush();seen.add(frame.pts)
            if len(seen)==len(wanted):break
    if seen!=set(wanted):raise RuntimeError('Native indexed PTS were not all decoded')

if __name__=='__main__':main()
