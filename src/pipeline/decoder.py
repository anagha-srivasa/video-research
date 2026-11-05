"""Isolated PyAV worker: never imports OpenCV's conflicting macOS FFmpeg dylibs."""
import json
import sys
import av

def main():
    for line in sys.stdin.buffer:
        try:
            request=json.loads(line)
            with av.open(request['path']) as c:
                stream=c.streams.video[0];stream.codec_context.thread_count=2
                c.seek(request['pts'],stream=stream,backward=True)
                rgb=None
                for frame in c.decode(stream):
                    if frame.pts==request['pts']:rgb=frame.to_ndarray(format='rgb24');break
                    if frame.pts is not None and frame.pts>request['pts']:break
                if rgb is None:raise RuntimeError('Indexed presentation timestamp not decoded')
                payload=rgb.tobytes()
                header=dict(shape=list(rgb.shape),bytes=len(payload))
        except Exception as e:header=dict(error=str(e),bytes=0);payload=b''
        sys.stdout.buffer.write(json.dumps(header).encode()+b'\n');sys.stdout.buffer.write(payload);sys.stdout.buffer.flush()

if __name__=='__main__':main()
