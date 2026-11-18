"""Isolated exact-PTS native RGB pipe. Never imports OpenCV or Torch."""
import json,struct,sys
from pathlib import Path
import av
ROOT=Path(__file__).resolve().parents[2]

def main():
 path=Path(sys.argv[1]).resolve()
 if not path.is_relative_to(ROOT):raise ValueError('Request outside project root')
 req=json.loads(path.read_text());video=req['video_id']
 if not video.startswith('P') or not video.replace('_','').isalnum():raise ValueError('Invalid video')
 wanted={int(r['pts_raw']):i for i,r in enumerate(req['frames'])};seen=set()
 with av.open(str(ROOT/f'dataset/videos/{video}.MP4')) as container:
  stream=container.streams.video[0];stream.codec_context.thread_count=2;container.seek(min(wanted),stream=stream,backward=True)
  for frame in container.decode(stream):
   if frame.pts in wanted:
    if (frame.width,frame.height)!=(1920,1080):raise ValueError('Unexpected native frame dimensions')
    raw=frame.to_ndarray(format='rgb24').tobytes();sys.stdout.buffer.write(struct.pack('<II',wanted[frame.pts],len(raw)));sys.stdout.buffer.write(raw);seen.add(frame.pts)
   if len(seen)==len(wanted):break
 if seen!=set(wanted):raise ValueError('Exact source PTS unavailable; no nearby frame substituted')
if __name__=='__main__':main()
