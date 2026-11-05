"""Isolated software decode of one exact source PTS; PNG stays in memory."""
import argparse,io,sys
from pathlib import Path
import av
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
def decode(video,pts):
 if not video.replace('_','').isalnum() or not video.startswith('P'):raise ValueError('Invalid video identifier')
 with av.open(str(ROOT/f'dataset/videos/{video}.MP4')) as container:
  stream=container.streams.video[0];stream.codec_context.thread_count=2;container.seek(pts,stream=stream,backward=True)
  for frame in container.decode(stream):
   if frame.pts==pts:
    if (frame.width,frame.height)!=(1920,1080):raise ValueError('Unexpected native dimensions')
    out=io.BytesIO();Image.fromarray(frame.to_ndarray(format='rgb24')).save(out,format='PNG',compress_level=0);return out.getvalue()
   if frame.pts is not None and frame.pts>pts:break
 raise ValueError('Exact native PTS unavailable; no approximate frame substituted')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('video');parser.add_argument('pts',type=int);args=parser.parse_args();sys.stdout.buffer.write(decode(args.video,args.pts))
