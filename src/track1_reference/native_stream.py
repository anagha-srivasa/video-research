"""Exact-PTS grayscale stream in an isolated PyAV process; no OpenCV import."""
import json,struct,sys
import av
from . import ROOT,WORK

def frames(clip):
 mapping=json.loads((WORK/'task_map.json').read_text());fs=mapping['clips'][clip];video=fs[0]['video_id']
 if not video.startswith('P') or not video.replace('_','').isalnum():raise ValueError('Invalid video')
 with av.open(str(ROOT/f'dataset/videos/{video}.MP4')) as container:
  stream=container.streams.video[0];stream.codec_context.thread_count=2;container.seek(fs[0]['pts_raw'],stream=stream,backward=True);i=0
  for frame in container.decode(stream):
   if frame.pts is None or frame.pts<fs[i]['pts_raw']:continue
   if frame.pts!=fs[i]['pts_raw']:raise ValueError(f'Missing exact PTS at clip {clip} frame {i}')
   if (frame.width,frame.height)!=(1920,1080):raise ValueError('Unexpected native dimensions')
   yield i,frame.to_ndarray(format='gray')
   i+=1
   if i==len(fs):return
 raise ValueError('Incomplete native clip')
if __name__=='__main__':
 for idx,gray in frames(sys.argv[1]):
  raw=gray.tobytes();sys.stdout.buffer.write(struct.pack('<II',idx,len(raw)));sys.stdout.buffer.write(raw)
