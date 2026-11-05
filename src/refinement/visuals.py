from pathlib import Path
import numpy as np,pyarrow.parquet as pq,cv2
from PIL import Image,ImageDraw
from pipeline import Dataset
from track1.artifacts import load_shard
from refinement.build import directory
from refinement.protocol import config,folder
from refinement.kernels import periodic_design
from pipeline.common import ROOT

def render_visuals():
    phi=periodic_design(np.arange(256)/256)
    for v in config()['train']+config()['validation']+config()['test']:
     new=load_shard(directory('spatial')/f'{v}.pt');old=load_shard(ROOT/f'derived/track1_spatial_v1/{v}.pt');lookup={r['mask_uid']:r for r in pq.read_table(directory('spatial')/f'{v}.parquet').to_pylist()};cases=[r for r in pq.read_table(folder('reports')/'spatial_geometry.parquet').to_pylist() if r['video_id']==v and r['method']=='refined'][:6];sheet=Image.new('RGB',(1440,660),(24,24,24));draw=ImageDraw.Draw(sheet)
     with Dataset() as ds:
      for n,c in enumerate(cases):
       r=lookup[c['mask_uid']];target=ds.get_mask(r['mask_uid']);h,w=target.shape;im=np.zeros((h,w,3),np.uint8);im[target>0]=(100,100,100);ys,xs=np.nonzero(target);curves=[]
       for state,col in [(old,(255,90,70)),(new,(50,255,230))]:
        curve=np.rint(phi@state['controls'][r['row_index']]*[w/r['width'],h/r['height']]).astype(np.int32);cv2.polylines(im,[curve],True,col,2);curves.append(curve)
       xy=np.vstack(curves);x0=max(0,min(xs.min(),xy[:,0].min())-10);x1=min(w,max(xs.max(),xy[:,0].max())+11);y0=max(0,min(ys.min(),xy[:,1].min())-10);y1=min(h,max(ys.max(),xy[:,1].max())+11);panel=Image.fromarray(im[y0:y1,x0:x1]);panel.thumbnail((470,265));x=n%3*480;top=n//3*330;sheet.paste(panel,(x+(480-panel.width)//2,top+(270-panel.height)//2));draw.text((x+8,top+275),f'{v} | {r["entity_name"][:32]} | IoU={c["iou"]:.3f}',fill='white');draw.text((x+8,top+294),f'{r["timestamp_us"]/1e6:.3f}s  components={r["component_count"]} holes={r["hole_count"]}',fill='white');draw.text((x+8,top+313),'Gray source / red baseline / cyan refined | crop',fill='white')
     sheet.save(folder('reports')/f'geometry.{v}.jpg',quality=94)
    ps=sorted(folder('reports').glob('geometry.*.jpg'));sheet=Image.new('RGB',(1440,880),(20,20,20))
    for i,p in enumerate(ps):
     im=Image.open(p);im.thumbnail((480,220));sheet.paste(im,(i%3*480,i//3*220))
    sheet.save(folder('reports')/'visual_overview.jpg',quality=94)
