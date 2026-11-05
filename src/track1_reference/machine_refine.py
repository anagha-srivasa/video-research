"""Two-pass local visual screening; approximate attention crops are not boundaries."""
import argparse,hashlib,io,json,time
from PIL import Image
from . import ROOT,REF,WORK
from . import machine as m
from .machine_report import build

EXTRA=''' Review ONLY the hand-to-cued-object interface, never the object's contact with a tray, board or another object. Here edge visibility means the required interaction-side border: visible only when that border is clearly exposed, partial if any needed portion is covered, hidden if covered, unknown if unresolved. A visible outer silhouette does not establish a visible gripping interface. Never call an edge visible while explaining that fingers hide it. Do not infer the object's contents from its cue. Also return interaction_box as [left,top,right,bottom] in normalized 0..1000 coordinates around the VISIBLE target hand/object interaction, or null if it cannot be located. This is only an approximate attention crop, never a boundary annotation. Explain every partial/hidden visibility in uncertainty_reasons. The cue may be incorrect: say so if unsupported.'''
def crop(raw,box):
 if not isinstance(box,list) or len(box)!=4 or any(type(x) not in [float,int] for x in box):return None
 x0,y0,x1,y1=box
 if not (0<=x0<x1<=1000 and 0<=y0<y1<=1000):return None
 im=Image.open(io.BytesIO(raw));w,h=im.size;pad=40
 bounds=(max(0,int((x0-pad)*w/1000)),max(0,int((y0-pad)*h/1000)),min(w,int((x1+pad)*w/1000)),min(h,int((y1+pad)*h/1000)))
 if bounds[2]-bounds[0]<48 or bounds[3]-bounds[1]<48:return None
 out=io.BytesIO();im.crop(bounds).save(out,format='PNG',compress_level=0);return out.getvalue(),bounds

def run(model,limit=0):
 import psutil
 tasks=json.loads((REF/'tasks.json').read_text());private=json.loads((WORK/'task_map.json').read_text());out=ROOT/'reports/track1_vlm'/f'{model}-refined';out.mkdir(exist_ok=True)
 m.PROMPT+=EXTRA
 # Verify each original source once; never use source annotations as visual evidence.
 import pyarrow.parquet as pq
 from .protocol import digest
 videos=sorted({private['tasks'][t['task_id']]['video_id'] for t in tasks[:limit or None]})
 for video in videos:
  expected=pq.read_table(ROOT/f'canonical/v1/videos.{video}.parquet',columns=['sha256']).to_pylist()[0]['sha256']
  print(f'Checking original source {video}',flush=True)
  if digest(ROOT/f'dataset/videos/{video}.MP4')!=expected:raise RuntimeError('Source checksum mismatch: '+video)
 for n,t in enumerate(tasks[:limit or None]):
  path=out/(t['task_id']+'.json')
  if path.exists():continue
  if psutil.virtual_memory().available<6*1024**3:raise RuntimeError('Available memory below 6 GiB; safely stopped; rerun to resume')
  try:
   fs=private['clips'][t['clip_id']];q=t['query_index'] if t['kind']=='contact' else len(fs)//2
   ix=[max(0,q-12),q,min(len(fs)-1,q+12)] if t['kind']=='contact' else [0,q,len(fs)-1]
   images=[(f'{label}: native frame {i}, timestamp {fs[i]["timestamp_us"]}us',m.decode(fs[i])) for label,i in zip(['BEFORE','CENTER','AFTER'],ix)]
   prompt=f'Unverified cue: {t["cue"]}. Task type: {t["kind"]}. Review the CENTER image. Sparse images do not establish event times, velocity or identity through occlusion.'
   with (ROOT/'work/track1_vlm/resources.jsonl').open('a') as log:log.write(json.dumps(dict(time=time.time(),task_id=t['task_id'],available_gib=psutil.virtual_memory().available/2**30,process_rss_gib=psutil.Process().memory_info().rss/2**30))+ '\n')
   start=time.time();print(f'{n+1}/{len(tasks)} {model} {t["cue"]}',flush=True);first=m.request(model,images,prompt);r1=m.parse(first['choices'][0]['message'].get('content',''));cut=crop(images[1][1],r1.get('interaction_box'));second=None;bounds=None;a=r1
   if cut:
    zoom,bounds=cut
    second=m.request(model,[images[1],('CENTER close-up; original pixels, no enhancement; the crop is model-proposed and may be wrong',zoom)],prompt+' Recheck the visible interface using the full image and this attention crop. If the crop misses the intended entities, state that and abstain. Do not infer contact from an occluded grip.');a=m.parse(second['choices'][0]['message'].get('content',''))
   disagreements=[k for k in ['entities_found','hand_edge_visibility','object_edge_visibility','projected_relation','physical_contact'] if r1.get(k)!=a.get(k)]
   result=dict(task_id=t['task_id'],cue=t['cue'],kind=t['kind'],model=model+'-refined',base_model=model,annotation_kind='machine_only_not_independent_reference',human_reference_eligible=False,requires_human_confirmation=True,boundaries_provided=False,event_timing_provided=False,annotation=a,schema_valid=all(r['choices'][0].get('finish_reason')=='stop' for r in [first,second] if r),seconds=time.time()-start,frame_indices=ix,image_sha256=[hashlib.sha256(b).hexdigest() for _,b in images],attention_crop_native=bounds,pass_disagreements=disagreements,raw_response=first,closeup_response=second,prompt_sha256=hashlib.sha256((m.PROMPT+prompt).encode()).hexdigest(),source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest())
   tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(path);build();print(f'  saved {result["seconds"]:.1f}s; crop={bool(cut)}; disagreements={disagreements}',flush=True)
  except Exception as error:
   failure=dict(task_id=t['task_id'],cue=t['cue'],model=model+'-refined',annotation=None,schema_valid=False,seconds=0,error=str(error),human_reference_eligible=False)
   path.write_text(json.dumps(failure,indent=2));build();print('  failed closed: '+str(error),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args();run(a.model,a.limit)
