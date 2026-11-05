"""Local raw-only machine reviews. Never writes human reference roles."""
import argparse,base64,hashlib,io,json,subprocess,time,urllib.request
from pathlib import Path
from PIL import Image
from . import ROOT,REF,WORK
OUT=ROOT/'reports/track1_vlm'
PROMPT='''You are a cautious visual evidence reviewer, NOT an annotator of ground truth. Images are original 1920x1080 RGB, no masks. A class cue may be wrong. Never infer an invisible border, physical touching from 2D overlap, or identity through occlusion. Report unknown when evidence is insufficient. Do not fabricate precise boundary coordinates. Return only a JSON object with keys: entities_found (boolean), description (string), object_visible (boolean), hand_edge_visibility (visible/partial/hidden/unknown), object_edge_visibility (visible/partial/hidden/unknown), projected_relation (gap/meeting/overlap/unknown), physical_contact (visible_evidence/visible_no_contact/unknown), uncertainty_reasons (array of strings), recommended_action (human_check/usable_description/unscorable), evidence (string). The CENTER image is the frame being reviewed. Neighbor images supply context only. Do not claim an onset or release time from these sparse images.'''
def request(model,images,prompt):
 content=[dict(type='text',text=prompt)]
 for label,raw in images:
  content.extend([dict(type='text',text=label),dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(raw).decode()))])
 body=dict(model=model,messages=[dict(role='system',content=PROMPT),dict(role='user',content=content)],temperature=0,max_tokens=4096 if "qwen" in model else 1400,stream=False)
 req=urllib.request.Request('http://127.0.0.1:1234/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=900) as r:return json.load(r)
def decode(f):
 cmd=['python3','-m','track1_reference.native_frame',f['video_id'],str(f['pts_raw'])]
 return subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=ROOT).stdout
def parse(text):
 s=text.strip();start=s.find('{');end=s.rfind('}');value=json.loads(s[start:end+1]);
 enums={'hand_edge_visibility':['visible','partial','hidden','unknown'],'object_edge_visibility':['visible','partial','hidden','unknown'],'projected_relation':['gap','meeting','overlap','unknown'],'physical_contact':['visible_evidence','visible_no_contact','unknown'],'recommended_action':['human_check','usable_description','unscorable']}
 for k,vs in enums.items():
  if value.get(k) not in vs:raise ValueError('Invalid '+k)
 for k in ['entities_found','object_visible']:
  if type(value.get(k)) is not bool:raise ValueError('Invalid '+k)
 for k in ['description','evidence']:
  if not isinstance(value.get(k),str):raise ValueError('Missing '+k)
 if not isinstance(value.get('uncertainty_reasons'),list):raise ValueError('Invalid uncertainty reasons')
 return value
def run(model,limit,ids=None):
 OUT.mkdir(exist_ok=True);tasks=json.loads((REF/'tasks.json').read_text());private=json.loads((WORK/'task_map.json').read_text());selected=[t for t in tasks if t['kind']=='contact' and (not ids or t['task_id'] in ids)]
 if limit:selected=selected[:limit]
 for n,t in enumerate(selected):
  path=OUT/model.replace('/','_')/(t['task_id']+'.json');path.parent.mkdir(exist_ok=True)
  if path.exists():continue
  import psutil
  if psutil.virtual_memory().available<6*1024**3:raise RuntimeError('Less than 6 GiB available: stopping before inference')
  fs=private['clips'][t['clip_id']];q=t['query_index'];indices=[max(0,q-12),q,min(len(fs)-1,q+12)];images=[];hashes=[]
  for label,i in zip(['BEFORE','CENTER','AFTER'],indices):
   raw=decode(fs[i]);images.append((f'{label}: native frame {i}, time {fs[i]["timestamp_us"]} microseconds',raw));hashes.append(hashlib.sha256(raw).hexdigest())
  started=time.time();print(f'{n+1}/{len(selected)} {model} {t["task_id"]} {t["cue"]}',flush=True)
  prompt=f'Unverified entity cue: {t["cue"]}. Review the CENTER frame. Describe only what the pixels establish.'
  result=request(model,images,prompt);choice=result['choices'][0];text=choice['message'].get('content','');valid=True;error=None
  try:annotation=parse(text)
  except (ValueError,TypeError) as e:annotation=None;valid=False;error=str(e)
  record=dict(task_id=t['task_id'],cue=t['cue'],model=model,annotation_kind='machine_only_not_independent_reference',human_reference_eligible=False,frame_indices=indices,image_sha256=hashes,prompt_sha256=hashlib.sha256((PROMPT+prompt).encode()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.time()-started,finish_reason=choice.get('finish_reason'),usage=result.get('usage'),raw_response=result,schema_valid=valid,error=error,annotation=annotation,requires_human_confirmation=True,boundaries_provided=False,event_timing_provided=False)
  if choice.get('finish_reason')!='stop':record['schema_valid']=False;record['error']='Incomplete generation'
  temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2));temp.replace(path);print(f'  saved {record["seconds"]:.1f}s valid={record["schema_valid"]}',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--limit',type=int,default=4);p.add_argument('--ids',nargs='*');a=p.parse_args();run(a.model,a.limit,a.ids)
