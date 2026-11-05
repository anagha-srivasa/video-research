"""Raw-only tasks, independent submissions and hash-bound adjudication."""
import hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from . import ROOT,REF,WORK,REPORT

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)
def ident(value):return hashlib.sha256(('track1-reference-v1:'+value).encode()).hexdigest()[:20]

def build():
 if (REF/'tasks.json').exists():
  assert (REF/'protocol.json').exists() and (WORK/'task_map.json').exists();return json.loads((REF/'tasks.json').read_text())
 episodes=pq.read_table(ROOT/'dataset/context_v3/episodes.parquet').to_pylist();frames=pq.read_table(ROOT/'dataset/context_v3/frames.parquet').to_pylist();contacts=pq.read_table(ROOT/'reports/track1_remediation/contact_proposals.parquet').to_pylist();assert len(contacts)==163
 byvideo={v:{r['mask_uid']:r for r in pq.read_table(ROOT/f'work/track1_readiness/inputs.{v}.parquet').to_pylist()} for v in {e['video_id'] for e in episodes}}
 # Source names are cues only, never source masks or identity decisions.
 names={v:{r['mask_uid']:r['entity_name'] for r in pq.read_table(ROOT/f'canonical/v2/spatial_metadata.{v}.parquet',columns=['mask_uid','entity_name']).to_pylist()} for v in byvideo}
 mapping={};public=[];clipframes={}
 for e in episodes:
  cid=ident(e['context_id']);local=[r for r in frames if r['video_id']==e['video_id'] and e['start_us']<=r['timestamp_us']<e['end_us']];clipframes[cid]=[dict(timestamp_us=r['timestamp_us'],video_id=r['video_id'],pts_raw=r['pts_raw'],path=f'dataset/{"context_v3" if r["storage_origin"]=="context" else "partition_v2"}/{r["video_id"]}.rgb.npy',storage_row=r['storage_row'],native_frame_uid=r['native_frame_uid']) for r in local]
  seed=dict(context_id=e['context_id'],video_id=e['video_id'],start_us=e['start_us'],end_us=e['end_us'],entity_track_id=e['entity_track_id'],clip_id=cid)
  taskid=ident('episode:'+e['context_id']);cue=e['entity_name'];public.append(dict(task_id=taskid,clip_id=cid,kind='motion_identity',cue=cue,query_index=0,frame_count=len(local),frame_times_us=[r['timestamp_us']-e['start_us'] for r in local],guide=guide('motion_identity',cue,None)));mapping[taskid]=dict(seed,query_timestamp_us=local[0]['timestamp_us'])
  for c in [c for c in contacts if c['context_id']==e['context_id']]:
   taskid=ident('contact:'+c['contact_uid']);query=min(range(len(local)),key=lambda i:abs(local[i]['timestamp_us']-c['timestamp_us']));hand=names[e['video_id']].get(c['hand_mask_uid'],'hand (confirm which hand)');obj=names[e['video_id']].get(c['target_mask_uid'],'object (target uncertain)');cue=hand+' / '+obj;public.append(dict(task_id=taskid,clip_id=cid,kind='contact',cue=cue,query_index=query,frame_count=len(local),frame_times_us=[r['timestamp_us']-e['start_us'] for r in local],guide=guide('contact',cue,query)));mapping[taskid]=dict(seed,contact_uid=c['contact_uid'],hand_mask_uid=c['hand_mask_uid'],object_mask_uid=c['target_mask_uid'],hand_track_id=byvideo[e['video_id']].get(c['hand_mask_uid'],{}).get('entity_track_id'),object_track_id=byvideo[e['video_id']].get(c['target_mask_uid'],{}).get('entity_track_id'),query_timestamp_us=local[query]['timestamp_us'],source_annotation_timestamp_us=c['timestamp_us'],frame_alignment_delta_us=local[query]['timestamp_us']-c['timestamp_us'])
 public.sort(key=lambda r:(r['kind']!='contact',r['task_id']));write(WORK/'task_map.json',dict(tasks=mapping,clips=clipframes));write(REF/'tasks.json',public)
 protocol=dict(version=1,reviewer_A='user; identity entered when reviewing',reviewer_B='unassigned independent reviewer',adjudicator='unassigned; distinct reviewer identity required',roles=['A','B','adjudicator'],prior_exposure_policy='Record prior candidate exposure per case. Raw-only service blinding does not erase prior familiarity; reviewer B should be independently naive.',blinding='Reviewer service serves raw RGB and the current role annotations only. No candidate outputs, source masks, scores or inferred contact status are served.',source_failure_audit='Separate phase after reference adjudication is hash-locked; source overlays never establish the reference.',required_phases=['pre_contact','first_contact','sustained_contact','release','occluded_contact','high_speed_contact','small_object_contact'],contact_scope='Visible native-image boundaries and projected separation; physical contact is separately judged and may remain unknown. No hidden-boundary or depth inference.',motion_scope='Fixed physical landmark through time; never a moving visible-mask centroid. Manual events are time intervals, not forced exact instants.',sampling='Browse all native frames. Aim for anchors every 40 ms in motion episodes; record invisibility rather than interpolate. Review a pilot before scaling.',source_failure_categories=['none','fragmentation','missing_component','wrong_identity','boundary_error','source_occlusion_ambiguity','unknown'],scientific_acceptance=False,locked_fresh_evaluation=False,track2_started=False,tasks=len(public),contact_tasks=163,motion_identity_tasks=len(episodes))
 write(REF/'protocol.json',protocol)
 for role in ['A','B','adjudicated','source_audit']:(REF/role).mkdir(exist_ok=True)
 write(REPORT/'status.json',dict(reference_ready=False,planned_tasks=len(public),reviewer_A_submitted=0,reviewer_B_submitted=0,adjudicated=0,scientific_acceptance=False,track2_started=False))
 return public

def guide(kind,cue,query):
 common=[f'Locate the intended entity or pair: {cue}. This is an unverified class cue. If several objects fit, record ambiguity instead of guessing.', 'Watch the whole six-second context first, then step through the relevant native frames. Your own marks are allowed; model predictions and source outlines are hidden.', 'Use native-pixel coordinates. Zoom to 100% or 200% for small gaps. Do not trace a boundary hidden by a hand, blur or occlusion.', 'Keep uncertainty: use an interval for an onset/release and mark unknown or unscorable with a reason when evidence is insufficient.']
 extra=[]
 if any(x in cue.lower() for x in ['knife','spoon','fork','spatula']):extra.append('Thin utensil: zoom to 200%. Trace the visible edge near the fingers; do not bridge blur or infer an edge hidden by the hand.')
 if any(x in cue.lower() for x in ['bottle','glass','cup']):extra.append('Curved or transparent object: follow the visible physical edge, not a reflection or highlight. Mark uncertain where transparency prevents a defensible trace.')
 if any(x in cue.lower() for x in ['bag','cloth','food','pepper','dough']):extra.append('Deformable target: distinguish a changing silhouette from physical translation. Mark ambiguous folds/occluded surfaces as uncertain.')
 if kind=='contact':return common+extra+[f'Return to query frame {query}. Click Hand boundary and trace only the visible interaction-side hand outline; then trace the relevant object outline. Use Contact region for short open traces, Full visible outline only for a complete closed contour.', 'Judge the visible projected gap separately from physical contact. A 2D overlap alone does not prove physical touching. Physical contact can be unknown even when projected overlap is visible.', 'Use nearby frames to bracket first contact and release. Mark earliest and latest plausible frames; do not infer contact through an unseen interval.', 'Add a short case note explaining the evidence and any blur, small-object or identity uncertainty. Save a draft while working, then submit only after reviewing your marks.']
 landmark='Choose a visible wrist crease or knuckle that stays the same physical point; avoid fingers that change pose.' if 'hand' in cue.lower() else 'Choose a stable rigid corner, handle junction or other identifiable physical point; avoid a changing silhouette centre.'
 return common+[landmark,'Enter the landmark definition. Mark it at roughly 25 Hz (40 ms) where visible, and more densely around an onset/reversal. When occluded, mark Not visible and leave the point empty; never interpolate human labels automatically.','Bracket motion onset/stop, contact onset/release if visible, occlusion start/end and visibly quiet image-motion intervals separately. Motion onset is not the same event as first contact.','For identity across occlusion, record same entity, different entity or uncertain based on visible evidence before and after. Preserve uncertainty and explain the cue used.']

def validate(payload,task,submitted=False):
 if not isinstance(payload,dict):raise ValueError('Expected an annotation object')
 if submitted and not str(payload.get('reviewer_id','')).strip():raise ValueError('Enter reviewer identity')
 if payload.get('outcome') not in ['usable','uncertain','unscorable']:raise ValueError('Choose usable, uncertain or unscorable')
 if submitted and not str(payload.get('notes','')).strip():raise ValueError('Explain visible evidence, uncertainty or exclusion')
 annotations=payload.get('frames',{})
 if not isinstance(annotations,dict):raise ValueError('Frames must be keyed by frame index')
 for index,value in annotations.items():
  if not str(index).isdigit() or not 0<=int(index)<task['frame_count']:raise ValueError('Frame outside task')
  if not isinstance(value,dict):raise ValueError('Invalid frame annotation')
  for key in ['landmark','hand_boundary','object_boundary']:
   points=value.get(key,[])
   if key=='landmark' and points and (len(points)!=1):raise ValueError('One landmark per frame')
   if not isinstance(points,list) or len(points)>4096:raise ValueError('Invalid point list')
   for point in points:
    if not isinstance(point,list) or len(point)!=2 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in point) or not (0<=point[0]<1920 and 0<=point[1]<1080):raise ValueError('Points must be finite native pixels')
 events=payload.get('events',[])
 if not isinstance(events,list) or len(events)>1000:raise ValueError('Invalid event list')
 for event in events:
  if not isinstance(event,dict):raise ValueError('Invalid event annotation')
  if event.get('kind') not in ['motion_onset','motion_stop','contact_onset','contact_release','occlusion_start','occlusion_end','identity_change','quiet_interval']:raise ValueError('Unknown event type')
  lo=event.get('start_index');hi=event.get('end_index')
  if not isinstance(lo,int) or not isinstance(hi,int) or not 0<=lo<=hi<task['frame_count']:raise ValueError('Invalid event interval')
 if submitted and payload['outcome']=='usable':
  if not str(payload.get('confirmed_entities','')).strip():raise ValueError('Identify the entity or pair you actually reviewed')
  if task['kind']=='contact':
   if any(v=='hidden' for v in payload.get('boundary_visibility',{}).values()):raise ValueError('An edge was marked hidden; keep the outcome uncertain/unscorable or mark a visible edge')
   f=annotations.get(str(task['query_index']),{});minimum=3 if payload.get('boundary_scope')=='full_visible_outline' else 2
   if any(len(f.get(k,[]))<minimum for k in ['hand_boundary','object_boundary']):raise ValueError('Trace both visible boundaries at the query frame, or mark uncertain/unscorable')
   if payload.get('projected_contact') not in ['separated','touching','overlap','unknown'] or payload.get('physical_contact') not in ['yes','no','unknown']:raise ValueError('Record projected and physical contact separately')
  elif not str(payload.get('landmark_definition','')).strip() or sum(bool(f.get('landmark')) and f.get('visibility')=='visible' for f in annotations.values())<7:raise ValueError('Define a fixed landmark and mark at least seven visible frames; sparse episodes may remain unscorable')
 return payload

class Repository:
 def __init__(self,base=REF,private=WORK/'task_map.json'):
  self.base=Path(base);self.tasks={r['task_id']:r for r in json.loads((self.base/'tasks.json').read_text())};self.private=json.loads(Path(private).read_text());self.protocol_hash=digest(self.base/'protocol.json')
 def path(self,role,task_id,draft=False):
  if role not in ['A','B','adjudicator'] or task_id not in self.tasks:raise ValueError('Unknown role or task')
  directory='adjudicated' if role=='adjudicator' else role
  return self.base/directory/(task_id+('.draft.json' if draft else '.json'))
 def own(self,role,task_id):
  for draft in [False,True]:
   p=self.path(role,task_id,draft)
   if p.exists():return json.loads(p.read_text())
  return None
 def save(self,role,task_id,payload,submit=False):
  task=self.tasks[task_id];validate(payload,task,submit);final=self.path(role,task_id)
  if final.exists():raise ValueError('Submitted review is immutable; revisions require a new protocol round')
  value=dict(payload,prior_candidate_exposure=payload.get('prior_candidate_exposure','unsure'),task_id=task_id,role=role,protocol_sha256=self.protocol_hash,submitted=submit,saved_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
  if role=='adjudicator':
   inputs=[]
   for r in ['A','B']:
    p=self.path(r,task_id)
    if not p.exists():raise ValueError('Two submitted independent reviews are required')
    x=json.loads(p.read_text())
    if x.get('protocol_sha256')!=self.protocol_hash or not x.get('submitted'):raise ValueError('Independent review does not match this protocol')
    inputs.append((p,x))
   people=[x['reviewer_id'].strip().casefold() for p,x in inputs]
   if people[0]==people[1]:raise ValueError('Reviewer A and B must be different people')
   if submit and (payload['reviewer_id'].strip().casefold() in people or not payload.get('resolution_notes')):raise ValueError('A distinct adjudicator and resolution notes are required; disagreements must remain recorded')
   value['input_review_hashes']={p.parent.name:digest(p) for p,x in inputs}
   a,b=[x for p,x in inputs];distances=[]
   for index,frame in a.get('frames',{}).items():
    other=b.get('frames',{}).get(index,{})
    if frame.get('landmark') and other.get('landmark'):distances.append(float(np.linalg.norm(np.array(frame['landmark'][0])-other['landmark'][0])))
   value['disagreement']=dict(shared_landmarks=len(distances),mean_landmark_px=float(np.mean(distances)) if distances else None,p95_landmark_px=float(np.quantile(distances,.95)) if distances else None,projected_contact_agrees=a.get('projected_contact')==b.get('projected_contact'),physical_contact_agrees=a.get('physical_contact')==b.get('physical_contact'),identity_agrees=a.get('identity_across_occlusion')==b.get('identity_across_occlusion'),events_agree=a.get('events')==b.get('events'),raw_reviews_retained=True)
  write(final if submit else self.path(role,task_id,True),value);return value
 def accepted(self,task_id):
  path=self.path('adjudicator',task_id)
  if not path.exists():return None
  x=json.loads(path.read_text())
  if x['protocol_sha256']!=self.protocol_hash:raise ValueError('Reference protocol changed')
  for role,h in x['input_review_hashes'].items():
   if digest(self.path(role,task_id))!=h:raise ValueError('Underlying independent review changed')
  return x

def status(repository=None):
 repo=repository or Repository();rows=[]
 for taskid,task in repo.tasks.items():
  a=repo.path('A',taskid).exists();b=repo.path('B',taskid).exists();final=repo.accepted(taskid);rows.append(dict(task_id=taskid,kind=task['kind'],reviewer_A_submitted=a,reviewer_B_submitted=b,adjudicated=final is not None,reference_usable=final is not None and final['outcome']=='usable',outcome=final['outcome'] if final else None))
 write(REPORT/'reference_ledger.json',rows);write(REPORT/'status.json',dict(reference_ready=any(r['reference_usable'] for r in rows),planned_tasks=len(rows),reviewer_A_submitted=sum(r['reviewer_A_submitted'] for r in rows),reviewer_B_submitted=sum(r['reviewer_B_submitted'] for r in rows),adjudicated=sum(r['adjudicated'] for r in rows),usable=sum(r['reference_usable'] for r in rows),scientific_acceptance=False,track2_started=False));return rows

if __name__=='__main__':build();status();print('Blinded tasks prepared; no reference labels generated')
