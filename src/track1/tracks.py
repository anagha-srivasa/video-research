"""Conservative source-chain identity, ambiguity splits and observation selection."""
from collections import defaultdict
import hashlib
import numpy as np

def uid(*parts):return hashlib.sha256(repr(parts).encode()).hexdigest()[:24]

def bbox_iou(a,b):
    lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:]);inter=np.prod(np.maximum(hi-lo,0))
    return float(inter/max(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter,1e-12))

def assign_tracks(rows,gap_seconds=.35):
    """No noun-only tracking. Chain links require unique names; cross-chain links require shared endpoints."""
    groups=defaultdict(list);manual=[];parent={};reason={}
    for i,r in enumerate(rows):
        if r['interpolation_id'] is None:manual.append(i);continue
        key=(r['subsequence'],r['interpolation_id'],r['entity_name_open'],r['class_id'])
        groups[key].append(i)
    nodes={};endpoints=defaultdict(list)
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def join(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for key,indices in sorted(groups.items(),key=lambda x:repr(x[0])):
        indices.sort(key=lambda i:(rows[i]['timestamp_us'],rows[i]['mask_uid']))
        times=[rows[i]['timestamp_us'] for i in indices]
        # Two same-noun instances cannot be assigned through arbitrary source mask IDs.
        ambiguous=len(times)!=len(set(times))
        segment=-1;previous=None
        for i in indices:
            r=rows[i];t=r['timestamp_us']
            if ambiguous or previous is None or t-previous>gap_seconds*1e6:segment+=1
            node=uid(r['video_id'],key,segment);parent.setdefault(node,node);nodes[i]=node
            reason[i]='ambiguous_duplicate_split' if ambiguous else 'unique_entity_in_source_interpolation'
            previous=t
        if not ambiguous and key[0] is not None:
            for i in [indices[0],indices[-1]]:
                r=rows[i];endpoints[(r['subsequence'],r['entity_name_open'],r['class_id'],r['timestamp_us'])].append(i)
    # Shared physical endpoint, same source subsequence, unique geometric agreement.
    for indices in endpoints.values():
        unique=list(dict.fromkeys(indices))
        if len(unique)==2:
            a,b=unique
            if rows[a]['interpolation_id']!=rows[b]['interpolation_id'] and bbox_iou(rows[a]['bbox'],rows[b]['bbox'])>.6:join(nodes[a],nodes[b])
    at_time=defaultdict(list)
    for i,node in nodes.items():
        r=rows[i];at_time[(r['subsequence'],r['entity_name_open'],r['class_id'],r['timestamp_us'])].append(i)
    for i in manual:
        r=rows[i];candidates=at_time.get((r['subsequence'],r['entity_name_open'],r['class_id'],r['timestamp_us']),[])
        matches={find(nodes[j]) for j in candidates if bbox_iou(r['bbox'],rows[j]['bbox'])>.5}
        if len(matches)==1:nodes[i]=next(iter(matches));reason[i]='manual_shared_endpoint_geometry'
        else:
            node=uid(r['video_id'],'manual',r['mask_uid']);parent[node]=node;nodes[i]=node;reason[i]='manual_identity_unlinked'
    chosen={};rank={'manual_sparse':0,'dense_gt_endpoint':1,'dense_interpolated':2}
    for i,node in nodes.items():
        r=rows[i];r['entity_track_id']=find(node);r['identity_evidence']=reason[i]
        key=(r['entity_track_id'],r['timestamp_us']);priority=(rank[r['source_kind']],r['mask_uid'])
        if key not in chosen or priority<chosen[key][0]:chosen[key]=(priority,i)
    selected={v[1] for v in chosen.values()}
    for i,r in enumerate(rows):r['selected_observation']=i in selected
    return rows
