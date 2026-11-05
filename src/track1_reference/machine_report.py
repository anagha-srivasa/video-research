"""Readable, separate machine-review ledger. No human annotation mutations."""
import json,html
from pathlib import Path
from . import ROOT

def build():
 out=ROOT/'reports/track1_vlm';rows=[]
 for p in sorted(out.glob('track1-*/*.json')):
  r=json.loads(p.read_text());a=r.get('annotation') or {};flags=[]
  if not r['schema_valid']:flags.append('Invalid or truncated response')
  if any(a.get(k) in ['partial','hidden','unknown'] for k in ['hand_edge_visibility','object_edge_visibility']):flags.append('Boundary visibility is incomplete; no precise boundary reference')
  if flags and not a.get('uncertainty_reasons'):flags.append('Model omitted its uncertainty explanation')
  if a.get('physical_contact')!='unknown':flags.append('Physical-contact claim requires independent verification')
  rows.append(dict(task_id=r['task_id'],cue=r['cue'],model=r['model'],seconds=r['seconds'],annotation=a,flags=flags,schema_valid=r['schema_valid']))
 payload=json.dumps(rows).replace('<','\\u003c');counts={m:sum(r['model']==m for r in rows) for m in {r['model'] for r in rows}}
 (out/'summary.json').write_text(json.dumps(dict(counts=counts,machine_only=True,independent_accuracy_measured=False,rows=rows),indent=2))
 page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Track 1 · Local AI reviews</title><style>body{max-width:1100px;margin:32px auto;padding:0 20px;background:#10161d;color:#edf3f7;font:16px/1.55 system-ui}article{padding:20px;border:1px solid #435367;border-radius:10px;margin:16px 0}small{color:#bdccd6}.warning{background:#323027;padding:16px}button,select{padding:10px;background:#223440;color:white}summary{cursor:pointer}li{margin:6px 0}</style><h1>Local AI review results</h1><p>The model has already described each case below. You do not need to trace anything to read its review.</p><p class="warning">These are machine suggestions, not independent reference labels. Boundary coordinates, contact-event timing and identity across occlusion have not been established. Physical-contact claims require verification. Human drafts are untouched.</p><label>Show model <select id="model"><option value="">All models</option></select></label><p id="count"></p><main id="cases"></main><script>const rows=PAYLOAD;const $=id=>document.getElementById(id);for(const m of [...new Set(rows.map(r=>r.model))]){const o=document.createElement('option');o.value=o.textContent=m;$('model').append(o)}function draw(){$('cases').replaceChildren();const shown=rows.filter(r=>!$('model').value||r.model===$('model').value);$('count').textContent=shown.length+' completed model/case reviews';for(const r of shown){const a=document.createElement('article'),h=document.createElement('h2');h.textContent=r.cue;a.append(h);const sub=document.createElement('small');sub.textContent=r.model+' · '+r.seconds.toFixed(1)+' seconds · '+r.task_id;a.append(sub);for(const [label,text] of [['What it saw',r.annotation.description],['Visible evidence',r.annotation.evidence],['Projected relation',r.annotation.projected_relation],['Uncertainty reported by model',(r.annotation.uncertainty_reasons||[]).join('; ')||'None stated']]){const p=document.createElement('p');p.textContent=label+': '+(text||'Unavailable');a.append(p)}const ul=document.createElement('ul');for(const flag of r.flags){const li=document.createElement('li');li.textContent=flag;ul.append(li)}a.append(ul);$('cases').append(a)}}$('model').onchange=draw;draw();</script></html>'''.replace('PAYLOAD',payload)
 (out/'dashboard.html').write_text(page)
 return counts
if __name__=='__main__':print(build())
