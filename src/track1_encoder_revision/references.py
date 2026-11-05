"""Read-only independent-reference ingestion audit; no machine-to-human conversion."""
import json
from pathlib import Path
from track1_reference.protocol import Repository,validate,digest

def audit():
 repo=Repository();rows=[]
 for task_id,task in repo.tasks.items():
  errors=[];accepted=None
  try:accepted=repo.accepted(task_id)
  except (ValueError,KeyError,TypeError) as exc:errors.append(str(exc))
  for role in ['A','B']:
   path=repo.path(role,task_id)
   if path.exists():
    try:
     value=json.loads(path.read_text());validate(value,task,True)
     if value.get('human_reference_eligible') is False or value.get('annotation_kind','').startswith('machine') or value.get('model'):raise ValueError('Machine annotation is not an independent human submission')
    except (ValueError,KeyError,TypeError) as exc:errors.append(role+': '+str(exc))
  if accepted is not None:
   try:
    validate(accepted,task,True)
    if not accepted.get('submitted') or accepted.get('role')!='adjudicator' or set(accepted.get('input_review_hashes',{}))!={'A','B'}:raise ValueError('Incomplete adjudication provenance')
    people=[str(accepted.get('reviewer_id','')).strip().casefold()]
    for role in ['A','B']:
     value=json.loads(repo.path(role,task_id).read_text());people.append(str(value.get('reviewer_id','')).strip().casefold())
    if not all(people) or len(set(people))!=3:raise ValueError('Three distinct reviewers required')
    if accepted.get('human_reference_eligible') is False or accepted.get('model'):raise ValueError('Machine output cannot be adjudicated human ground truth')
   except (ValueError,KeyError,TypeError,OSError) as exc:errors.append('adjudication: '+str(exc))
  usable=bool(accepted and accepted['outcome']=='usable' and not errors)
  rows.append(dict(task_id=task_id,kind=task['kind'],accepted=usable,adjudicated=accepted is not None,errors=errors,development_only=True))
 return dict(tasks=rows,accepted_usable=sum(r['accepted'] for r in rows),locked_fresh_labels_opened=False,scientific_acceptance=False,contact_weighted_training_allowed=False,final_freeze_allowed=False)

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 from pipeline.common import ROOT,atomic_json
 out=args.output.resolve()
 if not out.is_relative_to(ROOT):raise ValueError('Output must stay inside project root')
 atomic_json(out,audit())
if __name__=='__main__':main()
