import json
import numpy as np
import pytest
from track1_reference.protocol import Repository,validate
from track1_reference.events import MotionEvidence,event_stream,interval_scores
from track1_reference.evaluate import interpolate,boundary_quality

def repository(tmp_path):
 task=dict(task_id='case',kind='contact',frame_count=3,query_index=1)
 (tmp_path/'tasks.json').write_text(json.dumps([task]));(tmp_path/'protocol.json').write_text('{}');(tmp_path/'private.json').write_text('{"tasks":{},"clips":{}}');return Repository(tmp_path,tmp_path/'private.json')

def annotation(name):return dict(reviewer_id=name,outcome='uncertain',notes='Fixture only: uncertainty preserved',frames={},events=[])

def test_adjudication_requires_distinct_people_and_preserves_disagreement(tmp_path):
 repo=repository(tmp_path);repo.save('A','case',annotation('Alice'),True);repo.save('B','case',annotation('Bob'),True)
 with pytest.raises(ValueError,match='distinct'):repo.save('adjudicator','case',dict(annotation('Alice'),resolution_notes='reviewed'),True)
 value=repo.save('adjudicator','case',dict(annotation('Charlie'),resolution_notes='Uncertainty remains unresolved'),True);assert len(value['input_review_hashes'])==2 and repo.accepted('case')['outcome']=='uncertain'
 with pytest.raises(ValueError,match='immutable'):repo.save('A','case',annotation('Alice'),True)
 p=repo.path('B','case');p.write_text(p.read_text()+' ')
 with pytest.raises(ValueError,match='changed'):repo.accepted('case')

def test_one_person_cannot_supply_both_independent_reviews(tmp_path):
 repo=repository(tmp_path);repo.save('A','case',annotation('Reviewer'),True);repo.save('B','case',annotation(' reviewer '),True)
 with pytest.raises(ValueError,match='different people'):repo.save('adjudicator','case',dict(annotation('Third'),resolution_notes='test'),True)

def test_native_coordinates_and_event_intervals_are_validated():
 task=dict(kind='contact',frame_count=3,query_index=1)
 bad=dict(annotation('Test'),frames={'1':dict(hand_boundary=[[1920,1],[1,1]])})
 with pytest.raises(ValueError,match='native pixels'):validate(bad,task,True)
 bad=dict(annotation('Test'),events=[dict(kind='motion_onset',start_index=2,end_index=1)])
 with pytest.raises(ValueError,match='interval'):validate(bad,task,True)


def test_event_channel_keeps_between_update_evidence_without_future_inputs():
 t=np.arange(0,1.02,.02);p=np.zeros((len(t),2));active=(t>.5)&(t<.68);p[active,0]=40*np.sin((t[active]-.5)/.18*np.pi)**2;var=np.ones_like(p);cutoffs=np.array([.25,.5,.75]);a,_=event_stream(t,p,var,cutoffs);changed=p.copy();changed[t>.75]+=10000;b,_=event_stream(t,changed,var,cutoffs);assert a==b;assert any(e['evidence_time']<e['decision_time'] for e in a['causal_evidence'])
 detector=MotionEvidence();point=np.zeros(2);detector.update(0.,point);point+=1000;np.testing.assert_array_equal(detector.samples[0][1],[0.,0.])
 with pytest.raises(ValueError,match='Future'):detector.update(.4,[1,2],.3)


def test_quiet_alarm_rate_uses_union_minutes_and_reference_gaps_are_not_filled():
 score=interval_scores([dict(decision_time=.8)],[],[(0,1),(.5,1.5)]);assert score['quiet_seconds']==1.5 and score['false_alarms_per_quiet_minute']==40
 assert interpolate(np.array([0.,.2]),np.zeros((2,2)),.1) is None
 score,p,r=boundary_quality([np.array([[100.,100.],[110,100],[110,110]])],[[0,0],[0,10]],'contact_region');assert score['bf2']==0 and len(p)==0


def test_original_video_ranges_do_not_transcode_or_misaddress_source_bytes():
 from track1_reference.review import byte_range
 assert byte_range(None,100)==(0,99,200)
 assert byte_range('bytes=10-19',100)==(10,19,206)
 assert byte_range('bytes=90-',100)==(90,99,206)
 assert byte_range('bytes=-12',100)==(88,99,206)
 assert byte_range('bytes=90-999',100)==(90,99,206)
 for invalid in ['bytes=100-','bytes=20-10','bytes=0-1,4-5','bytes=-0']:
  with pytest.raises(ValueError):byte_range(invalid,100)
