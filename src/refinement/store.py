from pathlib import Path
from collections import OrderedDict
import json
import duckdb
from pipeline.common import ROOT
from track1.artifacts import load_shard
from track1.temporal import collocation

class RefinedDataset:
    def __init__(self,path=None,require_ready=True):
        self.path=Path(path or ROOT/'derived/track1_refined_v2');status=self.path/'status.json'
        if require_ready and (not status.exists() or not json.loads(status.read_text()).get('ready')):raise RuntimeError('Refinement has not passed its freeze gates')
        self.cache=OrderedDict();self.db=duckdb.connect();self.db.execute('SET threads=2')
        for stage in ['spatial','functional']:self.db.read_parquet(str(self.path/stage/'index.parquet')).create_view(stage)
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):self.cache.clear();self.db.close()
    def get_state(self,video_id,entity_track_id,window_end_us):
        query=self.db.execute('SELECT * FROM functional WHERE video_id=? AND entity_track_id=? AND window_end_us=?',[video_id,entity_track_id,window_end_us]);names=[r[0] for r in query.description];rows=query.fetchall()
        if len(rows)!=1:raise KeyError((video_id,entity_track_id,window_end_us))
        meta=dict(zip(names,rows[0]))
        if video_id not in self.cache:
            if len(self.cache)>=2:self.cache.popitem(last=False)
            self.cache[video_id]=load_shard(self.path/'functional'/f'{video_id}.pt')
        return dict(**meta,**{k:v[meta['row_index']].copy() for k,v in self.cache[video_id].items()})
    def get_collocation(self,video_id,entity_track_id,window_end_us):
        state=self.get_state(video_id,entity_track_id,window_end_us);return collocation(state['coefficients'])
