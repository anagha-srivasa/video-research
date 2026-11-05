"""Memory-mapped model-facing Track 1 access, never full-dataset tensor loading."""
from pathlib import Path
from collections import OrderedDict
import json
import duckdb
import numpy as np
from pipeline.common import ROOT
from .artifacts import load_shard
from .temporal import evaluate,collocation

class Track1Dataset:
    def __init__(self,path=None,require_ready=True):
        self.path=Path(path or ROOT/'derived');self.cache=OrderedDict();self.db=duckdb.connect();self.db.execute('SET threads=2')
        status=self.path/'track1_status.json'
        if require_ready and (not status.exists() or not json.loads(status.read_text()).get('ready')):raise RuntimeError('Track 1 freeze has not passed its gates')
        for stage in ['spatial','functional']:
            self.db.read_parquet(str(self.path/f'track1_{stage}_v1/index.parquet')).create_view(stage)
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):self.cache.clear();self.db.close()
    def _shard(self,stage,video):
        key=(stage,video)
        if key not in self.cache:
            if len(self.cache)>=2:self.cache.popitem(last=False)
            self.cache[key]=load_shard(self.path/f'track1_{stage}_v1/{video}.pt')
        return self.cache[key]
    def get_state(self,video_id,entity_track_id,window_end_us):
        cur=self.db.execute('SELECT * FROM functional WHERE video_id=? AND entity_track_id=? AND window_end_us=?',[video_id,entity_track_id,window_end_us])
        names=[x[0] for x in cur.description];rows=cur.fetchall()
        if len(rows)!=1:raise KeyError((video_id,entity_track_id,window_end_us))
        result=dict(zip(names,rows[0]));shard=self._shard('functional',video_id);idx=result['row_index']
        result.update({k:v[idx].copy() for k,v in shard.items()})
        return result
    def get_collocation(self,video_id,entity_track_id,window_end_us):
        state=self.get_state(video_id,entity_track_id,window_end_us)
        return collocation(state['coefficients'],state['window_seconds'])
