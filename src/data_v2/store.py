"""Bounded native RGB reader with explicit retrospective-label access."""
import collections,json
import numpy as np
import pyarrow.parquet as pq
from .common import *

class Dataset:
    def __init__(self):
        self.rows=pq.read_table(FRAMES/'frames.parquet').to_pylist()
        self.lookup={(r['video_id'],r['timestamp_us']):r for r in self.rows};self.cache=collections.OrderedDict()
    def get_frame(self,video_id,timestamp_us):
        r=self.lookup[(video_id,int(timestamp_us))]
        if video_id not in self.cache:
            if len(self.cache)>=2:self.cache.popitem(last=False)[1]._mmap.close()
            self.cache[video_id]=np.load(FRAMES/f'{video_id}.rgb.npy',mmap_mode='r')
        self.cache.move_to_end(video_id)
        return np.array(self.cache[video_id][r['storage_row']],copy=True)
    def get_mask(self,video_id,mask_uid,*,allow_partial=False,online=False):
        if online:raise ValueError('These retrospective source labels have no verified online availability')
        import pyarrow.dataset as ds
        rows=ds.dataset(OUT/f'masks.{video_id}.parquet').to_table(filter=ds.field('mask_uid')==mask_uid).to_pylist()
        if len(rows)!=1:raise KeyError(mask_uid)
        r=rows[0];segments=json.loads(r['segments_raw'])
        if not r['geometry_complete']:
            if not allow_partial or not r['partial_geometry_valid']:raise ValueError('Incomplete geometry; explicit partial access required')
            segments=[segments[i] for i in r['retained_segment_indices']]
        from pipeline.geometry import rasterize
        return rasterize(segments,r['annotation_width'],r['annotation_height']),dict(complete=r['geometry_complete'],source_kind=r['source_kind'],available_at_us=None)
    def close(self):
        for a in self.cache.values():a._mmap.close()
        self.cache.clear()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def verify():
    result=[]
    with Dataset() as data:
        for v in VIDEOS:
            marker=json.loads((FRAMES/f'{v}.json').read_text());a=np.load(FRAMES/f'{v}.rgb.npy',mmap_mode='r')
            rows=[r for r in data.rows if r['video_id']==v]
            assert len(rows)==len(a)==marker['frames'];assert a.dtype==np.uint8 and a.shape[1:]==(1080,1920,3)
            assert marker['sha256']==digest(FRAMES/f'{v}.rgb.npy')
            for i in sorted(set([0,len(rows)//2,len(rows)-1])):
                rgb=data.get_frame(v,rows[i]['timestamp_us']);assert rgb.shape==(1080,1920,3) and np.isfinite(rgb).all()
            a._mmap.close()
            holes=0;partial=0;n=0
            for batch in pq.ParquetFile(OUT/f'masks.{v}.parquet').iter_batches(batch_size=256,columns=['mask_uid','hole_count','contours_json','geometry_complete','partial_geometry_valid','online_observation_allowed']):
                for r in batch.to_pylist():
                    assert sum(x['role']=='hole' for x in json.loads(r['contours_json']))==(r['hole_count'] or 0)
                    assert not r['online_observation_allowed'];holes+=bool(r['hole_count']);partial+=r['partial_geometry_valid'];n+=1
            for meta in pq.ParquetFile(OUT/f'masks.{v}.parquet').metadata.row_group(0).column(0),:
                assert meta.compression=='UNCOMPRESSED'
            result.append(dict(video_id=v,frames=len(rows),observations=n,hole_masks=holes,partial_evidence=partial))
    atomic_json(REPORT/'verification.json',dict(videos=result,numerical_checks_passed=True,full_rgb_sha256_checked=True,
        online_labels_available=False,scientific_acceptance=False,finalized=False,code=identity()))
    atomic_json(OUT/'status.json',dict(partition_rgb_ready=True,retrospective_geometry_ready=True,partial_geometry_requires_explicit_opt_in=True,
        training_release_replaced=False,temporal_and_appearance_rebuild_pending=True,scientific_acceptance=False,finalized=False,track2_started=False))
