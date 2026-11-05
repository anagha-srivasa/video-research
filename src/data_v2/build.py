"""Stream corrected geometry and explicit partial-evidence observations."""
import collections, json, time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pipeline.geometry import geometry
from .geometry import normalize, VERSION
from .common import *

def video(v):
    split(v); started=time.monotonic(); counts=collections.Counter(); path=OUT/f'masks.{v}.parquet'
    temp=path.with_suffix('.parquet.tmp'); writer=None
    try:
        for batch in pq.ParquetFile(ROOT/f'canonical/v1/masks.{v}.parquet').iter_batches(batch_size=256):
            guard(); rows=[]
            for source in batch.to_pylist():
                row=dict(source); complete=row['geometry_status']!='quarantined'; partial=None; rejected=[]; accepted=[]
                if not complete:
                    for i, segment in enumerate(json.loads(row['segments_raw'])):
                        candidate=geometry([segment],row['annotation_width'],row['annotation_height'])
                        (accepted if candidate['geometry_status']!='quarantined' else rejected).append(i)
                    if accepted:
                        segments=json.loads(row['segments_raw']); partial=geometry([segments[i] for i in accepted],row['annotation_width'],row['annotation_height'])
                        if partial['geometry_status']=='quarantined': partial=None
                rings=normalize(row['contours_json']); partial_rings=normalize(partial['contours_json']) if partial else []
                row.update(contours_json=json.dumps(rings,separators=(',',':')), geometry_schema=VERSION,
                           tuning_split=split(v), geometry_complete=complete,
                           partial_geometry_valid=bool(partial), partial_contours_json=json.dumps(partial_rings,separators=(',',':')),
                           retained_segment_indices=accepted, rejected_segment_indices=rejected,
                           available_at_us=None, online_observation_allowed=False,
                           availability_status='retrospective_label_availability_unknown',
                           dependency_group=f'{v}:{row["interpolation_id"]}' if row['interpolation_id'] is not None else row['mask_uid'],
                           position_semantics='visible_mask_centroid_not_material_anchor',
                           identity_status='provisional_source_scope_not_verified', independent_reference=False)
                counts['observations']+=1;counts['complete']+=complete;counts['partial_evidence']+=bool(partial)
                counts['holes']+=sum(r['role']=='hole' for r in rings);counts['hole_masks']+=any(r['role']=='hole' for r in rings)
                rows.append(row)
            # Explicit types prevent null-only early batches from freezing the wrong schema.
            table=pa.Table.from_pylist(rows)
            schema=pa.schema(list(batch.schema)+[f for f in table.schema if f.name not in batch.schema.names])
            for name,typ in [('available_at_us',pa.int64()),('retained_segment_indices',pa.list_(pa.int64())),('rejected_segment_indices',pa.list_(pa.int64()))]:
                idx=schema.get_field_index(name);schema=schema.set(idx,pa.field(name,typ))
            table=table.cast(schema)
            if writer is None: writer=pq.ParquetWriter(temp,schema,compression=None)
            writer.write_table(table)
        writer.close();writer=None;temp.replace(path)
    finally:
        if writer: writer.close()
    # Correct downstream metadata without altering frozen tensors or claiming they encode holes.
    old=ROOT/f'derived/track1_refined_v2/spatial/{v}.parquet'
    import duckdb
    with duckdb.connect() as db:
        db.execute("SET memory_limit='1GB'");db.execute('SET threads=1')
        table=db.execute('''SELECT s.* EXCLUDE(hole_count),m.hole_count,
            m.component_count AS source_component_count,m.geometry_schema,
            false AS stored_shape_encodes_holes
            FROM read_parquet(?) s JOIN read_parquet(?) m USING(mask_uid)''',[str(old),str(path)]).fetch_arrow_table()
    pq.write_table(table,OUT/f'spatial_metadata.{v}.parquet',compression=None)
    result=dict(video_id=v,split=split(v),counts=dict(counts),seconds=time.monotonic()-started,
                source_sha256=digest(ROOT/f'canonical/v1/masks.{v}.parquet'),output_sha256=digest(path),code=identity())
    atomic_json(REPORT/f'labels.{v}.json',result);return result

def run():
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(max_workers=CFG['workers'],mp_context=multiprocessing.get_context('spawn')) as ex:
        results=list(ex.map(video,VIDEOS))
    atomic_json(REPORT/'labels.json',dict(videos=results,complete_geometry_and_partial_evidence_separate=True,
                source_labels_are_retrospective=True,ingestion_ready=False,scientific_acceptance=False,code=identity()))
