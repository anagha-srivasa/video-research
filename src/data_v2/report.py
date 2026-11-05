"""Dataset release notes and reproducible task/availability boundaries."""
import html,json
from .common import *
def write():
    labels=json.loads((REPORT/'labels.json').read_text());partition=json.loads((REPORT/'partition.json').read_text());evaluation=json.loads((REPORT/'geometry.json').read_text());verification=json.loads((REPORT/'verification.json').read_text())
    counts={}
    for v in labels['videos']:
        for k,n in v['counts'].items():counts[k]=counts.get(k,0)+n
    metrics=evaluation['metrics'];rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(r[k]))}</td>' for k in ['split','cohort','variant','masks','hole_masks','iou','boundary_fscore_2px'])+'</tr>' for r in metrics)
    doc=f'''<!doctype html><meta charset="utf-8"><title>Dataset v2 development partition</title>
<style>body{{font:16px system-ui;margin:32px auto;max-width:1200px;padding:20px}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #ccc}}table{{border-collapse:collapse}}p{{line-height:1.5}}</style>
<h1>Dataset v2: uncompressed development partition</h1><p>Track 1 remains open. Track 2 remains on hold. Independent scientific acceptance is pending.</p>
<p>{partition['plan']['frames']:,} native RGB frames in {partition['plan']['episodes']:,} selected clips, seven participants. RGB bytes: {partition['plan']['rgb_bytes']:,}; total product estimate: {partition['plan']['total_product_estimate_bytes']:,} / {CFG['budget_bytes']:,} bytes. NPY images and Parquet metadata have no compression.</p>
<p>Geometry observations: {counts['observations']:,}; source masks with holes: {counts['hole_masks']:,}; partial geometry evidence: {counts['partial_evidence']:,}. Partial evidence does not become complete ground truth. Labels are retrospective and have no certified online availability.</p>
<h2>Matched contour reconstruction</h2><table><tr><th>Split</th><th>Cohort</th><th>Variant</th><th>Masks</th><th>Hole masks</th><th>IoU</th><th>BF2</th></tr>{rows}</table>
<p>Matched means use the same manual-mask cohort as historical architecture probes; additional hole examples are a separate cohort. These measure reconstruction from the supplied contour. Numerical release checks and file hashes passed; independent motion/contact/identity references remain absent.</p>
<p><a href="geometry.json">Detailed geometry results</a> · <a href="partition.json">Partition</a> · <a href="labels.json">Label rebuild</a> · <a href="verification.json">Checks</a></p>
<p>The old geometry-dependent appearance, compound, calibration and certificate evidence must be rebuilt with the new schema before promotion. Current corrected spatial pilot fits are not a new full-timeline temporal release.</p>'''
    (REPORT/'dashboard.html').write_text(doc)
    (FRAMES/'README.md').write_text(f'''# Native RGB development partition

{partition['plan']['frames']:,} frames in {partition['plan']['episodes']:,} contiguous selected clips. Every stored frame is native 1920×1080 RGB uint8, without compression or resizing. One `.rgb.npy` shard per video; `frames.parquet` maps original PTS/timestamps/UIDs to storage rows. `episodes.parquet` records physical half-open intervals and selection strata. Duplicate overlap frames are stored once.

The total planned product, including existing source files and canonical/v2 tables, is {partition['plan']['total_product_estimate_bytes']:,} bytes within a 250,000,000,000-byte budget. Source MP4s remain their original publisher bytes. Existing compressed archives were not recompressed. New RGB and Parquet outputs are uncompressed.

Five train and two exploratory validation participants are separate. No fresh locked test set has been created. Source masks are retrospective labels; source interpolations are not causal sensor measurements. Complete and partial geometry evidence remain distinct.

Use `from data_v2.store import Dataset` with `PYTHONPATH=src:work/deps:work/track1_deps`. `get_frame(video_id, timestamp_us)` requires an exact selected native timestamp. `get_mask(video_id, mask_uid)` rejects quarantined masks; `allow_partial=True` explicitly opts into incomplete geometry. `online=True` rejects these labels. Geometry and mask coordinates remain in the recorded annotation grid; use the canonical annotation transform before comparing with native RGB.

Labels: `../../canonical/v2`. Evidence: `../../reports/dataset_v2`. Temporary files and logs: `../../work/dataset_v2`. Reproduce with `python3 -m data_v2.run --stage all` using the above PYTHONPATH. Extraction checkpoints are content-hash verified; differing inputs require a new output version. Partition selection preserves the existing pilot and uses no old test participants.
''')
    files=list(OUT.glob('*.parquet'))+list(FRAMES.glob('*.parquet'))+list(FRAMES.glob('*.rgb.npy'))
    # RGB hashes were just verified; reuse exact verified receipts rather than reread 180 GB again.
    receipts={str(FRAMES/f'{r["video_id"]}.rgb.npy'):json.loads((FRAMES/f'{r["video_id"]}.json').read_text())['sha256'] for r in verification['videos']}
    atomic_json(REPORT/'manifest.json',dict(finalized=False,track2_started=False,code=identity(),files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=receipts[str(p)] if str(p) in receipts else digest(p)) for p in files}))
