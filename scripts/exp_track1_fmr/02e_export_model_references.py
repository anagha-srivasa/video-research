"""Export complete model-reviewed labels with provenance, never as human adjudication."""
import argparse
import json
import importlib.util
from pathlib import Path
from pipeline.common import ROOT, atomic_json, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',default='reports/track1_fmr/stage2_vlm_review_v2_support')
    parser.add_argument('--name',default='stage2_qwen38_v2_support');args=parser.parse_args()
    if not args.name.replace('_','').isalnum():raise ValueError('Invalid name')
    parser_path=Path(__file__).with_name('02b_review_and_calibrate.py')
    spec=importlib.util.spec_from_file_location('fmr_review_validation',parser_path);validation=importlib.util.module_from_spec(spec);spec.loader.exec_module(validation)
    report=ROOT/args.review;manifest=json.loads((report/'manifest.json').read_text());rows=[];hashes={}
    for task in manifest['tasks']:
        path=report/'responses'/f"{task['episode_id']}_{task['timestamp_us']}.json"
        if not path.exists():raise ValueError('Review incomplete')
        response=json.loads(path.read_text())
        if response['task']!=task or response['manifest_sha256']!=digest(report/'manifest.json'):raise ValueError('Review task/manifest mismatch')
        if digest(ROOT/task['image'])!=task['image_sha256']:raise ValueError('Image changed')
        if response.get('human_adjudicated') is not False:raise ValueError('Model review must preserve human_adjudicated=false')
        label=response['label']
        if label is not None:
            content=''.join(item['content'] for item in response['raw']['output'] if item['type']=='message')
            parsed=validation.parse_labels(content,response['request_settings']['batch_size'])
            if parsed[response['request_settings']['batch_index']]!=label:raise ValueError('Normalized label differs from raw response')
        hashes[str(path.relative_to(ROOT))]=digest(path)
        rows.append(dict(episode_id=task['episode_id'],video_id=task['video_id'],timestamp_us=task['timestamp_us'],
            target_id=task['target_id'],x=None if label is None else label.get('x'),y=None if label is None else label.get('y'),
            bbox=None if label is None else label.get('bbox'),visibility=None if label is None else label['visible'],confidence=None if label is None else label['confidence'],
            reviewer=response['model'],reviewer_kind='vlm',human_adjudicated=False,
            schema_valid=label is not None,response_sha256=digest(path)))
    destination=ROOT/'references/track1/model_reviewed'/args.name
    destination.mkdir(parents=True,exist_ok=False)
    atomic_json(destination/'labels.json',dict(rows=rows,reference_kind='local_vlm_silver',human_adjudicated=False,
        target_definition='Visible foreground hand/manipulator including attached visible forearm and worn glove; model-estimated mask centroid and tight bbox',
        warning='VLM confidence is not verified accuracy. Foreground support alignment is requested, not independently established; no human adjudication.'))
    atomic_json(destination/'manifest.json',dict(labels_sha256=digest(destination/'labels.json'),
        review_manifest_sha256=digest(report/'manifest.json'),response_hashes=hashes,reviewed=len(rows),
        valid=sum(r['schema_valid'] for r in rows),human_adjudicated=False,
        source='User-authorized LM Studio Qwen3.8 review',independent_human_qualification=False))
    atomic_json(report/'verification.json',dict(passed=True,reviewed=len(rows),valid=sum(r['schema_valid'] for r in rows),
        model_response_values_verified=True,manifest_and_image_hashes_verified=True,export_labels_sha256=digest(destination/'labels.json'),human_adjudicated=False))
    print(destination,flush=True)


if __name__=='__main__':main()
