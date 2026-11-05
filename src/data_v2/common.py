import hashlib, json, pathlib, psutil
import pyarrow as pa
import pyarrow.parquet as pq
from pipeline.common import ROOT, atomic_json, digest
CFG = json.loads((ROOT/'configs/dataset_v2.json').read_text())
OUT = ROOT/'canonical/v2'
FRAMES = ROOT/'dataset/partition_v2'
REPORT = ROOT/'reports/dataset_v2'
WORK = ROOT/'work/dataset_v2'
for p in [OUT, FRAMES, REPORT, WORK]:
    p.mkdir(parents=True, exist_ok=True)
VIDEOS = CFG['train'] + CFG['validation']
def split(v):
    if v not in VIDEOS:
        raise ValueError('Exposed test participant excluded')
    return 'train' if v in CFG['train'] else 'validation'
def save(path, rows):
    table = pa.Table.from_pylist(rows)
    temp = path.with_suffix(path.suffix+'.tmp')
    pq.write_table(table, temp, compression=None)
    temp.replace(path)
def guard():
    proc = psutil.Process()
    rss = proc.memory_info().rss + sum(p.memory_info().rss for p in proc.children(recursive=True) if p.is_running())
    if rss > CFG['ram_limit_gib']*2**30 or psutil.virtual_memory().available < CFG['min_available_gib']*2**30:
        raise MemoryError('Dataset memory guard')
def identity():
    files = list((ROOT/'src/data_v2').glob('*.py')) + [ROOT/'configs/dataset_v2.json']
    return {str(p.relative_to(ROOT)): digest(p) for p in files}
def rank(s):
    return hashlib.sha256((str(CFG['seed'])+s).encode()).hexdigest()
