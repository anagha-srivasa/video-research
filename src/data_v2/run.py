import argparse, logging, os
# Bound native math threads before numerical imports.
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[name]='1'
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['labels','partition','evaluate','verify','all'],default='all');a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True)
    if a.stage in ['labels','all']:
        from .build import run
        run()
    if a.stage in ['partition','all']:
        from .partition import run
        run()
    if a.stage in ['evaluate','all']:
        from .evaluation import run
        run()
    if a.stage in ['verify','all']:
        from .store import verify
        verify()
        from .report import write
        write()
if __name__=='__main__':main()
