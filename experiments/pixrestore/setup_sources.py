"""Fetch immutable upstream revisions without vendoring code in this project."""
from pathlib import Path
import json,subprocess
ROOT=Path(__file__).resolve().parent
SOURCES={
    'PixRestore':('https://github.com/csslc/PixRestore.git','909ca06e614cf72e218d52b4fa41077c603d872b'),
    'dinov2':('https://github.com/facebookresearch/dinov2.git','7764ea0f912e53c92e82eb78a2a1631e92725fc8'),
}
def main():
    records={}
    for name,(url,revision) in SOURCES.items():
        dest=ROOT/'external'/name
        if dest.exists():
            actual=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
            status=subprocess.check_output(['git','-C',str(dest),'status','--porcelain'],text=True)
            # Running upstream Python creates untracked bytecode cache folders;
            # these do not alter its source and should not break repeated setup.
            dirty=[line for line in status.splitlines() if not (line.startswith('?? ') and '__pycache__' in line[3:].replace('\\','/').split('/'))]
            if actual!=revision or dirty:raise RuntimeError(f'{name} exists with a different revision or local changes. Preserve it and choose a separate checkout.')
        else:
            dest.mkdir(parents=True)
            subprocess.run(['git','init',str(dest)],check=True)
            subprocess.run(['git','-C',str(dest),'remote','add','origin',url],check=True)
            subprocess.run(['git','-C',str(dest),'fetch','--depth','1','origin',revision],check=True)
            subprocess.run(['git','-C',str(dest),'checkout','--detach','FETCH_HEAD'],check=True)
        records[name]={'url':url,'commit':revision,'path':dest.relative_to(ROOT).as_posix()}
    (ROOT/'external'/'source_revisions.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('All upstream checkouts match their fixed revisions.')
if __name__=='__main__':main()
