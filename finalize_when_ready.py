"""Finalize this finite experiment after all 60 evaluated runs are saved."""
from pathlib import Path
import sys,time,json,subprocess,zipfile,hashlib,platform
from importlib.metadata import version
root=Path(__file__).resolve().parent
out=root/'results';start=time.monotonic()
while len(list((out/'runs').glob('*.json'))) < 60:
    if time.monotonic()-start > 7200:
        raise SystemExit('Timed out waiting for all 60 runs; experiment remains incomplete.')
    time.sleep(20)
versions={p:version(p) for p in ['numpy','scipy','scikit-learn','torch','numba','joblib','pandas','matplotlib']}
versions['python']=platform.python_version()
(out/'package_versions.json').write_text(json.dumps(versions,indent=2))
for script in ['verify_results.py','summarize_results.py']:
    result=subprocess.run([sys.executable,script],cwd=root,capture_output=True,text=True)
    print(result.stdout,flush=True);print(result.stderr,flush=True)
    result.check_returncode()
paths=[p for p in out.rglob('*') if p.is_file()]
bundle=root.parent/'mira-csi-verified-results.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as archive:
    for path in paths:archive.write(path,'results/'+str(path.relative_to(out)))
    for path in root.glob('*.log'):archive.write(path,path.name)
    for path in root.glob('*.json'):archive.write(path,path.name)
    for path in root.glob('*.py'):archive.write(path,path.name)
print('FINAL VERIFIED BUNDLE:',str(bundle),'bytes=',bundle.stat().st_size,
      'sha256=',hashlib.sha256(bundle.read_bytes()).hexdigest(),flush=True)
