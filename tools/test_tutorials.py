"""Run every tutorial with its normal settings and capture independent results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path, nargs='?', default=Path(__file__).resolve().parent.parent)
parser.add_argument('results', type=Path, nargs='?', help='Log directory (default: ROOT/build/tutorial-tests)')
args = parser.parse_args()
root = args.root.resolve()
results = (args.results or root / 'build/tutorial-tests').resolve()
results.mkdir(parents=True, exist_ok=True)
dist = root / 'tutorials/dist'
env = os.environ.copy()
if sys.platform == 'win32':
    env['PATH'] = str(root / 'hiprt/win') + os.pathsep + env.get('HIP_PATH', '').rstrip('\\/') + '/bin' + os.pathsep + env.get('PATH', '')
    executables = sorted((dist / 'bin/Release').glob('*64.exe'))
else:
    env['LD_LIBRARY_PATH'] = str(root / 'hiprt/linux64') + ':' + env.get('HIP_PATH', '/opt/rocm').rstrip('/') + '/lib:' + env.get('LD_LIBRARY_PATH', '')
    executables = sorted((dist / 'bin/Release').glob('*64'))
if len(executables) != 20:
    raise SystemExit('Expected 20 tutorials, found ' + str(len(executables)))
report = []
for exe in executables:
    previous = {p.name: p.stat().st_mtime_ns for p in dist.glob('*.png')}
    start = time.time()
    with (results / (exe.stem + '.log')).open('w', encoding='utf-8') as log:
        try:
            result = subprocess.run([str(exe)], cwd=str(dist), env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 'timeout'
    images = [{'file': p.name, 'bytes': p.stat().st_size} for p in dist.glob('*.png') if previous.get(p.name) != p.stat().st_mtime_ns]
    minimum_images = 4 if exe.stem.startswith('19_') else 0 if exe.stem.startswith(('00_', '17_')) else 1
    direct_hip_ran = not exe.stem.startswith('17_') or "Executing on '" in (results / (exe.stem + '.log')).read_text(encoding='utf-8', errors='replace')
    passed = code == 0 and len(images) >= minimum_images and direct_hip_ran
    item = {'tutorial': exe.stem, 'passed': passed, 'returncode': code, 'seconds': round(time.time() - start, 2), 'images': images}
    report.append(item)
    print(json.dumps(item), flush=True)
    (results / 'results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
sys.exit(0 if all(item['passed'] for item in report) else 1)
