"""Fetch exact commits; refuse to modify existing mismatched or dirty checkouts."""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
for name, pin in json.loads((root/'environment/sources.lock.json').read_text()).items():
    path = root/'external'/name
    if not path.exists():
        subprocess.run(['git', 'clone', '--no-checkout', pin['url'], str(path)], check=True)
        subprocess.run(['git', '-C', str(path), 'checkout', '--detach', pin['commit']], check=True)
    sha = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if sha != pin['commit'] or dirty:
        raise SystemExit(f'{name}: source drift; inspect checkout manually')
    print(name, sha)
