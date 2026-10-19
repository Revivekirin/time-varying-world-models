"""Read-only host inventory. No installation or simulation."""
import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
report = {'os': platform.platform(), 'architecture': platform.machine(), 'python': sys.version,
          'executable': sys.executable, 'packages': {}}
for pkg in ['torch','gymnasium','mani_skill','sapien','numpy','torchrl','tensordict','hydra-core','omegaconf']:
    try: report['packages'][pkg] = md.version(pkg)
    except md.PackageNotFoundError: report['packages'][pkg] = None
for command in [['nvidia-smi'], ['ldconfig','-p'], ['vulkaninfo','--summary']]:
    if shutil.which(command[0]):
        try:
            p = subprocess.run(command, capture_output=True, text=True, timeout=20)
            report[command[0]] = {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
        except subprocess.TimeoutExpired: report[command[0]] = 'TIMEOUT'
    else: report[command[0]] = 'unavailable'
try:
    import torch
    report['torch'] = {'version':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available()}
    if torch.cuda.is_available(): report['torch']['gpu'] = torch.cuda.get_device_name(0)
except ImportError: pass
(out/'inventory.json').write_text(json.dumps(report,indent=2)+'\n')
print(out/'inventory.json')
