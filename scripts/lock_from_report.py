"""Freeze a pip resolver report to exact URLs and SHA256 hashes (no installation)."""
import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, unquote


def render(report):
    if report['environment']['python_version'] != '3.11' or report['environment']['platform_system'] != 'Linux':
        raise ValueError('expected reviewed Linux Python 3.11 environment')
    expected = {'torch': '2.7.1+cu118', 'numpy': '1.24.4', 'gymnasium': '0.29.1',
                'tensordict': '0.8.3', 'torchrl': '0.8.1', 'hydra-core': '1.3.2',
                'omegaconf': '2.3.0', 'sapien': '3.0.3', 'mani-skill': '3.0.1'}
    seen = {}
    lines = ['# Linux x86_64 / Python 3.11; exact resolved artifacts.',
             '# Editable ManiSkill is excluded; verify environment/sources.lock.json separately.',
             '# Source archive build dependencies are not locked by this artifact lock.']
    for item in report['install']:
        metadata = item['metadata']
        name = re.sub(r'[-_.]+', '-', metadata['name']).lower()
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', name) or name in seen:
            raise ValueError('invalid or duplicate package name')
        seen[name] = metadata['version']
        info = item['download_info']
        url = info['url']
        if name == 'mani-skill':
            if not (info.get('dir_info', {}).get('editable') and urlsplit(url).scheme == 'file'
                    and Path(unquote(urlsplit(url).path)).resolve() == (Path.cwd()/'external/ManiSkill').resolve()):
                raise ValueError('ManiSkill must be the local pinned editable source')
            continue
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or parsed.hostname not in {'files.pythonhosted.org', 'download.pytorch.org', 'download-r2.pytorch.org'}:
            raise ValueError(f'unreviewed artifact origin: {url}')
        digest = info.get('archive_info', {}).get('hashes', {}).get('sha256', '')
        if not re.fullmatch('[0-9a-f]{64}', digest) or any(c.isspace() for c in url) or parsed.fragment:
            raise ValueError(f'missing hash or invalid URL: {name}')
        lines.append(f'# {name}=={metadata["version"]}')
        lines.append(f'{name} @ {url} --hash=sha256:{digest}')
    for name, version in expected.items():
        if seen.get(name) != version:
            raise ValueError(f'expected {name}=={version}; found {seen.get(name)}')
    return '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('report', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    result = render(json.loads(args.report.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(result)
    print(args.output)

if __name__ == '__main__':
    main()
