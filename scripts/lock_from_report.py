"""Freeze a pip resolver report to exact URLs and SHA256 hashes (no installation)."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, unquote
from urllib.request import urlopen


def artifact_hash(info, fetch_missing=False):
    """Accept pip hash representations; optionally fingerprint an exact HTTPS artifact."""
    url = info['url']
    parsed = urlsplit(url)
    allowed = {'files.pythonhosted.org', 'download.pytorch.org', 'download-r2.pytorch.org'}
    if (parsed.scheme != 'https' or parsed.hostname not in allowed
            or parsed.username or parsed.password or any(c.isspace() for c in url)):
        raise ValueError(f'unreviewed artifact URL: {url}')
    archive = info.get('archive_info', {})
    candidates = []
    if 'sha256' in archive.get('hashes', {}):
        candidates.append(archive['hashes']['sha256'])
    legacy = archive.get('hash', '')
    if legacy.startswith('sha256='):
        candidates.append(legacy.split('=', 1)[1])
    if parsed.fragment:
        if not parsed.fragment.startswith('sha256='):
            raise ValueError('unsupported URL fragment')
        candidates.append(parsed.fragment.split('=', 1)[1])
    if any(not re.fullmatch('[0-9a-fA-F]{64}', value) for value in candidates):
        raise ValueError('malformed SHA256')
    candidates = {value.lower() for value in candidates}
    if len(candidates) > 1:
        raise ValueError('conflicting SHA256 values')
    url = parsed._replace(fragment='').geturl()
    if candidates:
        return url, candidates.pop(), False
    if not fetch_missing:
        raise ValueError(f'no SHA256 in report for {url}; use --hash-missing to fingerprint this artifact')
    # This is a downloaded-byte fingerprint, not a publisher-provided checksum.
    digest = hashlib.sha256()
    total = 0
    with urlopen(url, timeout=60) as response:
        final = urlsplit(response.geturl())
        if final.scheme != 'https' or final.hostname not in allowed:
            raise ValueError('unreviewed download redirect')
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > 64 * 1024 * 1024:
                raise ValueError('missing-hash artifact exceeds 64 MiB limit')
            digest.update(chunk)
    if not total:
        raise ValueError('empty artifact')
    return url, digest.hexdigest(), True


def render(report, fetch_missing=False):
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
        url, digest, downloaded = artifact_hash(info, fetch_missing)
        if downloaded:
            lines.append('# SHA256 fingerprinted from downloaded bytes; absent in resolver report.')
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
    p.add_argument('--hash-missing', action='store_true',
                   help='Download exact artifacts missing SHA256 (max 64 MiB each); do not install.')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'output already exists: {args.output}')
    result = render(json.loads(args.report.read_text()), args.hash_missing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(result)
    print(args.output)

if __name__ == '__main__':
    main()
