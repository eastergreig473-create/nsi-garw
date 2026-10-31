"""Check the release file hashes without third-party packages."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
count = 0
for line in (ROOT / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
    expected, name = line.split('  ', 1)
    actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit('FAIL: ' + name)
    count += 1
print(f'PASS: {count} release files verified')
