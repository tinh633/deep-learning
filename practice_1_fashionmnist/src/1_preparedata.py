import gzip
import hashlib
import urllib.request

from config import DATA_FOLDER, FILES, SOURCE

folder = DATA_FOLDER
folder.mkdir(parents=True, exist_ok=True)
for name, expected in FILES.items():
    target = folder / name
    if not target.exists() or hashlib.md5(target.read_bytes()).hexdigest() != expected:
        print(f'Downloading {name}', flush=True)
        request = urllib.request.Request(SOURCE + name, headers={'User-Agent': 'FashionMNIST-Practice-1'})
        with urllib.request.urlopen(request, timeout=120) as response:
            content = response.read()
        if hashlib.md5(content).hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {name}')
        temporary = target.with_suffix('.part')
        temporary.write_bytes(content)
        temporary.replace(target)
    print(f'MD5 verified: {name}', flush=True)
    # Preserve extracted IDX files for direct loading in subsequent scripts.
    unpacked = target.with_suffix('')
    content = gzip.decompress(target.read_bytes())
    if not unpacked.exists() or unpacked.read_bytes() != content:
        temporary = unpacked.with_suffix('.part')
        temporary.write_bytes(content)
        temporary.replace(unpacked)
