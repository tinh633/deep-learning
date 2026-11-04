"""What: Prepare ten ImageNet-1K validation classes, fifty images each.
Why: Compare pretrained models on a small balanced labeled set outside train.
Who: Presenter runs after granting dataset access and logging in locally.
When: Before benchmark; never train or substitute ImageNetV2/Imagenette.
Where: data/imagenet1k images, manifest and dataset metadata.
How: Stream official Hugging Face validation data, keep original encoded image bytes.
"""
import csv
import hashlib
import io
import json
import random
from pathlib import Path
from PIL import Image
from torchvision.models import ResNet50_Weights
import config


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


if __name__ == '__main__':
    try:
        from datasets import load_dataset, Image as DatasetImage
        from huggingface_hub import get_token, HfApi
    except ImportError as error:
        raise SystemExit('Install requirements.txt first (datasets and huggingface_hub).') from error
    if not get_token():
        raise SystemExit('Login locally with hf auth login, after accepting access on ILSVRC/imagenet-1k. Do not paste tokens in chat.')
    categories = ResNet50_Weights.IMAGENET1K_V1.meta['categories']
    for label, name in config.SELECTED_CLASSES.items():
        if categories[label] != name:
            raise ValueError(f'Unexpected ImageNet mapping: {label}/{name}')
    revision = HfApi().dataset_info(config.HF_DATASET, token=True).sha
    print('Source:', config.HF_DATASET, '| revision:', revision, flush=True)
    source = load_dataset(config.HF_DATASET, split='validation', streaming=True,
                          revision=revision, token=True,
                          cache_dir=str(config.ROOT / 'data' / 'hf_cache'))
    source = source.cast_column('image', DatasetImage(decode=False))
    if source.features['label'].num_classes != 1000:
        raise ValueError('Expected ImageNet-1K labels')
    # Official HF labels follow sorted synsets; check chosen human-readable names too.
    for label, name in config.SELECTED_CLASSES.items():
        actual = source.features['label'].int2str(label).split(',')[0].replace('_', ' ').strip()
        if actual.lower() != name.lower():
            raise ValueError(f'HF/torchvision class mismatch: {label}, {actual}, {name}')
    counts = dict.fromkeys(config.SELECTED_CLASSES, 0)
    rows, hashes = [], set()
    for index, example in enumerate(source):
        label = int(example['label'])
        if label not in counts:
            continue
        if counts[label] >= config.IMAGES_PER_CLASS:
            raise ValueError('Source has more than fifty validation images for selected class')
        encoded = example['image']
        raw = encoded['bytes']
        if raw is None:
            raw = Path(encoded['path']).read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in hashes:
            raise ValueError('Duplicate selected image bytes')
        hashes.add(digest)
        target = config.MANIFEST.parent / 'images' / str(label) / f'validation_{index:05d}.JPEG'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        rows.append(dict(id=f'imagenet1k/validation/{index}',
            path=target.relative_to(config.MANIFEST.parent).as_posix(), label=label,
            split='validation', image_sha256=digest))
        counts[label] += 1
        if len(rows) % 50 == 0:
            print(f'Saved {len(rows)}/500 images; scanned {index + 1} validation records', flush=True)
        if all(count == config.IMAGES_PER_CLASS for count in counts.values()):
            break
    if len(rows) != 500 or any(count != 50 for count in counts.values()):
        raise ValueError(f'Incomplete validation subset: {counts}')
    random.Random(config.SEED).shuffle(rows)
    temporary = config.MANIFEST.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(config.MANIFEST)
    metadata = dict(dataset=config.DATASET_NAME, split='validation', revision=revision,
        source=f'https://huggingface.co/datasets/{config.HF_DATASET}', n=len(rows),
        selected_classes=config.SELECTED_CLASSES, counts=counts, seed=config.SEED,
        manifest_sha256=sha256(config.MANIFEST),
        selection='All fifty validation images for each of ten predefined classes; shuffled seed42')
    (config.MANIFEST.parent / 'dataset_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Prepared:', config.MANIFEST, '| images:', len(rows))
