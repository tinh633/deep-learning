"""What: Log, metadata và checkpoint. Why: Loop train giữ ngắn, artifact kiểm chứng được.
Who: Script train/smoke. When: Tạo run, ghi kết quả hoặc save/resume.
Where: outputs/runs/lan_N. How: logging, JSON, torch.save và trạng thái RNG.
"""
import hashlib
import ctypes
import gc
import json
import logging
import platform
import random
import sys
from datetime import datetime, timezone

import numpy as np
import psutil
import torch
import torchvision

from config import OUTPUT_DIR, PROJECT


def memory_snapshot():
    result = dict(ram_available_mib=psutil.virtual_memory().available / 2**20,
                  process_rss_mib=psutil.Process().memory_info().rss / 2**20)
    if sys.platform == 'win32':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        ('total_phys', ctypes.c_ulonglong), ('avail_phys', ctypes.c_ulonglong),
                        ('total_page', ctypes.c_ulonglong), ('avail_page', ctypes.c_ulonglong),
                        ('total_virtual', ctypes.c_ulonglong), ('avail_virtual', ctypes.c_ulonglong),
                        ('extended', ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError('Cannot read Windows memory status')
        result['commit_available_mib'] = status.avail_page / 2**20
    return result


def guard_memory():
    memory = memory_snapshot()
    if (memory['ram_available_mib'] < 2048 or memory.get('commit_available_mib', 4096) < 3072) and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        memory = memory_snapshot()
    if memory['ram_available_mib'] < 2048 or memory.get('commit_available_mib', 4096) < 3072:
        raise RuntimeError(f'Dừng trước bước nặng vì RAM/Windows commit thấp: {memory}')
    return memory


def state_fingerprints(model):
    # Đọc lần lượt mỗi tensor; tránh clone nguyên backbone chỉ để kiểm tra bất biến.
    return {name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
            for name, value in model.state_dict().items()}


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                         encoding='utf-8')
    temporary.replace(path)


def create_run(settings):
    runs = OUTPUT_DIR / 'runs'
    runs.mkdir(exist_ok=True, parents=True)
    number = 1
    while (runs / f'lan_{number}').exists():
        number += 1
    folder = runs / f'lan_{number}'
    folder.mkdir()
    log = logging.getLogger(str(folder))
    log.setLevel(logging.INFO)
    log.propagate = False
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    for handler in (logging.FileHandler(folder / 'run.log', encoding='utf-8'),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    settings.update(created=datetime.now(timezone.utc).isoformat(),
                    python=platform.python_version(), os=platform.platform(),
                    torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                    cpu=platform.processor(), cpu_threads=psutil.cpu_count(),
                    ram_total_mib=psutil.virtual_memory().total / 2**20,
                    memory_at_start=memory_snapshot(),
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (PROJECT / 'src').glob('*.py')})
    save_json(folder / 'config.json', settings)
    log.info('Config: %s', json.dumps(settings, ensure_ascii=False))
    return folder, log


def save_checkpoint(path, model, optimizer, scaler, epoch, generator, settings):
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                 scaler=scaler.state_dict(), epoch=epoch, settings=settings,
                 python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                 torch_rng=torch.get_rng_state(), loader_rng=generator.get_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, scaler, generator):
    # Chỉ nạp checkpoint local do pipeline này tạo, không nạp pickle không rõ nguồn.
    state = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scaler.load_state_dict(state['scaler'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    generator.set_state(state['loader_rng'])
    if state['cuda_rng'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['epoch']
