"""Regression CPU với model/dữ liệu tổng hợp; không tải weights hoặc đọc CIFAR test."""
import contextlib
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
import config
import data
import logger
import model


class SmallClassifier(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(3, width, 1)
        self.fc = nn.Linear(width, 100)

    def forward(self, images):
        features = self.pool(images)
        features = self.conv(features).flatten(1)
        return self.fc(features)


def synthetic_model(name, strategy, pretrained=True):
    result = SmallClassifier(2048 if name == 'resnet50' else 768)
    for parameter in result.parameters():
        parameter.requires_grad = strategy == 'fine_tune'
    for parameter in result.fc.parameters():
        parameter.requires_grad = True
    return result


def synthetic_loaders(batch_size, num_workers=0, train_images=None, val_images=None):
    generator = torch.Generator().manual_seed(7)
    n = train_images or 4
    m = val_images or 4
    train = TensorDataset(torch.rand(n, 3, 224, 224, generator=generator), torch.arange(n) % 2)
    val = TensorDataset(torch.rand(m, 3, 224, 224, generator=generator), torch.arange(m) % 2)
    # Smoke lưu IDs/transform trong config, không cần dataset thật.
    train.indices, val.indices = list(range(n)), list(range(100, 100+m))
    train.dataset = val.dataset = type('TransformInfo', (), {'transform': 'synthetic tensors'})()
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False,
                            generator=torch.Generator().manual_seed(42))
    return train_loader, val_loader, [f'class_{i}' for i in range(100)]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        (PROJECT / 'data').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='pipeline_check_', dir=PROJECT / 'data')
        self.root = Path(self.temp.name)
        assert self.root.resolve().parent == (PROJECT/'data').resolve()
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.object(logger, 'OUTPUT_DIR', self.root/'outputs'))
        self.stack.enter_context(patch.object(config, 'OUTPUT_DIR', self.root/'outputs'))
        self.stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=False))
        self.stack.enter_context(patch.object(logger, 'guard_memory', return_value={'test': 'mock CPU guard'}))
        self.stack.enter_context(patch.object(model, 'build_model', side_effect=synthetic_model))
        self.stack.enter_context(patch.object(data, 'build_loaders', side_effect=synthetic_loaders))
        self.stack.enter_context(patch.multiple(config, PHASE='tuning', RUN_PURPOSE='experiment', TRAINING_APPROVED=True,
            DUPLICATE_POLICY='keep_reported', EXPERIMENT='B', CANDIDATE='LR1', SELECTED_FROM=None,
            BATCH_SIZE=2, EPOCHS=2, LEARNING_RATE=1e-4, WEIGHT_DECAY=0.0,
            RESUME_RUN=None, NUM_WORKERS=0, USE_AMP=False, RUN_TEST=False,
            ALLOW_SMOKE_CHECKPOINT=False, MAX_EVAL_IMAGES=None,
            EVAL_RUN=None, EVAL_SPLIT='validation', EVAL_CHECKPOINT='best.pt', DEMO_IMAGE=None,
            SMOKE_EXPERIMENTS=('A', 'B', 'C', 'D')))
        self.original_create = logger.create_run

        def create(settings, folder=None):
            settings['test_scope'] = 'synthetic CPU integration; no pretrained or CIFAR data'
            return self.original_create(settings, folder)
        self.stack.enter_context(patch.object(logger, 'create_run', side_effect=create))

    def tearDown(self):
        self.stack.close()
        for log in list(logger.logging.Logger.manager.loggerDict.values()):
            if isinstance(log, logger.logging.Logger) and str(self.root) in log.name:
                for handler in list(log.handlers):
                    handler.close()
                    log.removeHandler(handler)
        assert self.root.resolve().parent == (PROJECT/'data').resolve()
        self.temp.cleanup()

    def run_script(self, name):
        # Giữ stdout thật để script có thể reconfigure trên Windows.
        runpy.run_path(str(PROJECT/'src'/name), run_name='__main__')

    def test_train_resume_evaluate_demo(self):
        with patch.object(config, 'TRAINING_APPROVED', False):
            with self.assertRaises(SystemExit):
                self.run_script('2_train.py')
        real_save = logger.save_checkpoint

        def interrupt_after_last(path, *args, **kwargs):
            real_save(path, *args, **kwargs)
            if path.name == 'last.pt':
                raise KeyboardInterrupt('simulated epoch-boundary interruption')

        with patch.object(logger, 'save_checkpoint', side_effect=interrupt_after_last):
            with self.assertRaises(KeyboardInterrupt):
                self.run_script('2_train.py')
        pilot = self.root/'outputs/runs/lan_1'
        self.assertEqual(json.loads((pilot/'summary.json').read_text())['status'], 'interrupted')
        config.RESUME_RUN = 'lan_1'
        self.run_script('2_train.py')
        self.assertEqual(json.loads((pilot/'summary.json').read_text())['status'], 'completed')
        self.assertEqual([row['epoch'] for row in json.loads((pilot/'history.json').read_text())], [1, 2])
        state = logger.read_checkpoint(pilot/'last.pt')
        incompatible = dict(state['settings'], lr=1e-2)
        tiny = synthetic_model('resnet50', 'fine_tune')
        optimizer = torch.optim.AdamW(tiny.parameters())
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        with self.assertRaisesRegex(ValueError, 'lr'):
            logger.load_checkpoint(pilot/'last.pt', tiny, optimizer, scaler,
                                   torch.Generator(), expected=incompatible)
        del state

        config.PHASE, config.CANDIDATE, config.SELECTED_FROM, config.RESUME_RUN = 'final', None, 'lan_1', None
        real_read = logger.read_checkpoint

        def resource_pilot(path):
            state = real_read(path)
            state['settings'] = dict(state['settings'], purpose='resource_probe')
            return state

        with patch.object(logger, 'read_checkpoint', side_effect=resource_pilot):
            with self.assertRaisesRegex(ValueError, 'resource_probe'):
                self.run_script('2_train.py')
        self.run_script('2_train.py')
        final = self.root/'outputs/runs/lan_2'
        self.assertEqual(json.loads((final/'summary.json').read_text())['status'], 'completed')
        resumed_state = logger.read_checkpoint(pilot/'last.pt')['model']
        continuous_state = logger.read_checkpoint(final/'last.pt')['model']
        self.assertTrue(all(torch.equal(value, continuous_state[key]) for key, value in resumed_state.items()))
        del resumed_state, continuous_state
        calls = []

        def evaluation_loader(split, batch_size, workers, maximum):
            calls.append(split)
            _, loader, classes = synthetic_loaders(batch_size)
            return loader, classes, list(range(len(loader.dataset)))

        config.EVAL_RUN, config.EVAL_CHECKPOINT, config.EVAL_SPLIT = 'lan_2', 'best.pt', 'validation'
        with patch.object(data, 'build_evaluation_loader', side_effect=evaluation_loader):
            self.run_script('3_evaluate.py')
            config.EVAL_SPLIT = 'test'
            with self.assertRaises(SystemExit):
                self.run_script('3_evaluate.py')
            config.RUN_TEST, config.EVAL_RUN = True, 'lan_1'
            with self.assertRaises(ValueError):
                self.run_script('3_evaluate.py')
            config.RUN_TEST, config.EVAL_RUN = False, 'lan_2'
        self.assertEqual(calls, ['validation'])
        result = next((final/'evaluation').glob('*/metrics.json'))
        metrics = json.loads(result.read_text())
        predictions = json.loads((result.parent/'predictions.json').read_text())
        self.assertEqual(metrics['n'], 4)
        self.assertEqual(len(metrics['per_class']), 100)
        self.assertEqual(metrics['accuracy'], sum(r['true']==r['pred'] for r in predictions)/4)
        self.assertIsNone(metrics['per_class'][99]['auc'])
        image = self.root/'example.png'
        Image.new('RGB', (32, 32), color='red').save(image)
        config.DEMO_IMAGE = str(image)
        self.run_script('4_demo.py')
        demo = json.loads(next((final/'demo').glob('*.json')).read_text())
        self.assertEqual(len(demo['top5']), 5)

    def test_smoke_and_post_replay_guard(self):
        config.PHASE = 'smoke'
        config.SMOKE_EXPERIMENTS = ('A', 'B', 'C', 'D')
        self.run_script('0_smoke.py')
        summaries = sorted((self.root/'outputs/runs').glob('*/summary.json'))
        self.assertEqual(len(summaries), 4)
        for path in summaries:
            result = json.loads(path.read_text())
            self.assertEqual(result['status'], 'passed')
            self.assertIn('memory_final_guard', result)
            self.assertIn('memory_after_save', result)
            self.assertIn('memory_after_load', result)
            self.assertIn('memory_after_replay_load', result)
        calls = 0

        def low_after_replay():
            nonlocal calls
            calls += 1
            if calls == 18:
                raise RuntimeError('RAM/Windows commit simulated low after replay')
            return {'test': 'mock CPU guard'}

        config.SMOKE_EXPERIMENTS = ('A',)
        with patch.object(logger, 'guard_memory', side_effect=low_after_replay):
            with self.assertRaises(SystemExit):
                self.run_script('0_smoke.py')
        latest = self.root/'outputs/runs/lan_5/summary.json'
        result = json.loads(latest.read_text())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['failed_check'], 'resources')


if __name__ == '__main__':
    unittest.main()
