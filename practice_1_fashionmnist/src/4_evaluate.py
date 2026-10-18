import torch
from torch.utils.data import DataLoader

from config import EVALUATE, RUNS_FOLDER
from data import build_loaders
from logger import open_log

from metrics import evaluate, load_checkpoint

folder = RUNS_FOLDER / EVALUATE['run']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, metadata = load_checkpoint(folder / EVALUATE['checkpoint'], device)
train, val, test = build_loaders(metadata['data_config'])
train = DataLoader(train.dataset, batch_size=train.batch_size, shuffle=False)
logger = open_log(folder, 'test.log' if EVALUATE['include_test'] else 'evaluate.log')
logger.info('Run=%s | checkpoint=%s | eval mode | device=%s', folder.name, EVALUATE['checkpoint'], device)
loaders = [('Train', train), ('Validation', val)]
if EVALUATE['include_test']:
    loaders.append(('Test', test))
for name, loader in loaders:
    result = evaluate(model, loader, device)
    logger.info('%s | samples=%d | loss=%.6f | accuracy=%.2f%%',
                name, result['count'], result['loss'], result['accuracy'] * 100)
