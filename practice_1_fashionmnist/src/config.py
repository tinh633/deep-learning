from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = ROOT / 'data' / 'FashionMNIST_raw'
RUNS_FOLDER = ROOT / 'outputs' / 'runs'

DATA = dict(seed=42, val_size=0.1, batch_size=64)
MODEL = dict(hidden1=512, hidden2=256, hidden3=128, dropout=0.2)
TRAIN = dict(seed=42, epochs=50, lr=0.001, weight_decay=1e-5,
             decay_step=15, decay_gamma=0.3)
EVALUATE = dict(run='lan_12', checkpoint='best.pt', include_test=True)

SOURCE = 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'
FILES = {
    'train-images-idx3-ubyte.gz': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
    'train-labels-idx1-ubyte.gz': '25c81989df183df01b3e8a0aad5dffbe',
    't10k-images-idx3-ubyte.gz': 'bef4ecab320f06d8554ea6380940ec79',
    't10k-labels-idx1-ubyte.gz': 'bb300cfdad3c16e7a12a480ee83cd310',
}
NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
