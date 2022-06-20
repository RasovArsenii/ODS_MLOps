import warnings
import click
import mlflow
import json
from collections import OrderedDict
from itertools import chain
import time
import random
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from sklearn.preprocessing import LabelBinarizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import compress_fasttext
from nltk.tokenize import word_tokenize


warnings.filterwarnings('ignore')
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def get_ft_embedding(txt, ft, max_len):
    tokens = word_tokenize(txt, language='russian')

    if len(tokens) > max_len:
        tokens = tokens[:max_len]

    else:
        tokens = ['<PAD>'] * (max_len - len(tokens)) + tokens

    return np.stack(map(ft.get_vector, tokens))

def evaluate(data, model, criterion, device, batch_size, best_f1=False):
    """
    Evaluation, return accuracy and loss
    """
    total_loss = 0.
    y_true = []
    y_pred = []

    model.eval()  # Set mode to evaluation to disable dropout & freeze BN
    data_loader = DataLoader(data, batch_size=batch_size)
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            total_loss += criterion(output, y_batch)
            y_pred.extend(sigmoid(output).cpu().numpy())  # don't forget to execute sigmoid function on logits
            y_true.extend(y_batch.cpu().numpy())
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_pred = np.asarray(y_pred)
    # finding the best threshold with highest f1 score
    accuracy = skm.f1_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average='micro')
    f1 = skm.f1_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average='macro', zero_division=1)

    return {
        'accuracy': '{:<6.4f}'.format(accuracy),
        'f1': '{:<6.4f}'.format(f1),
        'loss': '{:<6.4f}'.format(total_loss / len(data))
    }

def predict_proba(data, model, device, batch_size):
    """
    Prediction, return numpy matrix of predictions (batch_size * n_classes)
    """
    y_pred = []

    model.eval()  # Set mode to evaluation to disable dropout & freeze BN
    data_loader = DataLoader(data, batch_size=batch_size)
    softmax = nn.Softmax()
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            output = model(x_batch)
            y_pred.extend(softmax(output).cpu().numpy())  # don't forget to execute sigmoid function on logits
    y_pred = np.asarray(y_pred)
    return y_pred

class ModuleParallel(nn.Module):
    """
    Execute multiple modules on the same input and concatenate the results
    """
    def __init__(self, modules: list, axis=1):
        super().__init__()
        self.modules_ = nn.ModuleList(modules)
        self.axis = axis

    def forward(self, input):
        return torch.cat([m(input) for m in self.modules_], self.axis)


class GlobalMaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.max(2)[0]

class EarlyStopping:
    """
    Identify whether metric has not been improved for certain number of epochs
    """

    def __init__(self,
                 mode: str = 'min',
                 min_delta: float = 0,
                 patience: int = 20):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience

        self.is_better = None
        if patience == 0:
            self.is_better = lambda *_: True
        else:
            self._init_is_better(mode, min_delta)

        self.best = None
        self.num_bad_epochs = 0

    def step(self, current) -> bool:
        """
        Make decision whether to stop training

        :param current: new metric value
        :return: whether to stop
        """
        if isinstance(current, torch.Tensor):
            current = current.cpu()
        if np.isnan(current):
            return True

        if self.best is None:
            self.best = current
        else:
            if self.is_better(current, self.best):
                self.num_bad_epochs = 0
                self.best = current
            else:
                self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True
        else:
            return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda value, best: value < best - min_delta
        if mode == 'max':
            self.is_better = lambda value, best: value > best + min_delta


class CNNTextClassifier(nn.Module):
    """
    CNN-based text classifier

    It can be used for both multi-class and multi-label classification problem,
     because loss is not specified
    """

    def __init__(self,
                 num_classes,
                 embed_dim,
                 filters=(600,),
                 kernel_sizes=(4,),
                 pooling_dropout=0.8,
                 dense_sizes=(1000,),
                 dense_dropout=0.8,
                 **kwargs):
        """
        :param num_classes: number of outputs (classes)
        :param word_to_id: dictionary used to compose lookup table
        :param use_pretrained_word_vectors: whether to use pre-trained word vectors
        :param word_vectors_path: path to word vectors file (should be in compatible format)
        :param trainable_word_vectors: whether to train (change) vectors
        :param embed_dim: embedding dimensionality in case of `use_pretrained_word_vectors=False`
        :param filters: number of filters (output channels) for each kernel size of the 1st CNN layer
        :param kernel_sizes: kernel sizes of the 1st CNN layer
        :param pooling_dropout: dropout coefficient after pooling layer
        :param dense_sizes: sizes of fully-connected layers
        :param dense_dropout: dropout coefficient after each fully-connected layer
        :param kwargs: ignored arguments
        """
        super().__init__()

        self.convs0 = ModuleParallel([
            nn.Sequential(OrderedDict([
                ('conv0_{}'.format(k), nn.Conv1d(embed_dim, f, k)),
                ('conv0_{}_bn'.format(k), nn.BatchNorm1d(f)),
                ('conv0_{}_relu'.format(k), nn.ReLU()),
                ('conv0_{}_pool'.format(k), GlobalMaxPooling()),
                ('conv0_{}_dp'.format(k), nn.Dropout(pooling_dropout)),
            ]))
            for k, f in zip(kernel_sizes, filters)
        ])

        dense_sizes_in = [sum(filters)] + list(dense_sizes)[:-1]
        self.fcs = nn.Sequential(OrderedDict(chain(*[
            [
                ('fc{}'.format(i), nn.Linear(dense_sizes_in[i], dense_sizes[i])),
                ('fc{}_bn'.format(i), nn.BatchNorm1d(dense_sizes[i])),
                ('fc{}_relu'.format(i), nn.ReLU(inplace=True)),
                ('fc{}_dp'.format(i), nn.Dropout(dense_dropout))
            ] for i in range(len(dense_sizes))
        ])))
        self.fc_last = nn.Linear(dense_sizes[-1], num_classes)

    def forward(self, x):
        # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        x = self.convs0(x.permute(0, 2, 1))
        x = self.fcs(x)
        x = self.fc_last(x)
        return x


@click.command()
@click.argument("train_path", type=click.Path())
@click.argument("valid_path", type=click.Path())
@click.argument("model_path", type=click.Path())
@click.argument("intents_path", type=click.Path())
@click.argument("metrics_path", type=click.Path())
@click.argument("ft_path", type=click.Path())
def train_pipeline(
        train_path: str,
        valid_path: str,
        model_path: str,
        intents_path: str,
        metrics_path: str,
        ft_path: str,
):
    ft = compress_fasttext.models.CompressedFastTextKeyedVectors.load(ft_path)
    EMBED_DIM = ft.vector_size

    train = pd.read_csv(train_path).reset_index(drop=True)
    valid = pd.read_csv(valid_path).reset_index(drop=True)
    MAX_LEN = max(train.text.apply(lambda x: len(word_tokenize(x))))

    all_labels = sorted(train.intent.unique())
    lb = LabelBinarizer()
    lb.classes_ = all_labels
    y_train = lb.transform(train["intent"].values)
    y_val = lb.transform(valid["intent"].values)

    X_train_encoded = np.stack(train["text"].apply(lambda x: get_ft_embedding(x, ft, MAX_LEN)))
    X_val_encoded = np.stack(valid["text"].apply(lambda x: get_ft_embedding(x, ft, MAX_LEN)))

    train_data = TensorDataset(torch.FloatTensor(X_train_encoded), torch.FloatTensor(y_train))
    val_data = TensorDataset(torch.FloatTensor(X_val_encoded), torch.FloatTensor(y_val))

    batch_size = 128
    device = 'cpu'
    model = CNNTextClassifier(
        num_classes=y_train.shape[1],
        embed_dim=EMBED_DIM,
        kernel_sizes=[1, 2],
        filters=[10, 10],
        dense_sizes=[200],
        pooling_dropout=0.45,
        dense_dropout=0.45,
        trainable_word_vectors=False
    )

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                            lr=0.00035)
    criterion = nn.BCEWithLogitsLoss(reduction='sum')   # sigmoid
    scheduler  = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00035, max_lr=0.015, mode='triangular2', cycle_momentum=False)
    early_stopping = EarlyStopping(mode='max', patience=10)
    best_valid_f1 = 0

    for epoch in range(0):
        model.train()
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

        for idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            loss = criterion(model(x_batch), y_batch)
            loss.backward()

            # clipping gradients
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)

            optimizer.step()

        train_metrics = evaluate(train_data, lb, model, criterion, device, batch_size)
        val_metrics = evaluate(val_data, lb, model, criterion, device, batch_size)

        if val_metrics['f1_val'] > best_valid_f1:
            best_valid_f1 = val_metrics['f1_val']
            torch.save(model.state_dict(), model_path)

        scheduler.step(val_metrics['f1_val'])

        print('Epoch {:3}, {}, {}, {}'
              .format(epoch + 1, time.ctime(),
                      ' '.join(['train_{}: {:<6.4f}'.format(k, v) for k, v in train_metrics.items()]),
                      ' '.join(['val_{}: {:<6.4f}'.format(k, v) for k, v in val_metrics.items()])))

        if early_stopping.step(val_metrics['f1_val']):
            break

    torch.save(model.state_dict(), model_path)
    with open(intents_path, 'w') as f:
        f.write(';'.join(lb.classes_))

    model.eval()
    with open(metrics_path, 'w') as f:
        metrics = json.dumps(evaluate(val_data, model, criterion, 'cpu', batch_size))
        f.write(metrics)


if __name__ == "__main__":
    train_pipeline()