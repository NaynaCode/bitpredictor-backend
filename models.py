"""PyTorch model definitions: LSTM, GRU, and MLP for trend prediction.

Supports two task modes via config['taskMode']:
  - 'classification' (default): sigmoid output, P(UP)
  - 'regression': linear output, next-day return
"""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden=16, num_layers=1, dropout=0.0, mode='classification'):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.mode = mode
        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        raw = self.fc(out).squeeze(-1)
        return torch.sigmoid(raw) if self.mode == 'classification' else raw


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden=16, num_layers=1, dropout=0.0, mode='classification'):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.mode = mode
        self.gru = nn.GRU(input_size, hidden, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        raw = self.fc(out).squeeze(-1)
        return torch.sigmoid(raw) if self.mode == 'classification' else raw


class MLPModel(nn.Module):
    def __init__(self, input_size, seq_len, hidden_sizes, activation='relu', dropout=0.0, mode='classification'):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.mode = mode
        flat = input_size * seq_len
        layers = []
        prev = flat
        act = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}[activation]
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        raw = self.net(x).squeeze(-1)
        return torch.sigmoid(raw) if self.mode == 'classification' else raw


def create_model(config, input_size):
    mt = config['modelType']
    mode = config.get('taskMode', 'classification')
    if mt == 'GRU':
        return GRUModel(input_size, config['hiddenUnits'], config['numLayers'],
                        config['dropoutRate'], mode)
    if mt == 'MLP':
        hidden = [config['mlpNeurons']] * config['mlpLayers']
        return MLPModel(input_size, config['sequenceLength'], hidden,
                        config['mlpActivation'], config['dropoutRate'], mode)
    return LSTMModel(input_size, config['hiddenUnits'], config['numLayers'],
                     config['dropoutRate'], mode)


def weights_to_dict(model, config, input_size):
    return {
        'model_type': config['modelType'],
        'config': config,
        'input_size': input_size,
        'state': {k: v.tolist() for k, v in model.state_dict().items()},
    }


def model_from_weights(weights):
    config = weights['config']
    input_size = weights['input_size']
    model = create_model(config, input_size)
    model.load_state_dict({k: torch.tensor(v) for k, v in weights['state'].items()})
    model.eval()
    return model