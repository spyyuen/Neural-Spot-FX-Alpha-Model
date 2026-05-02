import torch
import torch.nn as nn

class FXModel(nn.Module):
    def __init__(self, input_dim, model_type="mlp", hidden_dim=32, num_layers=1):
        super().__init__()

        self.model_type = model_type.lower()

        if self.model_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )

        elif self.model_type == "rnn":
            self.rnn = nn.RNN(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_dim, 1)

        elif self.model_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_dim, 1)

        elif self.model_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_dim, 1)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def forward(self, x):
        if self.model_type == "mlp":
            return self.net(x)

        else:
            # x shape: (batch_size, seq_len, input_dim)
            output, _ = self.rnn(x)

            # take final timestep output
            last_hidden = output[:, -1, :]
            return self.fc(last_hidden)
