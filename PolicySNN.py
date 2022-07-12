import numpy as np
import torch
from norse.torch import ConstantCurrentLIFEncoder, LIFRecurrentCell, LIFParameters, LILinearCell


class SNN(torch.nn.Module):
    def __init__(self, args):
        super(SNN, self).__init__()
        self.args = args
        if args.environment == "CartPole-v1":
            self.state_dim = 4
            self.output_features = 2
        elif args.environment == "LunarLander-v2":
            self.state_dim = 8
            self.output_features = 4

        # self.input_features = args.h1
        self.hidden_features = args.h2
        self.constant_current_encoder = ConstantCurrentLIFEncoder(args.sqlen)
        self.lif = LIFRecurrentCell(
            2 * self.state_dim,
            self.hidden_features,
            p=LIFParameters(method="super", alpha=100.0),
        )
        self.dropout = torch.nn.Dropout(p=args.dropout)
        self.readout = LILinearCell(self.hidden_features, self.output_features)

        self.last_action = None
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        scale = self.args.scale
        x_pos = self.constant_current_encoder(torch.nn.functional.relu(scale * x))
        x_neg = self.constant_current_encoder(torch.nn.functional.relu(-scale * x))
        x = torch.cat([x_pos, x_neg], dim=2)

        seq_length, batch_size, _ = x.shape

        voltages = torch.zeros(
            seq_length, batch_size, self.output_features, device=x.device
        )
        # State for the hidden and output layers
        s1 = so = None
        # sequential integration loop
        for ts in range(seq_length):
            z1, s1 = self.lif(x[ts, :, :], s1)
            z1 = self.dropout(z1)
            vo, so = self.readout(z1, so)
            voltages[ts, :, :] = vo

        m, _ = torch.max(voltages, 0)
        return torch.nn.functional.softmax(m, dim=1)
