import torch


class ANN(torch.nn.Module):
    def __init__(self, args):
        super(ANN, self).__init__()
        self.args = args
        if args.environment == "CartPole-v1":
            self.state_space = 4
            self.action_space = 2
        elif args.environment == "LunarLander-v2":
            self.state_space = 8
            self.action_space = 4

        self.l1 = torch.nn.Linear(self.state_space, args.h1, bias=False)
        self.l2 = torch.nn.Linear(args.h1, args.h2, bias=False)
        self.l3 = torch.nn.Linear(args.h2, self.action_space, bias=False)
        self.dropout = torch.nn.Dropout(p=args.dropout)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.l1(x)
        x = self.dropout(x)
        x = torch.nn.functional.relu(x)
        x = self.l2(x)
        x = self.dropout(x)
        x = torch.nn.functional.relu(x)
        x = self.l3(x)
        x = self.dropout(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.softmax(x, dim=-1)
        return x
