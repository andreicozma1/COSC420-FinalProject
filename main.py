import os
from argparse import ArgumentParser
from GymTrainer import GymTrainer

os.environ["WANDB_SILENT"] = "true"


def create_args():
    parser = ArgumentParser("CS420 Final Project")
    parser.add_argument("--episodes", type=int, default=250,
                        help="Number of episodes to run.")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="Initial learning rate before decaying.")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Discount factor to use")
    parser.add_argument("--h1", type=int, default=64,
                        help="Input features")
    parser.add_argument("--h2", type=int, default=32,
                        help="Hidden features")
    parser.add_argument("--sqlen", type=int, default=20,
                        help="Sequence length")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout")
    parser.add_argument("--scale", type=int, default=50,
                        help="Scale")
    parser.add_argument("--log-interval", type=int, default=5,
                        help="In which intervals to display learning progress.")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="In which intervals to save best model checkpoints.")
    parser.add_argument("--model", type=str, default="super", choices=["super"],
                        help="Model to use for training.")
    parser.add_argument("--policy", type=str, default="snn", choices=["snn", "ann"],
                        help="Select policy to use.")
    parser.add_argument("--render", type=bool, default=False,
                        help="Render the environment")
    parser.add_argument("--environment", type=str, default="CartPole-v1",
                        choices=["CartPole-v1", "LunarLander-v2", "BipedalWalker-v2"],
                        help="Gym environment to use.")
    parser.add_argument("--subdir", type=str, default="",
                        help="Subdirectory to save results in.")
    parser.add_argument("--seed", type=int, default=420,
                        help="Random seed to use, or -1 to use a random seed.")
    parser.add_argument("--quiet", type=bool, default=False,
                        help="Don't print to stdout")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="To use GPU or CPU?")
    return parser.parse_args()


def main():
    args = create_args()
    GymTrainer(args).train()


if __name__ == "__main__":
    main()
