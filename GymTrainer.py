# Parts of this code are adaptations from:
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
import os

import numpy as np
import torch

import wandb
from PolicyANN import ANN
from PolicySNN import SNN
from Utils import init_env, get_running_reward, set_seeds


class GymTrainer:
    def __init__(self, args):
        self.args = args
        set_seeds(args.seed)
        self.device = torch.device(args.device)
        self.env = init_env(args.environment, args.seed)
        self.policy, self.optimizer, self.scheduler = self.init_policy()
        self.create_logdir()
        self.running_reward, self.rewards_r, self.rewards_e = 10, [], []

    def init_policy(self):
        # Select the policy
        if self.args.policy == "ann":
            policy = ANN(self.args).to(self.device)
        elif self.args.policy == "snn":
            policy = SNN(self.args).to(self.device)
        else:
            raise ValueError(f"Unknown policy {self.args.policy}")
        # Define the optimizer and scheduler for learning rate decay on plateau
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, min_lr=1e-5, factor=0.25, verbose=True)
        return policy, optimizer, scheduler

    def train(self, episodes=None):
        if episodes is None:
            episodes = self.args.episodes
        self.init_wandb(self.args)

        for e_i in range(episodes):
            e_reward, e_steps = self.simulate(self.policy)
            self.running_reward = get_running_reward(e_reward, self.running_reward)
            e_loss = self.finish_episode(self.policy, self.optimizer, self.args.gamma)

            self.log(e_i, e_loss, e_reward, e_steps, episodes)
            self.rewards_e.append(e_reward)
            self.rewards_r.append(self.running_reward)
            if e_i % self.args.save_interval == 0:
                self.save_checkpoint()
            # self.scheduler.step(e_loss)
            if self.running_reward > self.env.spec.reward_threshold:
                print(f"Solved! Running reward is now {self.running_reward} "
                      f"and the last episode runs to {e_steps} time steps!")
                break

    def simulate(self, policy, steps=750):
        state, total_reward, step = self.env.reset(), 0, 0
        for step in range(1, steps):
            action = self.select_action(state, policy, device=self.device)
            state, reward, done, _ = self.env.step(action)
            if self.args.render:
                self.env.render()
            policy.rewards.append(reward)
            total_reward += reward
            if done:
                break
        return total_reward, step

    def select_action(self, state, policy, device):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = policy(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        policy.saved_log_probs.append(m.log_prob(action))
        action = action.item()
        policy.last_action = action
        return action

    def finish_episode(self, policy, optimizer, gamma):
        eps = np.finfo(np.float32).eps.item()
        R, loss, returns = 0, [], []
        # Calculate the discounted return R for each time step to use as the target for the loss
        for r in policy.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        # Normalize the returns to be unit normal
        returns = torch.as_tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        # Calculate the loss as a weighted sum of the returns
        for log_prob, R in zip(policy.saved_log_probs, returns):
            loss.append(-log_prob * R)
        # Reset the optimizer gradient
        optimizer.zero_grad()
        # Perform the backward pass to calculate the gradients then update the weights
        loss = torch.cat(loss).sum()
        loss.backward()
        optimizer.step()
        # Delete the saved log probs and rewards to save memory
        del policy.rewards[:]
        del policy.saved_log_probs[:]
        return loss

    def get_best_saved(self):
        return np.load("best_running_rewards.npy", allow_pickle=True)[-1] \
            if os.path.exists("best_running_rewards.npy") else None

    def create_logdir(self):
        label = f"{self.args.environment}-{self.args.policy}"
        if self.args.subdir != "" and self.args.subdir is not None:
            results_dir = os.path.join("results", label, self.args.subdir)
        else:
            results_dir = os.path.join("results", label)
        os.makedirs(results_dir, exist_ok=True)
        os.chdir(results_dir)

    def init_wandb(self, args):
        wandb_config = {
            "Environment": args.environment,
            "Policy": args.policy,
            "LR": args.lr,
            "Gamma": args.gamma,
            "H1": args.h1,
            "H2": args.h2,
            "Dropout": args.dropout,
        }
        run_name = f"p{args.policy}-lr{args.lr}-g{args.gamma}-" \
                   f"h1{args.h1}-h2{args.h2}-dr{args.dropout}"
        if args.policy == "snn":
            wandb_config.update({"Sqlen": args.sqlen,
                                 "Scale": args.scale})
            run_name += f"-sq{args.sqlen}-s{args.scale}"

        proj_name = f"{args.environment}-{args.policy}"
        if args.subdir != "" and args.subdir is not None:
            proj_name += f"-{args.subdir}"
        wandb.init(project=proj_name, entity="utkteam",
                   name=run_name,
                   config=wandb_config)

    def log(self, e_i, e_loss, e_reward, e_steps, episodes):
        if not self.args.quiet and e_i % self.args.log_interval == 0:
            print(f"{e_i}/{episodes}.\tRunning Reward: {self.running_reward:.2f}\t Last Reward: {e_reward}\t")
        wandb.log({"Episode": e_i,
                   "Reward": e_reward,
                   "Policy Loss": e_loss,
                   "Running Reward": self.running_reward,
                   "Steps": e_steps
                   })

    def save_checkpoint(self):
        best_running_reward = self.get_best_saved()
        if best_running_reward is None or self.running_reward > best_running_reward:
            # If the current running reward is better than the last, save the current model
            print(f"# Saving new best model w/ running reward: {self.running_reward}")
            np.save("best_running_rewards.npy", np.array(self.rewards_r))
            np.save("best_episode_rewards.npy", np.array(self.rewards_e))
            torch.save(self.optimizer.state_dict(), "best_optimizer.pt")
            torch.save(self.scheduler.state_dict(), "best_scheduler.pt")
            torch.save(self.policy.state_dict(), "best_policy.pt")
            # Save optimizer, scheduler, and policy to wandb for future use
            wandb.save("optimizer.pt")
            wandb.save("scheduler.pt")
            wandb.save("policy.pt")
            with open("best_args.txt", "w") as fp:
                fp.write(str(self.args))
