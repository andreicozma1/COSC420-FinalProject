import multiprocessing
import os
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb


def init_env(name, seed=None):
    env = gym.make(name)
    env.reset()
    if seed is not None and seed != -1:
        env.seed(seed)
    return env


def unit_normal(arr):
    eps = np.finfo(np.float32).eps.item()
    arr = torch.as_tensor(arr)
    arr = (arr - arr.mean()) / (arr.std() + eps)
    return arr


def get_running_reward(ep_reward, running_reward):
    return 0.05 * ep_reward + (1 - 0.05) * running_reward


def set_seeds(seed):
    if seed is not None and seed != -1:
        # print(f"# Setting seed to {seed}")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def run(policy, gamma, h1, h2, sqlen, dropout, scale, lr, subdir, episodes):
    env = "CartPole-v1"
    subdir = f"{subdir}_{episodes}"
    command = f"python3 main.py " \
              f"--environment {env} --policy {policy} --episodes {episodes} " \
              f"--lr {lr} --gamma {gamma} " \
              f"--h1 {h1} --h2 {h2} --dropout {dropout} " \
              f"--sqlen {sqlen} --scale {scale} " \
              f"--quiet True --seed -1 --subdir {subdir}"
    os.system(command)


def run_pool(configurations, shuffle=True, workers=5):
    print("=" * 80)
    if shuffle:
        random.shuffle(configurations)
    print("# Total configurations:", len(configurations))
    with multiprocessing.Pool(workers) as p:
        print(f"# Starting multiprocessing with {workers} workers")
        p.starmap(run, configurations)


def wandb_get_best(param_name, policy_name, metric_name="Running Reward", team_name="utkteam", env_name="CartPole-v1"):
    api = wandb.Api()
    print("------------------------------------------------------")

    # Project is specified by <entity/project-name>
    name = f"{team_name}/{env_name}-{policy_name}-{param_name}"
    runs = api.runs(name)
    print(f"# Looking for best {param_name} in `{name}` ({len(runs)} runs)")
    summary_list = []
    config_list = []
    name_list = []
    for run in runs:
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # run.config is the input metrics.
        # We remove special values that start with _.
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        config_list.append(config)

        # run.name is the name of the run.
        name_list.append(run.name)

    import pandas as pd
    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({'name': name_list})
    all_df = pd.concat([name_df, config_df, summary_df], axis=1)

    by_param = all_df.groupby(param_name).mean()
    by_param.sort_values(metric_name, ascending=False)
    print(by_param)
    best_param = by_param.idxmax()[metric_name]
    print(f"# Best {param_name}: {best_param} (Mean {metric_name} {by_param.loc[best_param][metric_name]})")
    print("------------------------------------------------------")

    return best_param


class History:
    # History object that also helps plot the data
    def __init__(self, run, x="_step", keys=None):
        self.run = run
        self.x = x
        self.keys = keys
        self.hist = run.history(pandas=True, x_axis=x, keys=keys)

    def mean(self, key):
        return self.hist[key].mean()

    def median(self, key):
        return self.hist[key].median()

    def std(self, key):
        return self.hist[key].std()

    def min(self, key):
        return self.hist[key].min()

    def max(self, key):
        return self.hist[key].max()

    def last(self, key):
        return self.hist[key].iloc[-1]

    def plot(self, x, y, label_prefix=None, **kwargs):
        if "label" not in kwargs:
            lb = f"End: {self.last(y):.2f}; Max: {self.max(y):.2f}"
            if label_prefix is not None:
                lb = f"{label_prefix}. {lb}"
            kwargs["label"] = lb
        # If the last is smaller than the max, plot it as a dotted line
        # if self.last(y) < self.max(y):
        #     kwargs["linestyle"] = "dotted"
        # Plot the data
        plt.plot(self.hist[x], self.hist[y], **kwargs)


class Histories:
    def __init__(self, project_name):
        api = wandb.Api(timeout=30)
        self.project_name = project_name
        self.runs = api.runs(project_name)
        self.data = self.get_data()
        self.policy_name = self.data["Policy"].unique()[0].upper()

    def get_data(self):
        summary_list = []
        config_list = []
        name_list = []
        for run in self.runs:
            # run.summary are the output key/values like accuracy.
            # We call ._json_dict to omit large files
            summary_list.append(run.summary._json_dict)

            # run.config is the input metrics.
            # We remove special values that start with _.
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            config_list.append(config)

            # run.name is the name of the run.
            name_list.append(run.name)

        import pandas as pd
        summary_df = pd.DataFrame.from_records(summary_list)
        config_df = pd.DataFrame.from_records(config_list)
        name_df = pd.DataFrame({'name': name_list})
        all_df = pd.concat([name_df, config_df, summary_df], axis=1)
        return all_df

    def get_agg_func(self, agg, fallback=None):
        # If the type is function, return it
        if callable(agg):
            return agg
        # If the type is string then return the corresponding function
        elif isinstance(agg, str):
            if agg == "mean":
                return np.mean
            elif agg == "median":
                return np.median
            elif agg == "std":
                return np.std
            elif agg == "min":
                return np.min
            elif agg == "max":
                return np.max
            else:
                raise ValueError(f"Unknown aggregation function: {agg}")
        elif fallback is not None:
            return fallback
        else:
            raise ValueError(f"Unknown aggregation function: {agg}")

    def plot(self, x="Episode", y="Running Reward", sort_by="Running Reward", reverse=True, n=5,
             agg=None, range=None, range_n=None, figsize=(8, 5), legend=True):
        print(f"# Plot: {y} vs {x}")
        if range_n is None:
            range_n = n
        # Plot the "Running Reward" vs "Episode" for the top 5 runs on the same plot in different colors
        max_n = max(n, range_n)
        runs_max = sorted(self.runs, key=lambda run: run.summary.get(sort_by), reverse=reverse)[:max_n]
        histories_max = [History(r, x=x, keys=[y]) for r in runs_max]
        histories_n = histories_max[:n]
        histories_range_n = histories_max[:range_n]
        # Concatenate the histories for agg and range
        df = pd.concat([h.hist for h in histories_range_n]).groupby(x)
        df_agg = df.agg(self.get_agg_func(agg, fallback=np.mean))

        # Plot the data
        plt.figure(figsize=figsize)

        # Plot the data for each run
        cmap = plt.cm.get_cmap('RdYlGn_r')
        colors = cmap(np.linspace(0, 1, len(histories_n)))

        title = f"{y} vs {x}"
        if agg is None:
            title = f"{title} (Top {n})"
            for i, h in enumerate(histories_n):
                h.plot(x, y, label_prefix=i + 1, color=colors[i])
        else:
            agg_label = agg[0].upper() + agg[1:]
            title = f"{agg_label} of {title} (Top {range_n})"
            plt.plot(df_agg.index, df_agg[y],
                     label=f"{agg_label}: {df_agg[y].iloc[-1]:.2f}")
            if range is not None:
                df_range = df.agg(self.get_agg_func(range))
                range_label = range[0].upper() + range[1:]
                title = f"{range_label} & {title}"
                plt.fill_between(df_agg.index, df_agg[y] - df_range[y], df_agg[y] + df_range[y],
                                 alpha=0.1, label=f"{range_label}: {df_range[y].iloc[-1]:.2f}")

        plt.title(f"{self.policy_name} - {title}")
        if legend:
            plt.legend()
        plt.xlabel(x)
        plt.ylabel(y)
        fname = self.get_fname("plot", y, x)
        fname += f"_top_{n}_runs"
        if agg is not None:
            fname += f"_{agg}"
        if range is not None:
            fname += f"_{range}"
        plt.savefig(fname)
        plt.show()

    def get_fname(self, prefix, m1, m2):
        p = self.project_name.split("/")[-1]
        fname = f"{prefix}_{p}_{m1.replace(' ', '')}_vs_{m2.replace(' ', '')}"
        return fname.lower()

    def boxplot(self, by, column="Running Reward"):
        self.data.boxplot(column=[column], by=by, figsize=(8, 5))
        plt.suptitle("")
        by_label = by
        if by == "LR":
            by_label = "Learning Rate"
        if by == "Dropout":
            by_label = "Dropout Rate"
        if self.policy_name == "SNN" and by == "H2":
            by_label = "Hidden State Neurons"
        if self.policy_name == "ANN":
            if by == "H1":
                by_label = "Hidden Layer 1 Neurons"
            if by == "H2":
                by_label = "Hidden Layer 2 Neurons"

        plt.title(f"{self.policy_name} - Effect of {by_label} on {column}")
        plt.ylabel(column)
        plt.xlabel(by_label)
        fname = self.get_fname("boxplot", column, by)
        plt.savefig(fname)
        plt.show()
