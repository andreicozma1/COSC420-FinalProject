from Utils import wandb_get_best, run_pool

policy = "ann"
num_trials = 10
num_trials_final = 25
episodes = 1000

# An initial set of hyperparameters to try
def_lr = 0.001
def_h1 = 256
def_h2 = 256
def_dropout = 0.5
def_gamma = 0.99
def_sqlen = 20
def_scale = 75

# ==============================================================
param_key = "LR"
configurations = []
for _ in range(num_trials):
    configurations.extend(
        (policy, def_gamma, def_h1, def_h2, def_sqlen, def_dropout, def_scale, param_val, param_key, episodes) for
        param_val in
        [0.05, 0.025, 0.0125, 0.005, 0.001, 0.0001])

# run_pool(configurations)
best_lr = wandb_get_best(param_key, policy)
# ==============================================================
param_key = "H1"
configurations = []
for _ in range(num_trials):
    configurations.extend(
        (policy, def_gamma, param_val, def_h2, def_sqlen, def_dropout, def_scale, best_lr, param_key, episodes) for
        param_val in
        [128, 256, 512])
# run_pool(configurations)
best_h1 = wandb_get_best(param_key, policy)
# ==============================================================
param_key = "H2"
configurations = []
for _ in range(num_trials):
    configurations.extend(
        (policy, def_gamma, best_h1, param_val, def_sqlen, def_dropout, def_scale, best_lr, param_key, episodes) for
        param_val in
        [128, 256, 512])
# run_pool(configurations)
best_h2 = wandb_get_best(param_key, policy)
# ==============================================================
param_key = "Dropout"
configurations = []
for _ in range(num_trials):
    configurations.extend(
        (policy, def_gamma, best_h1, best_h2, def_sqlen, param_val, def_scale, best_lr, param_key, episodes) for
        param_val in
        [0.75, 0.5, 0.25])
# run_pool(configurations)
best_dropout = wandb_get_best(param_key, policy)
# ==============================================================
param_key = "Gamma"
configurations = []
for _ in range(num_trials):
    configurations.extend(
        (policy, param_val, best_h1, best_h2, def_sqlen, best_dropout, def_scale, best_lr, param_key, episodes) for
        param_val in
        [0.99, 0.97, 0.95])
# run_pool(configurations)
best_gamma = wandb_get_best(param_key, policy)
# ==============================================================
# param_key = "Sqlen"
# configurations = []
# for _ in range(num_trials):
#     configurations.extend(
#         (policy, best_gamma, best_h1, best_h2, param_val, best_dropout, def_scale, best_lr, param_key, episodes) for param_val in
#         [25, 20, 15])
# run_pool(configurations)
# best_sqlen = wandb_get_best(param_key, policy)
# # ==============================================================
# param_key = "Scale"
# configurations = []
# for _ in range(num_trials):
#     configurations.extend(
#         (policy, best_gamma, best_h1, best_h2, best_sqlen, best_dropout, param_val, best_lr, param_key, episodes) for param_val in
#         [25, 50, 75, 100])
# run_pool(configurations)
# best_scale = wandb_get_best(param_key, policy)
# ==============================================================
print("=" * 80)
print("# Running with the following parameters:")
print(f" - Policy: {policy}")
print(f" - LR: {best_lr}")
print(f" - H1: {best_h1}")
print(f" - H2: {best_h2}")
print(f" - Dropout: {best_dropout}")
print(f" - Gamma: {best_gamma}")
# print(f" - Sqlen: {best_sqlen}")
# print(f" - Scale: {best_scale}")
print("=" * 80)

if policy == "ann":
    configurations = [(policy, best_gamma, best_h1, best_h2, 0, best_dropout, 0, best_lr, "best_overall", episodes)
                      for _ in range(num_trials_final)]
elif policy == "snn":
    configurations = [(policy, best_gamma, best_h1, best_h2, 0, best_dropout, 0, best_lr, "best_overall", episodes)
                      for _ in range(num_trials_final)]
else:
    raise ValueError("Invalid policy")

run_pool(configurations)
best_scale = wandb_get_best(param_key, policy)
