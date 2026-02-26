import os
import random
from dataclasses import dataclass
from typing import List
import csv
from datetime import datetime

import torch
from torch import nn
from torch.distributions import Normal

from create_simulation_fixed import create_sim_test_nn

print("Torch version :", torch.__version__)
print("CUDA dispo    :", torch.cuda.is_available())
print("Nb GPU        :", torch.cuda.device_count())
print("Device choisi :", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

if torch.cuda.is_available():
    print("GPU name      :", torch.cuda.get_device_name(0))
    print("CUDA version  :", torch.version.cuda)
    x = torch.rand(3, 3).to("cuda")
    print("Test tensor GPU OK :", x.device, x)
else:
    print("Aucun GPU CUDA détecté, exécution sur CPU")
# ----------------------------- Models -----------------------------
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor) -> Normal:
        mu = self.net(x)
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ----------------------------- Env -----------------------------
@dataclass
class EnvConfig:
    width: int = 500
    height: int = 500
    max_steps: int = 600
    speed: float = 2.0
    rot_speed: float = 3.0
    nb_spikes: int = 0     # PHASE A
    nb_food: int = 12
    nb_boules: int = 1


class BouleEnv:
    def __init__(self, cfg: EnvConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.rng = random.Random(seed)
        self.board = None
        self.steps = 0
        self.prev_eaten = 0

    def reset(self):
        self.board = create_sim_test_nn(
            self.cfg.width,
            self.cfg.height,
            self.cfg.nb_spikes,
            self.cfg.nb_food,
            self.cfg.nb_boules,
            seed=self.seed + self.rng.randint(0, 10_000),
            actor_model=None,
        )

        self.steps = 0
        self.prev_eaten = self._eaten_count()

        for b in self.board.boules:
            b.set_pilot(False)
            b.reset_sight()

        return self._get_state()

    def _eaten_count(self) -> int:
        return sum(1 for f in self.board.foods if f.get_eaten())

    def _get_state(self) -> torch.Tensor:
        return self.board.boules[0].get_nn_input()

    def step(self, action: torch.Tensor):
        boule = self.board.boules[0]
        E0 = boule.energy

        a = torch.tanh(action)
        dx = float(a[0].item() * self.cfg.speed)
        dy = float(a[1].item() * self.cfg.speed)
        drot = float(a[2].item() * self.cfg.rot_speed)

        boule.input_movement(dx, dy, drot)
        self.board.run()
        self.steps += 1

        E1 = boule.energy
        done = (boule.energy <= 0) or (self.steps >= self.cfg.max_steps)

        # Reward simple: gros signal "manger"
        energy_gain = max(0, E1 - E0)          # 300 quand il mange
        eat_reward = 0.03 * float(energy_gain) # +9 par nourriture

        eaten = self._eaten_count()
        delta_eaten = eaten - self.prev_eaten
        self.prev_eaten = eaten
        eat_count_reward = 1.0 * float(delta_eaten)

        time_pen = -0.005
        act_pen = 0.001 * float((a * a).sum().item())

        reward = time_pen + eat_reward + eat_count_reward - act_pen

        if boule.energy <= 0:
            reward -= 40.0

        return self._get_state(), float(reward), done, {}


# ----------------------------- Rollout & GAE (bootstrap fix) -----------------------------
@torch.no_grad()
def rollout_episode(env: BouleEnv, actor: Actor, critic: Critic, device):
    s = env.reset()

    states, actions, logps, rewards, values, dones = [], [], [], [], [], []
    start_eaten = env._eaten_count()

    steps = 0
    died = False

    for _ in range(env.cfg.max_steps):
        s_t = s.to(device) if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32, device=device)
        dist = actor(s_t)
        v = critic(s_t)

        a = dist.sample()
        logp = dist.log_prob(a).sum()

        s2, r, done, _ = env.step(a.detach())

        states.append(s_t)
        actions.append(a)
        logps.append(logp)
        rewards.append(torch.tensor(r, dtype=torch.float32, device=device))
        values.append(v)
        dones.append(torch.tensor(float(done), dtype=torch.float32, device=device))

        s = s2
        steps += 1
        if done:
            died = (env.board.boules[0].energy <= 0)
            break

    # Bootstrap si time limit (non-terminal)
    last_value = torch.tensor(0.0, device=device)
    if steps > 0:
        time_limit = (steps >= env.cfg.max_steps) and (env.board.boules[0].energy > 0)
        if time_limit:
            s_last = s.to(device) if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32, device=device)
            last_value = critic(s_last).detach()

    end_eaten = env._eaten_count()
    eaten_this_ep = end_eaten - start_eaten
    ep_return = float(torch.stack(rewards).sum().item()) if rewards else 0.0

    info = {
        "return": ep_return,
        "steps": steps,
        "died": 1.0 if died else 0.0,
        "eaten": float(eaten_this_ep),
        "seed": getattr(env.board, "_seed", None),
    }

    return states, actions, logps, rewards, values, dones, last_value, info


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    vals = values + [last_value]
    gae = torch.tensor(0.0, device=rewards[0].device)
    adv = []

    for t in reversed(range(len(rewards))):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * vals[t + 1] * not_done - vals[t]
        gae = delta + gamma * lam * not_done * gae
        adv.insert(0, gae)

    returns = [a + v for a, v in zip(adv, values)]
    return adv, returns


def collect_batch(env, actor, critic, n_rollouts, device):
    all_states, all_actions, all_logps = [], [], []
    all_adv, all_rets = [], []
    infos = []

    for _ in range(n_rollouts):
        states, actions, logps, rewards, values, dones, last_value, info = rollout_episode(env, actor, critic, device)
        if len(rewards) == 0:
            continue
        adv, rets = compute_gae(rewards, values, dones, last_value)
        all_states += states
        all_actions += actions
        all_logps += logps
        all_adv += adv
        all_rets += rets
        infos.append(info)

    return all_states, all_actions, all_logps, all_adv, all_rets, infos


def mean(xs):
    return sum(xs) / max(1, len(xs))


def init_logger(path="train_log.csv"):
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new_file:
        w.writerow(["time","it","mean_return","mean_eaten","death_rate","mean_steps","pl","vl","best","state_dim"])
    return f, w


# ----------------------------- PPO Update -----------------------------
def ppo_update(actor, critic, opt_a, opt_c, states, actions, logps_old, returns, adv, device, clip_eps=0.2, entropy_coef=0.01, debug=False):
    states = torch.stack(states).to(device)
    actions = torch.stack(actions).to(device)
    logps_old = torch.stack(logps_old).detach().to(device)
    returns = torch.stack(returns).detach().to(device)
    adv = torch.stack(adv).detach().to(device)

    if debug:
        print(f"states={states.device}, actions={actions.device}, returns={returns.device}, adv={adv.device}")

    if states.shape[0] < 2:
        return 0.0, 0.0

    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    dist = actor(states)
    logps = dist.log_prob(actions).sum(dim=1)
    ratio = torch.exp(logps - logps_old)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    entropy = dist.entropy().sum(dim=1).mean()
    policy_loss = policy_loss - entropy_coef * entropy

    values = critic(states)
    value_loss = nn.MSELoss()(values, returns)

    loss = policy_loss + 0.5 * value_loss

    opt_a.zero_grad()
    opt_c.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
    opt_a.step()
    opt_c.step()

    return float(policy_loss.item()), float(value_loss.item())


def save_checkpoint(path, actor, critic, opt_a, opt_c, it, best, extra=None):
    ckpt = {"it": it, "best": best, "actor": actor.state_dict(), "critic": critic.state_dict(),
            "opt_a": opt_a.state_dict(), "opt_c": opt_c.state_dict()}
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


# ----------------------------- Main -----------------------------
def main():
    print(torch.__version__)
    print("CUDA:", torch.cuda.is_available())
    print(torch.cuda.device_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    torch.manual_seed(0)
    random.seed(0)

    cfg = EnvConfig()
    env = BouleEnv(cfg, seed=0)

    # ✅ détecte state_dim automatiquement (12, 20, etc.)
    tmp_state = env.reset()
    state_dim = int(tmp_state.numel())
    print("Detected state_dim =", state_dim)

    actor = Actor(state_dim, 3).to(device)
    critic = Critic(state_dim).to(device)
    print("Actor device  :", next(actor.parameters()).device)
    print("Critic device :", next(critic.parameters()).device)
    # ✅ ne charge policy.pt QUE si compatible
    if os.path.exists("policy.pt"):
        sd = torch.load("policy.pt", map_location=device, weights_only=True)
        if isinstance(sd, dict) and "actor" in sd:
            sd = sd["actor"]
        try:
            actor.load_state_dict(sd)
            actor.eval()
            print("policy.pt chargé")
        except RuntimeError as e:
            print("policy.pt incompatible avec state_dim actuel -> on repart de 0")
            print("Erreur:", e)

    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)

    N_ROLLOUTS = 16
    PPO_EPOCHS = 10
    ITERS = 2000

    log_f, log_w = init_logger("train_log.csv")
    os.makedirs("checkpoints", exist_ok=True)

    best = None
    try:
        for it in range(1, ITERS + 1):
            states, actions, logps, adv, rets, infos = collect_batch(env, actor, critic, N_ROLLOUTS, device)
            if len(states) < 2:
                continue

            pl, vl = 0.0, 0.0
            for epoch in range(PPO_EPOCHS):
                pl, vl = ppo_update(
                    actor, critic, opt_a, opt_c,
                    states, actions, logps, rets, adv, device,
                    debug=(it == 1 and epoch == 0)
                )
            mean_return = mean([i["return"] for i in infos])
            mean_eaten  = mean([i["eaten"] for i in infos])
            death_rate  = mean([i["died"] for i in infos])
            mean_steps  = mean([i["steps"] for i in infos])

            if best is None or mean_return > best:
                best = mean_return
                torch.save(actor.state_dict(), "policy.pt")

            if it % 10 == 0:
                ckpt_path = f"checkpoints/ckpt_it{it:05d}_R{mean_return:.2f}_eat{mean_eaten:.2f}_dead{death_rate*100:.0f}.pt"
                save_checkpoint(ckpt_path, actor, critic, opt_a, opt_c, it=it, best=best,
                                extra={"mean_return": mean_return, "mean_eaten": mean_eaten, "death_rate": death_rate, "mean_steps": mean_steps, "state_dim": state_dim})
                print(f"it={it:4d} R={mean_return:8.2f} eat={mean_eaten:5.2f} dead={death_rate*100:5.1f}% steps={mean_steps:6.1f} pl={pl:7.3f} vl={vl:7.3f} best={best:8.2f}")

            log_w.writerow([datetime.now().isoformat(timespec="seconds"), it,
                            round(mean_return,6), round(mean_eaten,6), round(death_rate,6), round(mean_steps,6),
                            round(pl,6), round(vl,6), round(best if best is not None else 0.0, 6), state_dim])
            log_f.flush()
    finally:
        log_f.close()

    print("OK -> policy.pt sauvegardé")


if __name__ == "__main__":
    main()