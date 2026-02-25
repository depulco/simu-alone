import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn
from torch.distributions import Normal

from create_simulation_fixed import create_sim_test_nn


# ----------------------------- Models -----------------------------
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
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
    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
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
    nb_spikes: int = 5
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
        # New random map each episode
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

        a = torch.tanh(action)
        dx = float(a[0].item() * self.cfg.speed)
        dy = float(a[1].item() * self.cfg.speed)
        drot = float(a[2].item() * self.cfg.rot_speed)

        boule.input_movement(dx, dy, drot)
        self.board.run()
        self.steps += 1

        eaten = self._eaten_count()
        delta_eaten = eaten - self.prev_eaten
        self.prev_eaten = eaten

        done = boule.is_dead() or (self.steps >= self.cfg.max_steps)

        # penalties / shaping
        min_spike = min(boule.saw_by_spike_eyes) if boule.saw_by_spike_eyes else 1.0
        danger_pen = (0.25 - min_spike) * 2.0 if min_spike < 0.25 else 0.0
        act_pen = 0.01 * float((a * a).sum().item())

        # optional shaping toward food (helps learning on random maps)
        min_food = min(boule.saw_by_food_eyes) if boule.saw_by_food_eyes else 1.0
        food_bonus = (1.0 - min_food) * 0.2

        reward = 0.05 + 10.0 * float(delta_eaten) - danger_pen - act_pen + food_bonus
        if boule.is_dead():
            reward -= 10.0

        return self._get_state(), float(reward), done, {}


# ----------------------------- Rollout & GAE -----------------------------
def rollout_episode(env: BouleEnv, actor: Actor, critic: Critic):
    s = env.reset()

    states, actions, logps, rewards, values, dones = [], [], [], [], [], []

    for _ in range(env.cfg.max_steps):
        s_t = s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32)

        dist = actor(s_t)
        v = critic(s_t)

        a = dist.sample()
        logp = dist.log_prob(a).sum()

        s2, r, done, _ = env.step(a.detach())

        states.append(s_t)
        actions.append(a)
        logps.append(logp)
        rewards.append(torch.tensor(r, dtype=torch.float32))
        values.append(v)
        dones.append(torch.tensor(float(done), dtype=torch.float32))

        s = s2
        if done:
            break

    return states, actions, logps, rewards, values, dones


def compute_gae(rewards: List[torch.Tensor], values: List[torch.Tensor], dones: List[torch.Tensor], gamma=0.99, lam=0.95):
    values = values + [torch.tensor(0.0)]
    gae = 0.0
    adv = []
    for t in reversed(range(len(rewards))):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        adv.insert(0, gae)
    returns = [a + v for a, v in zip(adv, values[:-1])]
    return adv, returns


def collect_batch(env: BouleEnv, actor: Actor, critic: Critic, n_rollouts: int):
    all_states, all_actions, all_logps, all_rewards, all_values, all_dones = [], [], [], [], [], []
    ep_returns = []

    for _ in range(n_rollouts):
        states, actions, logps, rewards, values, dones = rollout_episode(env, actor, critic)
        all_states += states
        all_actions += actions
        all_logps += logps
        all_rewards += rewards
        all_values += values
        all_dones += dones
        ep_returns.append(float(torch.stack(rewards).sum().item()) if rewards else 0.0)

    return all_states, all_actions, all_logps, all_rewards, all_values, all_dones, ep_returns


# ----------------------------- PPO Update -----------------------------
def ppo_update(
    actor: Actor,
    critic: Critic,
    opt_a,
    opt_c,
    states,
    actions,
    logps_old,
    returns,
    adv,
    clip_eps=0.2,
    entropy_coef=0.01,
):
    states = torch.stack(states)
    actions = torch.stack(actions)
    logps_old = torch.stack(logps_old).detach()
    returns = torch.stack(returns).detach()
    adv = torch.stack(adv).detach()

    # skip too-small batches
    if states.shape[0] < 2:
        return 0.0, 0.0

    # normalize advantages safely
    adv = adv - adv.mean()
    adv = adv / (adv.std(unbiased=False) + 1e-8)

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

    if not torch.isfinite(loss):
        return float("nan"), float("nan")

    opt_a.zero_grad()
    opt_c.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
    opt_a.step()
    opt_c.step()

    return float(policy_loss.item()), float(value_loss.item())


# ----------------------------- Main -----------------------------
def main():
    torch.manual_seed(0)
    random.seed(0)

    cfg = EnvConfig()
    env = BouleEnv(cfg, seed=0)

    actor = Actor(state_dim=12, action_dim=3)
    critic = Critic(state_dim=12)

    # resume actor if available
    if os.path.exists("policy.pt"):
        actor.load_state_dict(torch.load("policy.pt", map_location="cpu"))
        actor.eval()
        print(" policy.pt chargé -> reprise de l'actor")
    else:
        print("ℹ policy.pt introuvable -> entraînement depuis 0")

    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # PPO batching
    N_ROLLOUTS = 16      # episodes/maps per iteration
    PPO_EPOCHS = 10      # gradient passes over the same batch
    ITERS = 2000         # each iter uses N_ROLLOUTS episodes (so 2000*16=32000 episodes)

    best = None
    for it in range(1, ITERS + 1):
        states, actions, logps, rewards, values, dones, ep_returns = collect_batch(env, actor, critic, N_ROLLOUTS)
        adv, rets = compute_gae(rewards, values, dones)

        pl, vl = 0.0, 0.0
        for _ in range(PPO_EPOCHS):
            pl, vl = ppo_update(actor, critic, opt_a, opt_c, states, actions, logps, rets, adv)

        mean_return = sum(ep_returns) / max(1, len(ep_returns))

        if best is None:
            best = mean_return 
        if mean_return > best:
            best = mean_return
            torch.save(actor.state_dict(), "policy.pt")

        if it % 10 == 0:
            print(f"it={it:4d} mean_return={mean_return:8.2f} pl={pl:7.3f} vl={vl:7.3f} best={best:8.2f}")

    print("OK -> policy.pt sauvegardé")


if __name__ == "__main__":
    main()