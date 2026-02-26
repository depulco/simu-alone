import os
import random
import csv
from dataclasses import dataclass
from datetime import datetime

import torch
from torch import nn
from torch.distributions import Normal

from env_tensor import EnvConfigTensor, BouleEnvTensor


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


def mean(xs):
    return sum(xs) / max(1, len(xs))


def init_logger(path="train_log.csv"):
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new_file:
        w.writerow(["time","it","mean_return","mean_steps","pl","vl","best","state_dim"])
    return f, w


@torch.no_grad()
def rollout_episode(env: BouleEnvTensor, actor: Actor, critic: Critic, device):
    s = env.reset()

    states, actions, logps, rewards, values, dones = [], [], [], [], [], []
    steps = 0
    died = False

    for _ in range(env.cfg.max_steps):
        s_t = s.to(device)
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
            died = True
            break

    last_value = torch.tensor(0.0, device=device)
    if steps > 0:
        time_limit = (steps >= env.cfg.max_steps) and (not died)
        if time_limit:
            s_last = s.to(device)
            last_value = critic(s_last).detach()

    ep_return = float(torch.stack(rewards).sum().item()) if rewards else 0.0

    info = {
        "return": ep_return,
        "steps": steps,
        "died": 1.0 if died else 0.0,
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
    ckpt = {
        "it": it,
        "best": best,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "opt_a": opt_a.state_dict(),
        "opt_c": opt_c.state_dict(),
    }
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


def main():
    print("Torch version :", torch.__version__)
    print("CUDA dispo    :", torch.cuda.is_available())
    print("Nb GPU        :", torch.cuda.device_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device choisi :", device)

    if torch.cuda.is_available():
        print("GPU name      :", torch.cuda.get_device_name(0))
        print("CUDA version  :", torch.version.cuda)
        x = torch.rand(3, 3, device=device)
        print("Test tensor GPU OK :", x.device, x)

    torch.manual_seed(0)
    random.seed(0)

    cfg = EnvConfigTensor(device=str(device))
    env = BouleEnvTensor(cfg, seed=0)

    tmp_state = env.reset()
    state_dim = int(tmp_state.numel())
    print("Detected state_dim =", state_dim)

    actor = Actor(state_dim, 3).to(device)
    critic = Critic(state_dim).to(device)

    print("Actor device  :", next(actor.parameters()).device)
    print("Critic device :", next(critic.parameters()).device)

    if os.path.exists("policy.pt"):
        sd = torch.load("policy.pt", map_location=device, weights_only=True)
        if isinstance(sd, dict) and "actor" in sd:
            sd = sd["actor"]
        try:
            actor.load_state_dict(sd)
            actor.eval()
            print("policy.pt chargé")
        except RuntimeError as e:
            print("policy.pt incompatible -> restart from scratch")
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
            mean_steps = mean([i["steps"] for i in infos])

            if best is None or mean_return > best:
                best = mean_return
                torch.save(actor.state_dict(), "policy.pt")

            if it % 10 == 0:
                ckpt_path = f"checkpoints/ckpt_it{it:05d}_R{mean_return:.2f}.pt"
                save_checkpoint(
                    ckpt_path,
                    actor, critic, opt_a, opt_c,
                    it=it,
                    best=best,
                    extra={"mean_return": mean_return, "mean_steps": mean_steps, "state_dim": state_dim},
                )
                print(
                    f"it={it:4d} R={mean_return:8.2f} steps={mean_steps:6.1f} "
                    f"pl={pl:7.3f} vl={vl:7.3f} best={best:8.2f}"
                )

            log_w.writerow([
                datetime.now().isoformat(timespec="seconds"),
                it,
                round(mean_return, 6),
                round(mean_steps, 6),
                round(pl, 6),
                round(vl, 6),
                round(best if best is not None else 0.0, 6),
                state_dim
            ])
            log_f.flush()
    finally:
        log_f.close()

    print("OK -> policy.pt sauvegardé")


if __name__ == "__main__":
    main()