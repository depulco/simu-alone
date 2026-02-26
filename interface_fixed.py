import os
import glob
import random
import pygame
import torch
import copy
from collections import deque

from create_simulation_fixed import create_sim_test_nn


# ----------------------- Pilot -----------------------
class BouleNNPilot:
    def __init__(self, boule, policy, speed=2.0, rot_speed=3.0, stochastic=False, device="cpu"):
        self.boule = boule
        self.policy = policy.to(device).eval()
        self.speed = float(speed)
        self.rot_speed = float(rot_speed)
        self.stochastic = stochastic
        self.device = device

    def get_move(self):
        with torch.inference_mode():
            s = self.boule.get_nn_input().to(self.device)  # [12]
            dist = self.policy(s)
            a = dist.sample() if self.stochastic else dist.mean
            a = torch.tanh(a)
        return (
            float(a[0].item() * self.speed),
            float(a[1].item() * self.speed),
            float(a[2].item() * self.rot_speed),
        )


# ----------------------- Model -----------------------
class Actor(torch.nn.Module):
    def __init__(self, state_dim=20, action_dim=3, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, action_dim),
        )
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        from torch.distributions import Normal
        mu = self.net(x)
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)


# ----------------------- Loading -----------------------
def load_policy_any(path, device="cpu"):
    policy = Actor().to(device)
    obj = torch.load(path, map_location=device)

    if isinstance(obj, dict) and "actor" in obj:
        policy.load_state_dict(obj["actor"])
    else:
        policy.load_state_dict(obj)

    policy.eval()
    return policy


def list_checkpoints():
    files = []
    if os.path.isdir("checkpoints"):
        files += sorted(glob.glob("checkpoints/*.pt"))
    if os.path.exists("policy.pt"):
        files.append("policy.pt")
    return files


# ----------------------- Game -----------------------
class Game:
    def __init__(self, win_w, win_h, board):
        pygame.init()
        self.board = board
        self.world = pygame.Surface((board.width, board.height))
        self.screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.running = True

        self.show_eyes = True
        self.show_spike_eyes = False  # ✅ en mode "sans spikes", inutile -> laisse False
        self.fps = 120

        self.paused = False
        self.step_once = False

        pygame.font.init()
        self.font = pygame.font.SysFont("consolas", 18)

        # rewind buffer
        self.history = deque(maxlen=600)
        self.history_idx = -1
        self.push_snapshot()  # snapshot initial

    # ----------- rewind helpers -----------
    def push_snapshot(self):
        while len(self.history) - 1 > self.history_idx:
            self.history.pop()
        self.history.append(copy.deepcopy(self.board))
        self.history_idx = len(self.history) - 1

    def pop_snapshot(self):
        if self.history_idx > 0:
            self.history_idx -= 1
            self.board = copy.deepcopy(self.history[self.history_idx])
            self.world = pygame.Surface((self.board.width, self.board.height))

    def clear_history(self):
        self.history.clear()
        self.history_idx = -1
        self.push_snapshot()

    # ----------- drawing -----------
    def draw_spikes(self):
        # ✅ pas de spikes -> ne dessine rien
        for spike in self.board.spikes:
            rect = pygame.Rect(
                spike.get_x() - (spike.get_radius() * 0.70),
                spike.get_y() - (spike.get_radius() * 0.70),
                spike.get_radius() * 1.41,
                spike.get_radius() * 1.41,
            )
            pygame.draw.rect(self.world, "red", rect)

    def draw_food(self):
        for food in self.board.foods:
            if not food.get_eaten():
                pygame.draw.circle(self.world, "green", (food.get_x(), food.get_y()), food.get_radius())

    def draw_boules(self):
        for boule in self.board.boules:
            if boule.is_dead():
                continue
            pygame.draw.circle(self.world, "blue", (boule.get_x(), boule.get_y()), boule.get_radius())

            if not self.show_eyes:
                continue

            # food eyes
            for i, eye in enumerate(boule.get_food_eyes()):
                color = "orange" if boule.saw_by_food_eyes[i] != 1 else "green"
                pygame.draw.line(self.world, color, (boule.x, boule.y), eye.get_end_sight())

            # spike eyes (optionnel)
            if self.show_spike_eyes:
                for i, eye in enumerate(boule.get_spike_eyes()):
                    color = "purple" if boule.saw_by_spike_eyes[i] != 1 else "yellow"
                    pygame.draw.line(self.world, color, (boule.x, boule.y), eye.get_end_sight())

    def overlay_text(self, lines):
        y = 8
        for line in lines:
            surf = self.font.render(line, True, (0, 0, 0))
            self.screen.blit(surf, (8, y))
            y += 20

    def handle_events(self):
        keys = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                keys.append(event.key)
        return keys

    def step_and_render(self, overlay_lines=None):
        keys = self.handle_events()

        if pygame.K_ESCAPE in keys:
            self.running = False
        if pygame.K_e in keys:
            self.show_eyes = not self.show_eyes
        if pygame.K_SPACE in keys:
            self.paused = not self.paused
        if pygame.K_n in keys:
            self.step_once = True
        if pygame.K_b in keys and self.paused:
            self.pop_snapshot()
        if pygame.K_x in keys:
            # ✅ option : afficher/cacher spike eyes (même si spikes=0)
            self.show_spike_eyes = not self.show_spike_eyes

        do_step = (not self.paused) or self.step_once
        if do_step:
            self.push_snapshot()
            self.board.run()
            self.step_once = False

        # render
        self.world.fill("white")
        self.draw_food()
        # self.draw_spikes()  # inutile si spikes=0, mais tu peux laisser si tu veux
        self.draw_boules()

        scaled = pygame.transform.scale(self.world, self.screen.get_size())
        self.screen.blit(scaled, (0, 0))

        if overlay_lines:
            self.overlay_text(overlay_lines)

        pygame.display.flip()
        self.clock.tick(self.fps)

        return keys


def board_signature(board):
    foods = [(f.x, f.y) for f in board.foods[:3]]
    return hash(tuple(foods))


def attach_policy_to_current_board(board, policy, stochastic=False):
    if board.boules:
        board.boules[0].set_pilot(BouleNNPilot(board.boules[0], policy, stochastic=stochastic))


def make_board(policy=None, stochastic=False):
    seed = random.randrange(0, 2**31 - 1)
    board = create_sim_test_nn(
        500, 500,
        nombre_spikes=0,   # ✅ AUCUN spike
        nombre_food=12,
        nombre_boule=1,
        seed=seed,
        actor_model=None,
    )
    print("NEW MAP seed=", getattr(board, "_seed", seed), "sig=", board_signature(board))

    if policy is not None and board.boules:
        attach_policy_to_current_board(board, policy, stochastic=stochastic)

    return board


def set_caption(files, idx, stochastic):
    name = os.path.basename(files[idx]) if files else "NO_MODEL"
    mode = "sample" if stochastic else "mean"
    pygame.display.set_caption(
        f"Boule NN (FOOD ONLY) | [{idx+1}/{len(files)}] {name} | mode={mode} | ←/→ change | SHIFT+←/→ change+reset | R reset | SPACE pause | N step | B back"
    )


# ----------------------- Main -----------------------
if __name__ == "__main__":
    device = "cpu"
    files = list_checkpoints()

    if not files:
        print("❌ Aucun modèle trouvé (ni checkpoints/*.pt ni policy.pt).")
        raise SystemExit(1)

    idx = 0
    stochastic = False

    policy = load_policy_any(files[idx], device=device)
    print("✅ chargé:", files[idx])

    board = make_board(policy=policy, stochastic=stochastic)
    game = Game(1000, 1000, board)
    set_caption(files, idx, stochastic)

    while game.running:
        boule = game.board.boules[0]
        remaining_food = sum(1 for f in game.board.foods if not f.get_eaten())
        overlay = [
            f"model: {os.path.basename(files[idx])}  ({idx+1}/{len(files)})",
            f"mode: {'sample' if stochastic else 'mean'}   paused: {game.paused}",
            f"pos: ({boule.x:.1f},{boule.y:.1f})  angle:{boule.angle:.1f}  energy:{boule.energy}",
            f"food remaining: {remaining_food}   dead:{boule.is_dead()}",
            "keys: ←/→ change | SHIFT+←/→ change+reset | R reset | S stochastic | SPACE pause | N step | B back | E eyes | X spike-eyes",
        ]

        keys = game.step_and_render(overlay_lines=overlay)

        # Reset map
        if pygame.K_r in keys:
            game.board = make_board(policy=policy, stochastic=stochastic)
            game.world = pygame.Surface((game.board.width, game.board.height))
            game.clear_history()

        # Toggle stochastic
        if pygame.K_s in keys:
            stochastic = not stochastic
            attach_policy_to_current_board(game.board, policy, stochastic=stochastic)
            set_caption(files, idx, stochastic)

        mods = pygame.key.get_mods()
        shift = (mods & pygame.KMOD_SHIFT) != 0

        # Previous model
        if pygame.K_LEFT in keys:
            idx = (idx - 1) % len(files)
            policy = load_policy_any(files[idx], device=device)
            print("✅ chargé:", files[idx])

            if shift:
                game.board = make_board(policy=policy, stochastic=stochastic)
                game.world = pygame.Surface((game.board.width, game.board.height))
                game.clear_history()
            else:
                attach_policy_to_current_board(game.board, policy, stochastic=stochastic)
                game.clear_history()

            set_caption(files, idx, stochastic)

        # Next model
        if pygame.K_RIGHT in keys:
            idx = (idx + 1) % len(files)
            policy = load_policy_any(files[idx], device=device)
            print(" chargé:", files[idx])

            if shift:
                game.board = make_board(policy=policy, stochastic=stochastic)
                game.world = pygame.Surface((game.board.width, game.board.height))
                game.clear_history()
            else:
                attach_policy_to_current_board(game.board, policy, stochastic=stochastic)
                game.clear_history()

            set_caption(files, idx, stochastic)