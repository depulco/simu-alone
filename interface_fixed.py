import os
import glob
import math
import random
import pygame
import torch

from env_tensor import EnvConfigTensor, BouleEnvTensor
from ai_boule_tensor import Actor


def list_checkpoints():
    files = []
    if os.path.isdir("checkpoints"):
        files += sorted(glob.glob("checkpoints/*.pt"))
    if os.path.exists("policy.pt"):
        files.append("policy.pt")
    return files


def load_policy_any(path, device="cpu"):
    policy = Actor(state_dim=20, action_dim=3).to(device)
    obj = torch.load(path, map_location=device, weights_only=True)

    if isinstance(obj, dict) and "actor" in obj:
        policy.load_state_dict(obj["actor"])
    else:
        policy.load_state_dict(obj)

    policy.eval()
    return policy


class Game:
    def __init__(self, env, policy, files, idx, device="cpu"):
        pygame.init()
        pygame.font.init()

        self.env = env
        self.policy = policy
        self.files = files
        self.idx = idx
        self.device = device

        self.screen = pygame.display.set_mode((1000, 1000), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)

        self.running = True
        self.paused = False
        self.step_once = False
        self.show_eyes = True
        self.show_spike_eyes = True
        self.fps = 120

        self.state = self.env.reset()
        self.done = False
        self.total_reward = 0.0
        self.steps = 0

        self.set_caption()

    def set_caption(self):
        name = os.path.basename(self.files[self.idx]) if self.files else "NO_MODEL"
        pygame.display.set_caption(
            f"Tensor Watch | [{self.idx+1}/{len(self.files)}] {name} | "
            f"SPACE pause | N step | R reset | E eyes | X spike-eyes | ←/→ model"
        )

    def reset_env(self):
        self.state = self.env.reset()
        self.done = False
        self.total_reward = 0.0
        self.steps = 0

    def change_model(self, delta):
        self.idx = (self.idx + delta) % len(self.files)
        self.policy = load_policy_any(self.files[self.idx], device=self.device)
        print("chargé:", self.files[self.idx])
        self.set_caption()
        self.reset_env()

    def handle_events(self):
        keys = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                keys.append(event.key)
        return keys

    def draw_text(self, lines):
        y = 8
        for line in lines:
            surf = self.font.render(line, True, (0, 0, 0))
            self.screen.blit(surf, (8, y))
            y += 20

    def draw_world(self):
        board = self.env.board

        world_w = int(board.width)
        world_h = int(board.height)
        world = pygame.Surface((world_w, world_h))
        world.fill("white")

        # ---------------- Food
        food_pos = board.food_pos.detach().cpu()
        food_alive = board.food_alive.detach().cpu()

        for i in range(board.nb_food):
            if bool(food_alive[i]):
                x, y = food_pos[i].tolist()
                pygame.draw.circle(
                    world,
                    "green",
                    (int(x), int(y)),
                    int(board.food_radius)
                )

        # ---------------- Spikes
        if board.nb_spikes > 0:
            spike_pos = board.spike_pos.detach().cpu()
            for i in range(board.nb_spikes):
                x, y = spike_pos[i].tolist()
                rect = pygame.Rect(
                    int(x - board.spike_radius * 0.70),
                    int(y - board.spike_radius * 0.70),
                    int(board.spike_radius * 1.41),
                    int(board.spike_radius * 1.41),
                )
                pygame.draw.rect(world, "red", rect)

        # ---------------- Boule
        dead = bool(board.boule_dead[0].item()) or self.done
        boule_pos = board.boule_pos[0].detach().cpu()
        bx, by = boule_pos.tolist()

        if not dead:
            pygame.draw.circle(
                world,
                "blue",
                (int(bx), int(by)),
                int(board.boule_radius)
            )

        # ---------------- Eyes
        if self.show_eyes and not dead:
            angle = float(board.boule_angle[0].item())

            # état courant :
            # [x, y, angle, energy] + spike_eyes(8) + food_eyes(8)
            state_cpu = self.state.detach().cpu()
            spike_eye_vals = state_cpu[4:12].tolist()
            food_eye_vals = state_cpu[12:20].tolist()

            # food eyes
            for i, offset in enumerate(board.food_eye_offsets.detach().cpu().tolist()):
                eye_angle_deg = angle + offset
                eye_angle_rad = math.radians(eye_angle_deg)
                ex = bx + math.cos(eye_angle_rad) * board.eye_length
                ey = by + math.sin(eye_angle_rad) * board.eye_length

                color = "orange" if food_eye_vals[i] < 1.0 else "green"
                pygame.draw.line(world, color, (bx, by), (ex, ey), 1)

            # spike eyes
            if self.show_spike_eyes:
                for i, offset in enumerate(board.spike_eye_offsets.detach().cpu().tolist()):
                    eye_angle_deg = angle + offset
                    eye_angle_rad = math.radians(eye_angle_deg)
                    ex = bx + math.cos(eye_angle_rad) * board.eye_length
                    ey = by + math.sin(eye_angle_rad) * board.eye_length

                    color = "purple" if spike_eye_vals[i] < 1.0 else "yellow"
                    pygame.draw.line(world, color, (bx, by), (ex, ey), )

        scaled = pygame.transform.scale(world, self.screen.get_size())
        self.screen.blit(scaled, (0, 0))

    def step_model(self):
        if self.done:
            return

        with torch.inference_mode():
            dist = self.policy(self.state.to(self.device))
            a = torch.tanh(dist.mean)

        self.state, r, self.done, _ = self.env.step(a)
        self.total_reward += r
        self.steps += 1

    def run(self):
        while self.running:
            keys = self.handle_events()

            if pygame.K_ESCAPE in keys:
                self.running = False
            if pygame.K_SPACE in keys:
                self.paused = not self.paused
            if pygame.K_n in keys:
                self.step_once = True
            if pygame.K_r in keys:
                self.reset_env()
            if pygame.K_e in keys:
                self.show_eyes = not self.show_eyes
            if pygame.K_x in keys:
                self.show_spike_eyes = not self.show_spike_eyes
            if pygame.K_LEFT in keys:
                self.change_model(-1)
            if pygame.K_RIGHT in keys:
                self.change_model(1)

            do_step = (not self.paused) or self.step_once
            if do_step:
                self.step_model()
                self.step_once = False

            self.screen.fill("white")
            self.draw_world()

            board = self.env.board
            remaining_food = int(board.food_alive.sum().item()) if board.nb_food > 0 else 0
            dead = bool(board.boule_dead[0].item()) or self.done
            bx = float(board.boule_pos[0, 0].item())
            by = float(board.boule_pos[0, 1].item())
            angle = float(board.boule_angle[0].item())
            energy = float(board.boule_energy[0].item())

            overlay = [
                f"model: {os.path.basename(self.files[self.idx])} ({self.idx+1}/{len(self.files)})",
                f"device: {self.device}   paused: {self.paused}   done: {self.done}   dead: {dead}",
                f"pos: ({bx:.1f},{by:.1f})  angle:{angle:.1f}  energy:{energy:.1f}",
                f"food remaining: {remaining_food}   return: {self.total_reward:.2f}   steps: {self.steps}",
                "keys: LEFT/RIGHT model | R reset | SPACE pause | N step | E eyes | X spike-eyes | ESC quit",
            ]
            self.draw_text(overlay)

            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()


if __name__ == "__main__":
    device = "cpu"
    files = list_checkpoints()

    if not files:
        print("Aucun modèle trouvé.")
        raise SystemExit(1)

    idx = len(files) - 1
    policy = load_policy_any(files[idx], device=device)

    env = BouleEnvTensor(
        EnvConfigTensor(
            width=500,
            height=500,
            nb_food=12,
            nb_spikes=0,
            max_steps=5000,
            device=device,
        ),
        seed=0,
    )

    game = Game(env, policy, files, idx, device=device)
    game.run()