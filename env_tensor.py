from dataclasses import dataclass
import random
import torch
from simulation_tensor import BoardTensor


@dataclass
class EnvConfigTensor:
    width: int = 500
    height: int = 500
    max_steps: int = 600
    speed: float = 2.0
    rot_speed: float = 3.0
    nb_spikes: int = 10
    nb_food: int = 20
    device: str = "cuda"


class BouleEnvTensor:
    def __init__(self, cfg: EnvConfigTensor, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.rng = random.Random(seed)
        self.board = None
        self.steps = 0

    def reset(self):
        self.board = BoardTensor(
            width=self.cfg.width,
            height=self.cfg.height,
            nb_food=self.cfg.nb_food,
            nb_spikes=self.cfg.nb_spikes,
            device=self.cfg.device,
        )
        self.steps = 0
        return self.board.reset(seed=self.seed + self.rng.randint(0, 10_000))

    def step(self, action: torch.Tensor):
        s, r, done, info = self.board.step(
            action,
            speed=self.cfg.speed,
            rot_speed=self.cfg.rot_speed,
        )
        self.steps += 1
        if self.steps >= self.cfg.max_steps:
            done = True
        return s, r, done, info