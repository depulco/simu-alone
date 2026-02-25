import random
import pygame
import torch

from create_simulation_fixed import create_sim_test_nn


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


class Actor(torch.nn.Module):
    def __init__(self, state_dim=12, action_dim=3, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden),
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


def load_policy(path="policy.pt", device="cpu"):
    policy = Actor()
    policy.load_state_dict(torch.load(path, map_location=device))
    policy.eval()
    return policy


class Game:
    def __init__(self, win_w, win_h, board):
        pygame.init()
        self.board = board
        self.world = pygame.Surface((board.width, board.height))
        self.screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
        pygame.display.set_caption("Boule NN - Random map (E=eyes, R=reset)")
        self.clock = pygame.time.Clock()
        self.running = True
        self.show_eyes = True

    def draw_spikes(self):
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

            for i, eye in enumerate(boule.get_food_eyes()):
                color = "orange" if boule.saw_by_food_eyes[i] != 1 else "green"
                pygame.draw.line(self.world, color, (boule.x, boule.y), eye.get_end_sight())

            for i, eye in enumerate(boule.get_spike_eyes()):
                color = "purple" if boule.saw_by_spike_eyes[i] != 1 else "yellow"
                pygame.draw.line(self.world, color, (boule.x, boule.y), eye.get_end_sight())

    def handle_events(self):
        keys = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                keys.append(event.key)
        return keys

    def step_and_render(self):
        keys = self.handle_events()

        if pygame.K_ESCAPE in keys:
            self.running = False
        if pygame.K_e in keys:
            self.show_eyes = not self.show_eyes

        self.board.run()

        self.world.fill("white")
        self.draw_food()
        self.draw_spikes()
        self.draw_boules()

        scaled = pygame.transform.scale(self.world, self.screen.get_size())
        self.screen.blit(scaled, (0, 0))
        pygame.display.flip()
        self.clock.tick(120)

        return keys


def board_signature(board):
    spikes = [(s.x, s.y) for s in board.spikes]
    foods = [(f.x, f.y) for f in board.foods[:3]]
    return hash((tuple(spikes), tuple(foods)))


def make_board(policy=None):
    # ✅ seed vraiment random à chaque fois
    seed = random.randrange(0, 2**31 - 1)

    board = create_sim_test_nn(
        500, 500,
        nombre_spikes=5,
        nombre_food=12,
        nombre_boule=1,
        seed=seed,
        actor_model=None,
    )

    print("NEW MAP seed=", getattr(board, "_seed", seed), "sig=", board_signature(board))

    if policy is not None and board.boules:
        board.boules[0].set_pilot(BouleNNPilot(board.boules[0], policy, stochastic=False))

    return board


if __name__ == "__main__":
    policy = None
    try:
        policy = load_policy("policy.pt", device="cpu")
        print("✅ policy.pt chargé")
    except FileNotFoundError:
        print("❌ policy.pt introuvable -> lance d'abord : python ai_boule_fixed.py")
    except Exception as e:
        print("❌ Erreur chargement policy:", e)

    board = make_board(policy=policy)
    game = Game(1000, 1000, board)

    while game.running:
        keys = game.step_and_render()

        if pygame.K_r in keys:
            game.board = make_board(policy=policy)
            game.world = pygame.Surface((game.board.width, game.board.height))