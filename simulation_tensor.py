import random
import torch


class BoardTensor:
    def __init__(
        self,
        width: int,
        height: int,
        nb_food: int,
        nb_spikes: int,
        device="cuda",
        boule_radius: float = 10.0,
        food_radius: float = 10.0,
        spike_radius: float = 10.0,
        eye_length: float = 140.0,
        n_food_eyes: int = 8,
        n_spike_eyes: int = 8,
    ):
        self.width = float(width)
        self.height = float(height)
        self.nb_food = nb_food
        self.nb_spikes = nb_spikes
        self.device = torch.device(device)

        self.boule_radius = boule_radius
        self.food_radius = food_radius
        self.spike_radius = spike_radius
        self.eye_length = eye_length

        self.n_food_eyes = n_food_eyes
        self.n_spike_eyes = n_spike_eyes

        self.food_eye_offsets = torch.linspace(
            0, 360 - 360 / n_food_eyes, n_food_eyes, device=self.device
        )
        self.spike_eye_offsets = torch.linspace(
            4, 4 + 360 - 360 / n_spike_eyes, n_spike_eyes, device=self.device
        )

        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.boule_pos = torch.tensor(
            [[self.width * 0.5, self.height * 0.5]],
            dtype=torch.float32,
            device=self.device,
        )
        self.boule_angle = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        self.boule_energy = torch.tensor([600.0], dtype=torch.float32, device=self.device)
        self.boule_dead = torch.tensor([False], dtype=torch.bool, device=self.device)

        self.food_pos = torch.empty((self.nb_food, 2), dtype=torch.float32, device=self.device)
        self.food_pos[:, 0] = torch.rand(self.nb_food, device=self.device) * (self.width - 40) + 20
        self.food_pos[:, 1] = torch.rand(self.nb_food, device=self.device) * (self.height - 40) + 20
        self.food_alive = torch.ones(self.nb_food, dtype=torch.bool, device=self.device)

        self.spike_pos = torch.empty((self.nb_spikes, 2), dtype=torch.float32, device=self.device)
        if self.nb_spikes > 0:
            self.spike_pos[:, 0] = torch.rand(self.nb_spikes, device=self.device) * (self.width - 40) + 20
            self.spike_pos[:, 1] = torch.rand(self.nb_spikes, device=self.device) * (self.height - 40) + 20

        self.spike_vel = torch.empty((self.nb_spikes, 2), dtype=torch.float32, device=self.device)
        if self.nb_spikes > 0:
            self.spike_vel[:, 0] = torch.randint(0, 2, (self.nb_spikes,), device=self.device).float() * 2 - 1
            self.spike_vel[:, 1] = torch.randint(0, 2, (self.nb_spikes,), device=self.device).float() * 2 - 1

        return self.get_nn_input()

    def _move_boule(self, action, speed=2.0, rot_speed=3.0):
        if self.boule_dead[0]:
            return

        a = torch.tanh(action)
        dx = a[0] * speed
        dy = a[1] * speed
        drot = a[2] * rot_speed

        new_x = self.boule_pos[0, 0] + dx
        new_y = self.boule_pos[0, 1] + dy

        x_ok = (new_x - self.boule_radius >= 0) & (new_x + self.boule_radius <= self.width)
        y_ok = (new_y - self.boule_radius >= 0) & (new_y + self.boule_radius <= self.height)

        self.boule_pos[0, 0] = torch.where(x_ok, new_x, self.boule_pos[0, 0])
        self.boule_pos[0, 1] = torch.where(y_ok, new_y, self.boule_pos[0, 1])
        self.boule_angle[0] = torch.remainder(self.boule_angle[0] + drot, 360.0)

    def _move_spikes(self):
        if self.nb_spikes == 0:
            return

        new_pos = self.spike_pos + self.spike_vel

        hit_left = new_pos[:, 0] - self.spike_radius < 0
        hit_right = new_pos[:, 0] + self.spike_radius > self.width
        hit_top = new_pos[:, 1] - self.spike_radius < 0
        hit_bottom = new_pos[:, 1] + self.spike_radius > self.height

        self.spike_vel[hit_left | hit_right, 0] *= -1
        self.spike_vel[hit_top | hit_bottom, 1] *= -1

        self.spike_pos += self.spike_vel
        self.spike_pos[:, 0].clamp_(self.spike_radius, self.width - self.spike_radius)
        self.spike_pos[:, 1].clamp_(self.spike_radius, self.height - self.spike_radius)

    def _food_collisions(self):
        if self.nb_food == 0 or self.boule_dead[0]:
            return torch.tensor(0, device=self.device)

        d = self.food_pos - self.boule_pos[0]
        dist2 = (d * d).sum(dim=1)
        collide = dist2 < (self.boule_radius + self.food_radius) ** 2
        eat_mask = collide & self.food_alive

        n_eaten = eat_mask.sum()
        self.food_alive[eat_mask] = False
        self.boule_energy[0] += 300.0 * n_eaten.float()
        return n_eaten

    def _spike_collisions(self):
        if self.nb_spikes == 0 or self.boule_dead[0]:
            return

        d = self.spike_pos - self.boule_pos[0]
        dist2 = (d * d).sum(dim=1)
        collide = dist2 < (self.boule_radius + self.spike_radius) ** 2
        if collide.any():
            self.boule_dead[0] = True

    def _eye_values_for_objects(self, obj_pos, obj_alive, eye_offsets_deg):
        n_eyes = eye_offsets_deg.numel()
        out = torch.ones(n_eyes, dtype=torch.float32, device=self.device)

        if obj_pos.numel() == 0:
            return out

        pos = self.boule_pos[0]
        base_angle = self.boule_angle[0]

        alive_pos = obj_pos[obj_alive] if obj_alive is not None else obj_pos
        if alive_pos.numel() == 0:
            return out

        rel = alive_pos - pos
        dist = torch.norm(rel, dim=1).clamp_min(1e-6)

        obj_ang = torch.rad2deg(torch.atan2(rel[:, 1], rel[:, 0]))
        obj_ang = torch.remainder(obj_ang, 360.0)

        eye_ang = torch.remainder(base_angle + eye_offsets_deg, 360.0)

        diff = torch.abs(obj_ang[None, :] - eye_ang[:, None])
        diff = torch.minimum(diff, 360.0 - diff)

        half_cone = 360.0 / (2.0 * n_eyes)
        visible = (diff <= half_cone) & (dist[None, :] <= self.eye_length)

        norm_dist = (dist / self.eye_length).clamp(max=1.0)

        for e in range(n_eyes):
            mask = visible[e]
            if mask.any():
                out[e] = norm_dist[mask].min()

        return out

    def get_nn_input(self):
        food_eyes = self._eye_values_for_objects(
            self.food_pos, self.food_alive, self.food_eye_offsets
        )
        spike_eyes = self._eye_values_for_objects(
            self.spike_pos, None, self.spike_eye_offsets
        )

        x = self.boule_pos[0, 0] / self.width
        y = self.boule_pos[0, 1] / self.height
        angle = self.boule_angle[0] / 360.0
        energy = self.boule_energy[0] / 1000.0

        return torch.cat([torch.stack([x, y, angle, energy]), spike_eyes, food_eyes])

    def step(self, action, speed=2.0, rot_speed=3.0):
        e0 = self.boule_energy[0].clone()

        if not self.boule_dead[0]:
            self.boule_energy[0] -= 1.0
            self._move_boule(action, speed=speed, rot_speed=rot_speed)
            self._move_spikes()
            eaten = self._food_collisions()
            self._spike_collisions()
        else:
            eaten = torch.tensor(0, device=self.device)

        if self.boule_energy[0].item() <= 0.0:
            self.boule_dead[0] = True

        done = bool(self.boule_dead[0].item())

        e1 = self.boule_energy[0]
        energy_gain = torch.clamp(e1 - e0, min=0.0)
        eat_reward = 0.03 * energy_gain
        eat_count_reward = eaten.float()
        time_pen = torch.tensor(-0.005, device=self.device)
        act_pen = 0.001 * (torch.tanh(action) ** 2).sum()

        reward = time_pen + eat_reward + eat_count_reward - act_pen
        if self.boule_dead[0]:
            reward = reward - 40.0

        return self.get_nn_input(), float(reward.item()), done, {}