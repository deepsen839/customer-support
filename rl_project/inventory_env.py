import numpy as np

class InventoryEnv:
    def __init__(self, horizon=120, max_inventory=60, max_order=20,
                 seed=0, lost_sales_penalty=8.0):
        self.horizon = int(horizon)
        self.max_inventory = int(max_inventory)
        self.max_order = int(max_order)
        self.lost_sales_penalty = float(lost_sales_penalty)
        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    @property
    def obs_dim(self):
        return 4

    @property
    def action_dim(self):
        return self.max_order + 1

    def _obs(self):
        phase = 2 * np.pi * self.day / 30.0
        return np.array([
            self.inventory / self.max_inventory,
            self.last_demand / 15.0,
            np.sin(phase),
            np.cos(phase)
        ], dtype=np.float32)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.day = 0
        self.inventory = 25.0
        self.last_demand = 5.0
        self.total_cost = 0.0
        return self._obs()

    def step(self, action):
        requested = int(np.clip(action, 0, self.max_order))
        accepted = min(requested, self.max_inventory - self.inventory)
        self.inventory += accepted

        seasonal_mean = 7.0 + 2.0 * np.sin(2 * np.pi * self.day / 30.0)
        demand = float(self.rng.poisson(max(1.0, seasonal_mean)))
        sold = min(self.inventory, demand)
        lost = max(0.0, demand - sold)
        self.inventory -= sold

        holding_cost = 0.25 * self.inventory
        order_cost = accepted + (4.0 if accepted > 0 else 0.0)
        lost_cost = self.lost_sales_penalty * lost
        cost = holding_cost + order_cost + lost_cost
        self.total_cost += cost

        self.last_demand = demand
        self.day += 1
        done = self.day >= self.horizon
        info = {
            "demand": demand, "sold": sold, "lost_sales": lost,
            "order": accepted, "cost": cost,
            "holding_cost": holding_cost, "order_cost": order_cost,
            "lost_cost": lost_cost
        }
        return self._obs(), -float(cost), done, info
