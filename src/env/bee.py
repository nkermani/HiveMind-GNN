import numpy as np
from enum import Enum
from typing import List


class BeeState(Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    COLLECTING = "collecting"
    RETURNING = "returning"


class Bee:
    def __init__(self, bee_id: int, start_node: int):
        self.bee_id = bee_id
        self.current_node = start_node
        self.state = BeeState.IDLE
        self.steps_taken = 0
        self.total_distance = 0.0
        self.nectar_collected = 0.0
        self.path_history = [start_node]

    def move_to(self, node: int, distance: float):
        self.current_node = node
        self.steps_taken += 1
        self.total_distance += distance
        self.state = BeeState.NAVIGATING
        self.path_history.append(node)

    def collect_nectar(self, amount: float):
        self.nectar_collected += amount
        self.state = BeeState.COLLECTING
