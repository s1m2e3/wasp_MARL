from enum import Enum
from dataclasses import dataclass, field

class Role(Enum):
    EXPLORER = 1
    COMMUNICATOR = 2
    RESCUE = 3
class Status(Enum):
    ENEMY = 1
    FOUND = 2
    EXPLORED = 3

@dataclass (frozen=False)
class LostAgent:
    id: int
    x: float
    y: float
    num_agents: int

@dataclass (frozen=False)
class DecoyAgent:
    id: int
    x: int
    y: str
    role: Status 

@dataclass (frozen=True)
class Position:
    x: float
    y: float

@dataclass (frozen=False)
class Agent:
    role: Role
    id: int
    x: float
    y: float
    c: float
    communication_threshold: int
    centroids: list = field(default_factory=list)
    centroids_schedule: list = field(default_factory=list)
    v_max: float = 0
    nest_radius: float = 0
    kappa: float = 0
    sigma: float = 0
    sensing_radius: float = 0
    load_state: bool = False
    found_state: bool = False
    found_location: Position = Position(0,0)
    exploration_centroid: Position = Position(0,0)
    exploration_radius: float = 0
    t: int = 0
    seen_neighbors: list = field(default_factory=list)
    communication_threshold_decay: float = 0.9
    exploration_buffer_radius: float = 2
    forget_last: int=0
    stored_messages: dict = field(default_factory=dict)
    stored_positions: list = field(default_factory=list)
    finished_exploring: bool = False
