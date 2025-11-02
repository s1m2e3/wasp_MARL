from agent import Status, DecoyAgent, Role
import logging
from movement_services import in_circle

class BehaviorService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def drop_decoy_found(self,agent,decoys,radius_decoys):
        if agent.found_state and agent.load_state and agent.role == Role.RESCUE:
            decoys.append(DecoyAgent(len(decoys),agent.found_location.x,agent.found_location.y,Status.FOUND,radius_decoys,agent.num_agents))
            agent.load_state = False
        
    def drop_decoy_explored(self,agent,decoys, radius_decoys):
        if agent.finished_exploring and agent.load_state and agent.role == Role.EXPLORER:
            decoys.append(DecoyAgent(len(decoys),agent.exploration_centroid.x,agent.exploration_centroid.y,Status.EXPLORED,radius_decoys))
            agent.load_state = False
    def saturate_decoys(self,agents,decoys):
        rescue_decoys = [decoy for decoy in decoys if decoy.role == Status.FOUND]
        for decoy in rescue_decoys:
            num_agents = len([agent for agent in agents if in_circle(agent.x,agent.y,decoy.x,decoy.y,decoy.radius) and agent.role == Role.RESCUE])
            if num_agents == decoy.num_agents:
                decoy.role = Status.SATURATED
            