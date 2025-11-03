from agent import Role
import logging

class MemoryService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def forget_seen_neighbors(self, agent):
        if agent.seen_neighbors:
            first_key = next(iter(agent.seen_neighbors))
            del agent.seen_neighbors[first_key]
            
    def forget_stored_positions(self,agent):
        if agent.stored_positions:
            agent.stored_positions = agent.stored_positions[1:] 
        
    def update_roles(self,agent):
        if agent.communication_threshold > 1 and agent.role == Role.COMMUNICATOR:
            if agent.communication_threshold == 100:
                agent.communication_threshold = 0
                agent.load_state = True
                agent.role = Role.EXPLORER
        if agent.communication_threshold < -1 and agent.role == Role.COMMUNICATOR:
            if agent.communication_threshold == -100:
                agent.communication_threshold = 0
                agent.role = Role.RESCUE
                agent.follower = True
        if abs(agent.communication_threshold)<1 and agent.role == Role.EXPLORER and not agent.finished_exploring:
            agent.load_state = True
    def update_leader(self,agent):
        if agent.role == Role.RESCUE and not agent.follower and not agent.updated and agent.communication_threshold==0:
            agent.kappa = agent.kappa/3
            agent.v_max = agent.v_max/2
            agent.updated = True
    