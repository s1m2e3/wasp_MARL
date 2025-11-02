from agent import Status, DecoyAgent, Role
import logging

class BehaviorService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def drop_decoy_found(self,agent,decoys):
        if agent.found_state and agent.load_state and agent.role == Role.RESCUE:
            decoys.append(DecoyAgent(len(decoys),agent.found_location.x,agent.found_location.y,Status.FOUND))
            agent.load_state = False
        
    def drop_decoy_explored(self,agent,decoys):
        if agent.finished_exploring and agent.load_state and agent.role == Role.EXPLORER:
            decoys.append(DecoyAgent(len(decoys),agent.exploration_centroid.x,agent.exploration_centroid.y,Status.EXPLORED))
            agent.load_state = False
            print(agent.id, agent.load_state, agent.finished_exploring)
            input('droped the beacon')
        