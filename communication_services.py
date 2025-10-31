from utils import in_circle
from agent import Role, Position
import logging

class CommunicationService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def sense(self,agent,agents,missing_agents,t):
        for other_agent in agents:
            if agent.id != other_agent.id:
                if in_circle(agent.x, agent.y, other_agent.x, other_agent.y, agent.sensing_radius):
                    agent.seen_neighbors.append(other_agent)
                    self.generate_communication_message(agent,other_agent,t)
        for missing_agent in missing_agents:
            if in_circle(agent.x, agent.y, missing_agent.x, missing_agent.y, agent.sensing_radius) and not agent.found_state:
                agent.found_state = True
                agent.communication_threshold = -missing_agent.num_agents*100
                agent.role = Role.RESCUE
                agent.found_location = Position(missing_agent.x,missing_agent.y)
        
    def generate_communication_message(self,agent,other_agent,t):
        if t not in agent.stored_messages:
            agent.stored_messages[t]=[]
        if t>30:
            if other_agent.role==Role.COMMUNICATOR and t in other_agent.stored_messages:
                agent.stored_messages[t].extend(other_agent.stored_messages[t])
            elif other_agent.role==Role.EXPLORER:
                agent.stored_messages[t].append(str(other_agent.id)+' '+str(other_agent.x)+' '+str(other_agent.y)+' '+str(t))
        if other_agent.communication_threshold >1 and agent.role == Role.COMMUNICATOR and other_agent.role == Role.EXPLORER:
            agent.communication_threshold += 100
            other_agent.communication_threshold -= 100
            print("communication from explorer to communicator")
        if agent.role == Role.COMMUNICATOR and other_agent.role == Role.RESCUE and other_agent.communication_threshold <-1:
            agent.communication_threshold -= 100
            other_agent.communication_threshold += 100