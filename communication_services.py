from utils import in_circle
from agent import Role
class CommunicationService:
    def sense(self,agent,agents,t):
        for other_agent in agents:
            if agent.id != other_agent.id:
                if in_circle(agent.x, agent.y, other_agent.x, other_agent.y, agent.sensing_radius):
                    agent.seen_neighbors.append(other_agent)
                    self.generate_communication_message(agent,other_agent,t)
    def generate_communication_message(self,agent,other_agent,t):
        if t not in agent.stored_messages:
            agent.stored_messages[t]=[]
        if other_agent.role==Role.COMMUNICATOR and t in other_agent.stored_messages:
            agent.stored_messages[t].extend(other_agent.stored_messages[t])
        elif other_agent.role==Role.EXPLORER:
            agent.stored_messages[t].append(str(other_agent.id)+' '+str(other_agent.x)+' '+str(other_agent.y)+' '+str(t))
            