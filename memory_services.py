class MemoryService:
    
    def forget_seen_neighbors(self, agent,t ,forget_frequency):
        if t % forget_frequency == 0:
            agent.seen_neighbors=agent.seen_neighbors[1:]
    def update_threshold(self, agent, other_agent):
        agent.communication_threshold = (other_agent.comunication_threshold + agent.communication_threshold)/2
    def forget_threshold(self, agent):
        agent.communication_threshold *= agent.communication_threshold_decay