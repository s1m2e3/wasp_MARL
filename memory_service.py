class MemoryService:
    def store_position(self, agent, x, y):
        agent.x = x
        agent.y = y
    def store_communication(self, agent, other_agent):
        agent.seen_neighbors.append(other_agent)
    def forget_communication(self, agent):
        agent.seen_neighbors=agent.seen_neighbors[1:]
    def update_threshold(self, agent, other_agent):
        agent.communication_threshold = (other_agent.comunication_threshold + agent.communication_threshold)/2
    def forget_threshold(self, agent):
        agent.communication_threshold *= agent.communication_threshold_decay