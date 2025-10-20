from agent import Role

def in_circle(x1,y1,x2,y2,radius):
    return ((x1-x2)**2 + (y1-y2)**2) <= radius**2

class PlanningService:
    def move(self,agent,grid):
        if agent.role == Role.EXPLORER:
            new_x,new_y = self.estimate_explorer_movement(agent,grid)
        elif agent.role == Role.COMMUNICATOR:
            new_x,new_y = self.estimate_communicator_movement(agent,grid)
        pass
    def estimate_explorer_movement(self,agent,grid):
        if in_circle(agent.x,agent.y,agent.exploration_centroid[0],agent.exploration_centroid[0],agent.exploration_radius):
        else:
            
    def estimate_communicator_movement(self,agent,grid):

    def constraint_movement(self,agent, prev_x,prev_y,next_x,next_y):
        pass
