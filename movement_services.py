from agent import Role
import numpy as np
import torch 
def in_circle(x1,y1,x2,y2,radius):
    return ((x1-x2)**2 + (y1-y2)**2) <= radius**2

class MovementService:
    def move(self,agent,agents):
        if agent.role == Role.EXPLORER:
            new_x,new_y = self.estimate_explorer_movement(agent)
        elif agent.role == Role.COMMUNICATOR:
            new_x,new_y = self.estimate_communicator_movement(agent, agents)
        pass
        
    def estimate_explorer_movement(self,agent):
        t = agent.t
        time_index = [i for i in range(len(agent.centroids_schedule)) if agent.centroids_schedule[i]<t == True]
        centroid = agent.centroids[time_index[-1]]
        x_centroid,y_centroid = centroid
        new_x = agent.x + agent.kappa*(x_centroid-agent.x)+agent.sigma*np.random.randn()
        new_y = agent.y + agent.kappa*(y_centroid-agent.y)+agent.sigma*np.random.randn()
        new_x,new_y = self.constraint_movement(agent,agent.x,agent.y,new_x,new_y)
        constraints = [self.max_speed_constraint(agent,agent.x,agent.y,new_x,new_y),self.exploration_movement(agent,new_x,new_y)]            
        if in_circle(agent.x,agent.y,agent.exploration_centroid[0],agent.exploration_centroid[1],agent.exploration_radius):
            new_x = torch.tensor([new_x])
            new_y = torch.tensor([new_y])
            prev_x = torch.tensor([agent.x])
            prev_y = torch.tensor([agent.y])
            v_max = torch.tensor([agent.v_max])
            nest_radius = torch.tensor([agent.nest_radius])
            while any(constraints):                                        
                constraint_functions = torch.softplus(torch.stack([self.max_speed_constraint_function(prev_x,prev_y,new_x,new_y,v_max),
                                                    self.exploration_movement_function(new_x,new_y,nest_radius)])).sum()
                grad = torch.autograd.grad(constraint_functions,torch.stack([new_x,new_y]))[0]
                new_x = new_x - 0.1*grad[0]
                new_y = new_y - 0.1*grad[1]
                constraints = [self.max_speed_constraint(agent,agent.x,agent.y,new_x.to_numpy()[0],new_y.to_numpy()[0]),
                               self.exploration_movement(agent,new_x.to_numpy()[0],new_y.to_numpy()[0])]
        else:
            constraints = constraints[0]
            while any(constraints):
                constraint_functions = torch.softplus(torch.stack([self.max_speed_constraint_function(prev_x,prev_y,new_x,new_y,v_max)])).sum()
                grad = torch.autograd.grad(constraint_functions,torch.stack([new_x,new_y]))[0]
                new_x = new_x - 0.1*grad[0]
                new_y = new_y - 0.1*grad[1]
                constraints = [self.max_speed_constraint(agent,agent.x,agent.y,new_x.to_numpy()[0],new_y.to_numpy()[0])]
        agent.x = new_x.to_numpy()[0]
        agent.y = new_y.to_numpy()[0]
    def estimate_communicator_movement(self,agent,agents):
        t = agent.t
        time_index = [i for i in range(len(agent.centroids_schedule)) if agent.centroids_schedule[i]<t == True]
        centroid = agent.centroids[time_index[-1]]
        x_centroid,y_centroid = centroid
        nearby_agents = self.get_neighbors(agent,agents)
        attraction_agents = [other.x-agent.x if other not in agent.seen_neighbors else -other.x+agent.x for other in nearby_agents ]
        new_x = agent.x + agent.kappa*(x_centroid-agent.x)+agent.sigma*np.random.randn()
        new_y = agent.y + agent.kappa*(y_centroid-agent.y)+agent.sigma*np.random.randn()

        new_x,new_y = self.constraint_movement(agent,agent.x,agent.y,new_x,new_y)
        constraints = [self.max_speed_constraint(agent,agent.x,agent.y,new_x,new_y),self.exploration_movement(agent,new_x,new_y)]            
        if in_circle(agent.x,agent.y,agent.exploration_centroid[0],agent.exploration_centroid[1],agent.exploration_radius):
            new_x = torch.tensor([new_x])
            new_y = torch.tensor([new_y])
            prev_x = torch.tensor([agent.x])
            prev_y = torch.tensor([agent.y])
            v_max = torch.tensor([agent.v_max])
            nest_radius = torch.tensor([agent.nest_radius])
            while any(constraints):                                        
                constraint_functions = torch.softplus(torch.stack([self.max_speed_constraint_function(prev_x,prev_y,new_x,new_y,v_max),
                                                    self.communication_movement_function(new_x,new_y,nest_radius)])).sum()
                grad = torch.autograd.grad(constraint_functions,torch.stack([new_x,new_y]))[0]
                new_x = new_x - 0.1*grad[0]
                new_y = new_y - 0.1*grad[1]
                constraints = [self.max_speed_constraint(agent,agent.x,agent.y,new_x.to_numpy()[0],new_y.to_numpy()[0]),
                               self.exploration_movement(agent,new_x.to_numpy()[0],new_y.to_numpy()[0])]
        
        agent.x = new_x.to_numpy()[0]
        agent.y = new_y.to_numpy()[0]
        

    def max_speed_constraint_condition(self,agent, prev_x,prev_y,new_x,new_y):
        return (new_x-prev_x)**2+(new_y-prev_y)**2 <= agent.v_max**2
    def exploration_movement_condition(self,agent,new_x,new_y):
        return new_x**2 + new_y**2 >= agent.nest_radius**2 
    def communication_movement_condition(self,agent,new_x,new_y):
        return new_x**2 + new_y**2 <= agent.nest_radius**2
    def max_speed_constraint_function(self, prev_x,prev_y,new_x,new_y,v_max):
        return (new_x-prev_x)**2+(new_y-prev_y)**2-v_max**2
    def exploration_movement_function(self,new_x,new_y,nest_radius):
        return -new_x**2 - new_y**2+nest_radius**2 
    def communication_movement_function(self,new_x,new_y,nest_radius):
        return new_x**2 + new_y**2-nest_radius**2
    def get_neighbors(self, agent, agents):
        return [
            other for other in agents if agent.id != other.id and in_circle(agent.x, agent.y, other.x, other.y, agent.sensing_radius)
        ]