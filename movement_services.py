from agent import Role
import numpy as np
import torch 
def in_circle(x1,y1,x2,y2,radius):
    return ((x1-x2)**2 + (y1-y2)**2) <= radius**2

def gradient_step(function,x,y,step_size):
    grad_x = torch.autograd.grad(function,x,retain_graph=True)[0]
    grad_y = torch.autograd.grad(function,y)[0]
    x = x - step_size*grad_x
    y = y - step_size*grad_y
    return x,y

class MovementService:
    def move(self,agent,agents,simulation_t):
        if agent.role == Role.EXPLORER:
            self.estimate_explorer_movement(agent)
        if simulation_t>5:
            if agent.role == Role.COMMUNICATOR:
                self.estimate_communicator_movement(agent, agents)
        
    def estimate_explorer_movement(self,agent):
        if in_circle(agent.x,agent.y,agent.exploration_centroid[0],agent.exploration_centroid[1],agent.exploration_buffer_radius):
            t = agent.t
            time_index = [i for i in range(len(agent.centroids_schedule)) if agent.centroids_schedule[i]>t]
            centroid = agent.centroids[time_index[0]]
            x_centroid,y_centroid = centroid
            new_x = agent.x + agent.kappa*(x_centroid-agent.x)+agent.sigma*np.random.randn()
            new_y = agent.y + agent.kappa*(y_centroid-agent.y)+agent.sigma*np.random.randn()
            constraints = [self.max_speed_constraint_condition(agent,agent.x,agent.y,new_x,new_y),self.exploration_movement_condition(agent,new_x,new_y)]
            new_x = torch.tensor([new_x],requires_grad=True,dtype=torch.float)
            new_y = torch.tensor([new_y],requires_grad=True,dtype=torch.float)
            prev_x = torch.tensor([agent.x],requires_grad=True,dtype=torch.float)
            prev_y = torch.tensor([agent.y],requires_grad=True,dtype=torch.float)
            v_max = torch.tensor([agent.v_max],requires_grad=True,dtype=torch.float)
            nest_radius = torch.tensor([agent.nest_radius],requires_grad=True,dtype=torch.float)
            while not all(constraints):                                        
                constraint_functions = torch.nn.functional.softplus(torch.stack([self.max_speed_constraint_function(prev_x,prev_y,new_x,new_y,v_max),
                                                    self.exploration_movement_function(new_x,new_y,nest_radius)])).sum()
                new_x,new_y = gradient_step(constraint_functions,new_x,new_y,0.1)
                constraints = [self.max_speed_constraint_condition(agent,agent.x,agent.y,new_x.item(),new_y.item()),
                               self.exploration_movement_condition(agent,new_x.item(),new_y.item())]
            agent.t += 1

        else:
            new_x = agent.x + agent.kappa*(agent.exploration_centroid[0]-agent.x)
            new_y = agent.y + agent.kappa*(agent.exploration_centroid[1]-agent.y)
            constraints = [self.max_speed_constraint_condition(agent,agent.x,agent.y,new_x,new_y)]                
            new_x = torch.tensor([new_x],requires_grad=True,dtype=torch.float)
            new_y = torch.tensor([new_y],requires_grad=True,dtype=torch.float)
            prev_x = torch.tensor([agent.x],requires_grad=True,dtype=torch.float)
            prev_y = torch.tensor([agent.y],requires_grad=True,dtype=torch.float)
            v_max = torch.tensor([agent.v_max],requires_grad=True,dtype=torch.float)
            nest_radius = torch.tensor([agent.nest_radius],requires_grad=True,dtype=torch.float)
            while not any(constraints):
                constraint_functions = torch.nn.functional.softplus(torch.stack([self.max_speed_constraint_function(prev_x,prev_y,new_x,new_y,v_max)])).sum()
                new_x,new_y = gradient_step(constraint_functions,new_x,new_y,0.1)
                constraints = [self.max_speed_constraint_condition(agent,agent.x,agent.y,new_x.item(),new_y.item())]
            new_x= np.round(new_x.item(),2)
            new_y= np.round(new_y.item(),2)
        agent.x = new_x
        agent.y = new_y
        
    def estimate_communicator_movement(self,agent,agents):

        nearby_agents = self.get_neighbors(agent,agents)
        attraction_x = sum([other.x-agent.x if other not in agent.seen_neighbors else -other.x+agent.x for other in nearby_agents])
        attraction_y = sum([other.y-agent.y if other not in agent.seen_neighbors else -other.y+agent.y for other in nearby_agents])
        new_x = agent.x + agent.kappa*attraction_x+agent.sigma*np.random.randn()
        new_y = agent.y + agent.kappa*attraction_y+agent.sigma*np.random.randn()
        constraints = [self.max_speed_constraint_condition(agent,agent.x,agent.y,new_x,new_y),self.communication_movement_condition(agent,new_x,new_y)]            
        new_x = torch.tensor([new_x],requires_grad=True,dtype=torch.float)
        new_y = torch.tensor([new_y],requires_grad=True,dtype=torch.float)
        prev_x = torch.tensor([agent.x],requires_grad=True,dtype=torch.float)
        prev_y = torch.tensor([agent.y],requires_grad=True,dtype=torch.float)
        v_max = torch.tensor([agent.v_max],requires_grad=True,dtype=torch.float)
        nest_radius = torch.tensor([agent.nest_radius],requires_grad=True,dtype=torch.float)
        
        while not all(constraints):                                        
            constraint_functions = torch.nn.functional.softplus(torch.stack([self.max_speed_constraint_function(prev_x,prev_y,new_x,new_y,v_max),
                                                self.communication_movement_function(new_x,new_y,nest_radius)])).sum()
            new_x,new_y = gradient_step(constraint_functions,new_x,new_y,0.1)
            constraints = [self.max_speed_constraint_condition(agent,agent.x,agent.y,new_x.item(),new_y.item()),
                            self.communication_movement_condition(agent,new_x.item(),new_y.item())]
        new_x=np.round(new_x.item(),2)
        new_y=np.round(new_y.item(),2)
        agent.x = new_x.item()
        agent.y = new_y.item()
        
    def max_speed_constraint_condition(self,agent, prev_x,prev_y,new_x,new_y):
        return ((new_x-prev_x)**2+(new_y-prev_y)**2)**(1/2) <= agent.v_max
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