from agent import Role, DecoyAgent, Status, Position
import numpy as np
import torch 
import logging
from utils import in_circle

def gradient_step(function,x,y,step_size):
    grad_x = torch.autograd.grad(function,x,retain_graph=True)[0]
    grad_y = torch.autograd.grad(function,y)[0]
    x = x - step_size*grad_x
    y = y - step_size*grad_y
    return x,y

class MovementService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def move(self,agent,agents,simulation_t):
        if agent.role == Role.EXPLORER:
            self.estimate_explorer_movement(agent)
        if agent.role == Role.COMMUNICATOR:
            if simulation_t<3:
                self.estimate_communicator_movement(agent, agents)
        
    def repulsive_movement(self, agent, positions):
        return sum([agent.x-position.x for position in positions]),sum([agent.y-position.y for position in positions])
    def attractive_movement(self, agent, positions):
        return sum([position.x-agent.x for position in positions]),sum([position.y-agent.y for position in positions])
    def movement_agent_explorer_in_centroid(self,agent,time_index):
        noise = np.random.randn()*agent.sigma
        attraction_x_centroid,attraction_y_centroid = self.attractive_movement(agent, [agent.centroids[time_index[0]]])
        repulsive_x_past,repulsive_y_past = self.repulsive_movement(agent,agent.stored_positions)
        new_x = agent.x + agent.kappa*(attraction_x_centroid+repulsive_x_past)+noise
        new_y = agent.y + agent.kappa*(attraction_y_centroid+repulsive_y_past)+noise
        return new_x,new_y
    def apply_speed_constraints(self,agent,new_x,new_y):
        constraints = [self.max_speed_constraint_condition(agent,agent.x,agent.y,new_x,new_y)]
        new_x = torch.tensor([new_x],requires_grad=True,dtype=torch.float)
        new_y = torch.tensor([new_y],requires_grad=True,dtype=torch.float)
        prev_x = torch.tensor([agent.x],requires_grad=True,dtype=torch.float)
        prev_y = torch.tensor([agent.y],requires_grad=True,dtype=torch.float)
        v_max = torch.tensor([agent.v_max],requires_grad=True,dtype=torch.float)
        while not all(constraints):                                        
            constraint_functions = torch.nn.functional.softplus(torch.stack([self.max_speed_constraint_function(prev_x,prev_y,new_x,new_y,v_max)])).sum()
            new_x,new_y = gradient_step(constraint_functions,new_x,new_y,0.1)
            constraints = [self.max_speed_constraint_condition(agent,agent.x,agent.y,new_x.item(),new_y.item())]
        return new_x.item(),new_y.item()
    
    def estimate_explorer_movement(self,agent):
        if in_circle(agent.x,agent.y,agent.exploration_centroid.x,agent.exploration_centroid.y,int(agent.exploration_buffer_radius*1.5)) and agent.load_state:
            t = agent.t
            time_index = [i for i in range(len(agent.centroids_schedule)) if agent.centroids_schedule[i]>t]
            if len(time_index) == 0:
                agent.finished_exploring = True
                return
            if in_circle(agent.x,agent.y,agent.centroids[time_index[0]].x,agent.centroids[time_index[0]].y,agent.sensing_radius/2):
                agent.centroids = agent.centroids[1:]
                agent.centroids_schedule = agent.centroids_schedule[1:]
                return 
            new_x, new_y = self.movement_agent_explorer_in_centroid(agent,time_index)
            new_x,new_y = self.apply_speed_constraints(agent,new_x,new_y)
            agent.t += 1
            
        else:
            if agent.load_state:
                attraction_x,attraction_y = self.attractive_movement(agent, [agent.exploration_centroid])
                new_x = agent.x + agent.kappa*(attraction_x)
                new_y = agent.y + agent.kappa*(attraction_y)
            else:
                noise = np.random.randn()*agent.sigma
                new_x = agent.x + agent.kappa*(-agent.x)+noise
                new_y = agent.y + agent.kappa*(-agent.y)+noise                
            new_x,new_y = self.apply_speed_constraints(agent,new_x,new_y)
            if not agent.load_state and in_circle(new_x,new_y,0,0,agent.nest_radius) and agent.communication_threshold == 0 and agent.finished_exploring == True:
                print(agent.id,agent.exploration_centroid.x,agent.exploration_centroid.y)
                agent.role = Role.COMMUNICATOR
                input("Now i am communicatr ")    
        agent.stored_positions.append(Position(agent.x,agent.y))
        agent.x = np.round(new_x,2)
        agent.y = np.round(new_y,2)
        if agent.id == 0:
            print(agent.x,agent.y)
            input('hipo')

    def movement_agent_communicator(self,agent,agents):
        nearby_agents = self.get_neighbors(agent,agents)
        seen_neighbors = [agent for agent in nearby_agents if agent in agent.seen_neighbors]
        not_seen_neighbors = [agent for agent in nearby_agents if agent not in agent.seen_neighbors]
        attraction_x,attraction_y = self.attractive_movement(agent,not_seen_neighbors)
        repulsive_neighbors_x,repulsive_neighbors_y = self.repulsive_movement(agent,seen_neighbors)
        repulsive_past_x, repulsive_past_y = self.repulsive_movement(agent,agent.stored_positions)
        noise = np.random.randn()*agent.sigma
        new_x = agent.x + agent.kappa*(attraction_x+repulsive_neighbors_x+repulsive_past_x)+noise
        new_y = agent.y + agent.kappa*(attraction_y+repulsive_neighbors_y+repulsive_past_y)+noise
        return new_x,new_y
    def apply_communicator_constraints(self,agent,new_x,new_y):
        constraints = [self.max_speed_constraint_condition(agent,agent.x,agent.y,new_x,new_y),self.communication_movement_condition(agent,new_x,new_y)]            
        new_x = torch.tensor([new_x],requires_grad=True,dtype=torch.float)
        new_y = torch.tensor([new_y],requires_grad=True,dtype=torch.float)
        prev_x = torch.tensor([agent.x],requires_grad=True,dtype=torch.float)
        prev_y = torch.tensor([agent.y],requires_grad=True,dtype=torch.float)
        v_max = torch.tensor([agent.v_max],requires_grad=True,dtype=torch.float)
        nest_radius = torch.tensor([agent.nest_radius],requires_grad=True,dtype=torch.float)
        buffer_radius = torch.tensor([agent.exploration_buffer_radius],requires_grad=True,dtype=torch.float)
        while not all(constraints):                                        
            constraint_functions = torch.nn.functional.softplus(torch.stack([self.max_speed_constraint_function(prev_x,prev_y,new_x,new_y,v_max),
                                                self.communication_movement_function(new_x,new_y,nest_radius+buffer_radius)])).sum()
            new_x,new_y = gradient_step(constraint_functions,new_x,new_y,0.1)
            constraints = [self.max_speed_constraint_condition(agent,agent.x,agent.y,new_x.item(),new_y.item()),
                            self.communication_movement_condition(agent,new_x.item(),new_y.item())]
        return new_x.item(),new_y.item()
    def estimate_communicator_movement(self,agent,agents):
        # print("communicator movement",agent.id)
        new_x,new_y = self.movement_agent_communicator(agent,agents)
        # print("here got movement")
        # print(new_x,new_y)
        new_x,new_y = self.apply_communicator_constraints(agent,new_x,new_y)
        # print(new_x,new_y)
        # print('here got constraints')
        agent.stored_positions.append(Position(agent.x,agent.y))
        agent.x = np.round(new_x,2)
        agent.y = np.round(new_y,2)
        
        # input('hipo')
        
    def max_speed_constraint_condition(self,agent, prev_x,prev_y,new_x,new_y):
        return ((new_x-prev_x)**2+(new_y-prev_y)**2)**(1/2) <= agent.v_max
    def exploration_movement_condition(self,agent,new_x,new_y):
        return new_x**2 + new_y**2 >= (agent.nest_radius+agent.exploration_buffer_radius)**2 
    def communication_movement_condition(self,agent,new_x,new_y):
        return new_x**2 + new_y**2 <= (agent.nest_radius+agent.exploration_buffer_radius)**2
    def max_speed_constraint_function(self, prev_x,prev_y,new_x,new_y,v_max):
        return (new_x-prev_x)**2+(new_y-prev_y)**2-v_max**2
    def exploration_movement_function(self,new_x,new_y,radius):
        return -new_x**2 - new_y**2+(radius)**2 
    def communication_movement_function(self,new_x,new_y,radius):
        return new_x**2 + new_y**2-(radius)**2
    def get_neighbors(self, agent, agents):
        return [
            other for other in agents if agent.id != other.id and in_circle(agent.x, agent.y, other.x, other.y, agent.sensing_radius)
        ]