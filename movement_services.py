from agent import Role, DecoyAgent, Status, Position, Speed, Acceleration, Noise
import numpy as np
import torch 
import logging
from utils import in_circle

def gradient_step(function,x,y,step_size):
    grad_x = torch.autograd.grad(function,x,retain_graph=True)[0]
    grad_y = torch.autograd.grad(function,y)[0]
    x = x - step_size*torch.sign(grad_x)*0.2
    y = y - step_size*torch.sign(grad_y)*0.2
    return x,y

class MovementService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def move(self,agent,agents,simulation_t,dt=0.1):
        if agent.role == Role.EXPLORER:
            self.estimate_explorer_movement(agent,dt)
        if agent.role == Role.COMMUNICATOR:
            if simulation_t>3:
                self.estimate_communicator_movement(agent, agents,dt)
        
    def repulsive_movement(self, agent, positions):
        return sum([agent.x-position.x for position in positions]),sum([agent.y-position.y for position in positions])
    def attractive_movement(self, agent, positions):
        return sum([position.x-agent.x for position in positions]),sum([position.y-agent.y for position in positions])
    
    def movement_agent_explorer_in_centroid(self,agent,time_index,dt=0.1):
        if time_index[0] != len(agent.centroids)-1:
            
            point = agent.centroids[time_index[0]+1]
            new_point = Position(point.x*0.9,point.y*0.9)
            attraction_x_centroid,attraction_y_centroid = self.attractive_movement(agent, [agent.centroids[time_index[0]],new_point])
        else:
            attraction_x_centroid,attraction_y_centroid = self.attractive_movement(agent, [agent.centroids[time_index[0]]])
        repulsion_x,repulsion_y = self.repulsive_movement(agent,agent.stored_positions)
        new_x,new_y = self.stochastic_integration(agent,[attraction_x_centroid,0.01*repulsion_x],[attraction_y_centroid,0.01*repulsion_y],dt)
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
    
    def estimate_explorer_movement(self,agent,dt=0.1):
        if in_circle(agent.x,agent.y,agent.exploration_centroid.x,agent.exploration_centroid.y,(agent.exploration_buffer_radius+agent.sensing_radius)*2) and agent.load_state:
            t = agent.t
            time_index = [i for i in range(len(agent.centroids_schedule)) if agent.centroids_schedule[i]>t]
            if len(time_index) == 0 and not agent.finished_exploring:
                agent.finished_exploring = True
                return
            if in_circle(agent.x,agent.y,agent.centroids[time_index[0]].x,agent.centroids[time_index[0]].y,agent.sensing_radius):
                agent.centroids = agent.centroids[1:]
                agent.centroids_schedule = agent.centroids_schedule[1:]
                positions = [[position.x,position.y] for position in agent.stored_positions]
                positions = np.array(positions)
                return 
            new_x, new_y = self.movement_agent_explorer_in_centroid(agent,time_index,dt)
            new_x,new_y = self.apply_explorer_constraints(agent,new_x,new_y)
            agent.t += 1
        else:
            if agent.load_state:
                attraction_x,attraction_y = self.attractive_movement(agent, [agent.exploration_centroid])
                new_x,new_y = self.stochastic_integration(agent,[attraction_x],[attraction_y],dt)
            else:
                attraction_x,attraction_y = self.attractive_movement(agent, [Position(0,0)])
                new_x,new_y = self.stochastic_integration(agent,[attraction_x],[attraction_y],dt)
                if agent.finished_exploring:
                    print('agent exploring finished, heading to base',agent.id)
            new_x,new_y = self.apply_speed_constraints(agent,new_x,new_y)
            if not agent.load_state and in_circle(new_x,new_y,0,0,agent.nest_radius) and agent.communication_threshold == 0 and agent.finished_exploring == True:
                agent.role = Role.COMMUNICATOR
        agent.stored_positions.append(Position(agent.x,agent.y))
        agent.x = np.round(new_x,2)
        agent.y = np.round(new_y,2)
    def get_noise(self,agent):    
        new_noise_x = agent.prev_noise.x*(1-agent.theta) + agent.sigma*np.random.randn()
        new_noise_x = np.clip(new_noise_x,-agent.sigma,agent.sigma) 
        new_noise_y = agent.prev_noise.y*(1-agent.theta) + agent.sigma*np.random.randn()
        new_noise_y = np.clip(new_noise_y,-agent.sigma,agent.sigma)
        return new_noise_x,new_noise_y
    def get_acceleration(self,agent,iterables_x,iterables_y):
        return agent.kappa*sum(iterables_x),agent.kappa*sum(iterables_y)
    def stochastic_integration(self,agent,iterables_x,iterables_y,dt=0.1):
        new_noise_x,new_noise_y = self.get_noise(agent)
        agent.prev_noise = Noise(new_noise_x,new_noise_y)
        acceleration_x,acceleration_y = self.get_acceleration(agent,iterables_x,iterables_y)
        agent.prev_acceleration = Acceleration(acceleration_x,acceleration_y)
        acceleration_x = np.clip(acceleration_x + new_noise_x,-agent.a_max,agent.a_max)
        acceleration_y = np.clip(acceleration_y + new_noise_y,-agent.a_max,agent.a_max)
        velocity_x = agent.prev_speed.x+acceleration_x*dt
        velocity_x = np.clip(velocity_x,-agent.v_max/dt,agent.v_max/dt)
        velocity_y = agent.prev_speed.y+acceleration_y*dt 
        velocity_y = np.clip(velocity_y,-agent.v_max/dt,agent.v_max/dt)
        agent.prev_speed = Speed(velocity_x,velocity_y)
        new_x = agent.x + velocity_x*dt
        new_y = agent.y + velocity_y*dt
        return new_x,new_y
    def movement_agent_communicator(self,agent,agents,dt=0.1):
        nearby_agents = self.get_neighbors(agent,agents)
        neighbors = []
        for neighbors_ in agent.seen_neighbors.values():
            neighbors.extend(neighbors_)
        seen_neighbors = [agent_ for agent_ in nearby_agents if agent_ in neighbors]
        not_seen_neighbors = [agent_ for agent_ in nearby_agents if agent_ not in neighbors]
        seen_neighbors = [Position(agent_.x,agent_.y) for agent_ in agents if agent_.id in seen_neighbors]
        not_seen_neighbors = [Position(agent_.x,agent_.y) for agent_ in agents if agent_.id in not_seen_neighbors]
        
        attraction_x,attraction_y = self.attractive_movement(agent,not_seen_neighbors)
        repulsive_neighbors_x,repulsive_neighbors_y = self.repulsive_movement(agent,seen_neighbors)
        repulsive_past_x, repulsive_past_y = self.repulsive_movement(agent,agent.stored_positions)
        new_x,new_y = self.stochastic_integration(agent,[attraction_x,repulsive_neighbors_x,repulsive_past_x],[attraction_y,repulsive_neighbors_y,repulsive_past_y],dt)
        return new_x,new_y
    def apply_explorer_constraints(self,agent,new_x,new_y):
        constraints = [self.max_speed_constraint_condition(agent,agent.x,agent.y,new_x,new_y),self.exploration_movement_condition(agent,new_x,new_y)]            
        new_x = torch.tensor([new_x],requires_grad=True,dtype=torch.float)
        new_y = torch.tensor([new_y],requires_grad=True,dtype=torch.float)
        prev_x = torch.tensor([agent.x],requires_grad=True,dtype=torch.float)
        prev_y = torch.tensor([agent.y],requires_grad=True,dtype=torch.float)
        v_max = torch.tensor([agent.v_max],requires_grad=True,dtype=torch.float)
        sensing_radius = torch.tensor([agent.sensing_radius],requires_grad=True,dtype=torch.float)
        buffer_radius = torch.tensor([agent.exploration_buffer_radius],requires_grad=True,dtype=torch.float)
        counter = 0
        while all(constraints):                                        
            constraint_functions = torch.nn.functional.softplus(torch.stack([self.max_speed_constraint_function(prev_x,prev_y,new_x,new_y,v_max),
                                                self.exploration_movement_function(new_x,new_y,agent.exploration_centroid,sensing_radius+buffer_radius)])).sum()
            new_x,new_y = gradient_step(constraint_functions,new_x,new_y,1.0)
            constraints = [self.max_speed_constraint_condition(agent,agent.x,agent.y,new_x.item(),new_y.item()),
                            self.exploration_movement_condition(agent,new_x.item(),new_y.item())]
            counter += 1
            if counter == 20:
                break
        return new_x.item(),new_y.item()
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
    def estimate_communicator_movement(self,agent,agents,dt):
        new_x,new_y = self.movement_agent_communicator(agent,agents,dt)
        new_x,new_y = self.apply_communicator_constraints(agent,new_x,new_y)
        agent.stored_positions.append(Position(agent.x,agent.y))
        agent.x = np.round(new_x,2)
        agent.y = np.round(new_y,2)
        
    def max_speed_constraint_condition(self,agent, prev_x,prev_y,new_x,new_y):
        return ((new_x-prev_x)**2+(new_y-prev_y)**2)**(1/2) <= agent.v_max
    def exploration_movement_condition(self,agent,new_x,new_y):
        return (new_x-agent.exploration_centroid.x)**2 + (new_y-agent.exploration_centroid.y)**2 <= (2*(agent.sensing_radius+agent.exploration_buffer_radius))**2 
    def communication_movement_condition(self,agent,new_x,new_y):
        return new_x**2 + new_y**2 <= (agent.nest_radius+agent.exploration_buffer_radius)**2
    def max_speed_constraint_function(self, prev_x,prev_y,new_x,new_y,v_max):
        return (new_x-prev_x)**2+(new_y-prev_y)**2-v_max**2
    def exploration_movement_function(self,new_x,new_y,exploration_centroid,radius):
        return +(new_x-exploration_centroid.x)**2 + (new_y-exploration_centroid.y)**2-(2*radius)**2 
    def communication_movement_function(self,new_x,new_y,radius):
        return new_x**2 + new_y**2-(radius)**2
    def get_neighbors(self, agent, agents):
        return [
            other.id for other in agents if agent.id != other.id and in_circle(agent.x, agent.y, other.x, other.y, agent.sensing_radius)
        ]