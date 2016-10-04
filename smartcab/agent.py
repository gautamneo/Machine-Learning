import random
import math
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
q = {}
actions = [None,'forward','left','right']
trial = 0
class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    
    def __init__(self, env):
        q.clear()
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.discount_factor = 1
        self.learning_rate = 0.1
        self.deadline = self.env.get_deadline(self)
                      
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # print to find number of states explored
#         global trial
#         trial += 1
#         if(trial == 99):
#             print len(q)
            
     

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        inputs.pop('right')
        deadline = self.env.get_deadline(self)
       
        self.state = (frozenset(inputs.items()),(deadline/5),self.next_waypoint)
        # self.state = (frozenset(inputs.items()),self.next_waypoint)
        
        # TODO: Select action according to your policy
        # Global variable q stores {State:(q1,q2,q3,q4)} key-value pairs where value[x] corresponds to q value for Global actions[x] action 
        
        global q
        global actions
        global trial
        
        # without exploration decay
              
        if self.state in q :
            action = actions[self.max_random(q[self.state])]
        else:
            q.update({self.state:np.zeros(4)})
            action = random.choice(actions)
               
            
        # with exploration decay  
        # compute Yes or No for exploitation vs exploration from the sigmoid function 
        
#         flag = random.random() < self.sigmoid(trial)
#           
#         if self.state in q :
#             if(flag):
#                 action = actions[self.max_random(q[self.state])]
#             else:
#                 action = random.choice(actions)
#         else:
#             q.update({self.state:np.zeros(4)})
#             action = random.choice(actions)
#            
                                  
            
        # select random for PART 1 of PROJECT
        
#         import random
#         action = random.choice(actions)
        
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.update_q(self.state,action,reward)
     
        print "LearningAgent.update(): deadline = {}, inputs = {}, waypoint = {}, action = {}, reward = {}".format(deadline, inputs,self.next_waypoint, action, reward)  # [debug]

    def sigmoid(self,x):
        
        # scale the input range (0,100) to active range of the sigmoid function(-sqrt(3),sqrt(3))        
        scaled_input = math.sqrt(3)*(-1) + x*math.sqrt(3)/50
        
        return 1/(1+math.exp(-scaled_input))    
    
    def update_q(self, state, action, reward):
        
        global q
        action_index = actions.index(action)
        next_state = self.build_state(self.env.sense(self))
        
        if next_state in q:
            max_next_q = max(q[next_state])
        else:
            max_next_q = 0
        cur_q = q[state][action_index] 
        
        # Implement the Q-learning Bellman Algorithm
        
        learned = reward + self.discount_factor*max_next_q
        new_q = cur_q + self.learning_rate*(learned - cur_q)
                
        q[state][action_index] = new_q
        
    def build_state(self,inputs):
        return (frozenset(inputs.items()),self.env.get_deadline(self)/5,self.next_waypoint)
        # return (frozenset(inputs.items()),self.next_waypoint)
    
    def get_max_q(self, state):
        
        if state in q:
            return max(q[state])
        q.update({state:np.zeros(4)})
        return 0
     
    def max_random(self,array):
        
        maxim = np.amax(array)    
        array_of_max = np.where(array == maxim)[0]
        return random.choice(array_of_max)
        
                            
         
def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    print "success : {}".format(e.success)

if __name__ == '__main__':
    run()
