import numpy as np


#############
# MDP CLASS #
#############
class Holder:
    """
    Create Marcov Decision Processes
    Input:
        State = array with Gridworlds (Strings)
        Actions = predefined possible actions (up, down, left, right)
        probability = probability of intended state change
        reward = standard short-term reward
    """
 
    def __init__(self, State, probability, reward, zip_policy, discount_factor):
        """
        call to get Gridworld
        """ 
        self.field = State
        self.zip_policy = zip_policy
        self.prob = probability
        self.reward = reward
        self.discount_factor = discount_factor
     
     
        
##################
# # EVALUATION # #
##################

def old_value(field,new_state,old,state):
    """
    return value of new state
    if the new state is an obsticle we do not move
    and return the value of the current state
    """
    
    i = (int)(new_state[0])
    j = (int)(new_state[1])
    
    old_i = (int)(state[0])
    old_j = (int)(state[1])
    
    M,N = field.shape
    
    # Not performing an action if we try to leave the grid world or move against an obstacle
    # => no state transition
    if i < 0 or i >= M or j < 0 or j >= N or field[i][j] == 'O':
        return old[old_i,old_j]
   
    return old[i,j]

def v(state, old, holder):
    """
    calculate value of state following policy
    
    state as current state (tuple)
    mdp as MDP object which holds field, probability, reward
    discount_factor as float
    policy as 2d array
    """
    field = holder.field
    
    # retrieve action (movement in x and y direction) from policy
    x_policy, y_policy = holder.zip_policy
    x,y = x_policy[state],y_policy[state]
    
    # calculate new state
    # add x,y for intended new state
    # add y,x for moving to the right of intended
    # add -y,-x for moving to the left of intended
    state_1 = (state[0] + x, state[1] + y)
    state_2 = (state[0] + y, state[1] + x)
    state_3 = (state[0] - y, state[1] - x)
    
    # probability of moving in an unintended direction
    prob = holder.prob
    un_prob = (1 - holder.prob/2)
    
    # formula from slides with old value function
    
    return (holder.reward + holder.discount_factor * (prob * old_value(field,state_1,old,state)
                                                + un_prob * old_value(field,state_2,old,state) 
                                                + un_prob * old_value(field,state_3,old,state)))

def evaluation(holder,iterations = 50):
    """
    policy evaluation
    
    in:
    field original grid world
    probability as float
    reward as float
    policy as dictionary (state -> action)
    discount_factor as float
    
    return evaluated policy as value function
    """
    
    # get original grid world
    field = holder.field
    M,N = field.shape
    
    # create a 2d array which is going to hold the previous value matrix for comparison
    old = np.zeros((M,N))
    v_matrix = np.zeros((M,N))
    
    # evaluate policy n times
    for _ in range(iterations):
        
        # new value matrix is all zeros
        v_matrix = np.zeros((M,N))
        
        # iterate over each and every state and perform updates
        for i in range(M):
            for j in range(N):
                
                # if there is an 'O' in the grid world we do not want to take this field into account
                # therefore we will assign -99 to the field (None's are bad for comparison)
                # we also ignore 'O' fields when doing the greedy policy update
                # => no need to worry about this artificial negative 99
                if field[i,j] == 'O':
                    v_matrix[i,j] = None
                    old[i,j] = None
                    
                # if state is exit, value = 1
                if field[i,j] == 'E':
                    v_matrix[i,j] = 1
                    old[i,j] = 1
                
                # if state is pitfall, value = -1 
                if field[i,j] == 'P':
                    v_matrix[i,j] = -1
                    old[i,j] = -1
                    
                # if state is normal field, calculate new value
                if field[i,j] == 'F':
                    v_matrix[i,j] = v((i,j),old,holder)
                            
        # calculated value matrix is now old matrix
        old = np.copy(v_matrix)
    
    return v_matrix
 
########################
# # POLICY ITERATION # #
########################
def _iterate(v, holder):
    """
    Iterate over every State once
    
    Input: 
        v = value function 2D-Array of expectred reward at each state
        MDP = MDP-Object (Markov-Decision-Processes) for which the optimal policy should be found
        policy = policy 2D-Array elem = action/ direction to move
        discount_factor = for policy evalutatin float elem[0,1]
    Return:
        v , policy
    """
    value_function = evaluation(holder)
    x_policy, y_policy = holder.zip_policy
    M, N = holder.field.shape
    
    # obstacle padding
    biggerState =[]
    
    for i in range(M+2):
        sublist = []
        for j in range(N+2):
            sublist.append('O')
        biggerState.append(sublist)
     
    biggerState = np.asarray(biggerState)
    
    biggerState[1:M+1,1:N+1] = holder.field
    
    bigger = np.ones((M+2,N+2)) * -9999999999
    bigger[1:M+1,1:N+1] = value_function
    value_function = bigger
    
    for i,j in np.argwhere(biggerState != 'O'):
            
            # finding max in 4 nbh
            # lets improve this part pls
            
            elem_list = []
            elem_list.append(value_function[i-1,j])
            elem_list.append(value_function[i+1,j])
            elem_list.append(value_function[i,j+1])
            elem_list.append(value_function[i,j-1])
            
            cords_list = []
            cords_list.append((i-1,j))
            cords_list.append((i+1,j))
            cords_list.append((i,j+1))
            cords_list.append((i,j-1))
            
            max_point =  cords_list[elem_list.index(max(elem_list))]
                  
            x,y = tuple(np.subtract(max_point,(i,j)))
            
            x_policy[i-1,j-1] = x
            y_policy[i-1,j-1] = y
    
    policy = (x_policy,y_policy)
    
    return value_function[1:M+1,1:N+1],policy

def _policyIteration(holder, iterations = None):
    """
    Find optimal policy by iterating over the policy until stopping-condition 
    is met either until converges or amount of steps reached

    Input:
        MDP = MDP-Object (Markov-Decision-Processes) for which the optimal policy should be found
        discount_factor = for policy evalutatin float elem[0,1]
        iterations = number of calls, not stated do until policy converges

    Return:
        Optimal policy as 2D-array
    """
    
    #Initialize value function
    M, N = holder.field.shape
    v_function = np.zeros((M,N))
    
    if callable(iterations):
        #iterate over policy n-times
        for i in range(iterations):
            v_function, zip_policy = _iterate(v_function, holder)
    
    else:
        #iterate over policy until converges
        
        while((v_function.any() != evaluation(holder).any())):
            v_function, zip_policy = _iterate(v_function, holder)
                
    return zip_policy
 
########
# MAIN #
########
import inspect, os #Imports

debug = True #Global Switch for Debug-Info

#Determine working directory and gridpath
scriptpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
gridsubpath = "/Grids/3by4.grid"
gridpath = scriptpath + gridsubpath

#Read Grid-File as 2-dimensional array 'grid' from 'gridpath'
with open(gridpath) as gridfile:
    grid = [line.split() for line in gridfile]

        
field = np.asarray(grid)
M,N = field.shape

x_policy = np.ones((M,N))
y_policy = np.zeros((M,N))
zip_policy = (x_policy,y_policy)

discount_factor = 0.7

holder = Holder(field,0.8,-0.04,zip_policy,discount_factor)

# hoch =  -1 0 
# rechts = 0 1
# links = 0 -1
# down = 1 0

perfect_policy = _policyIteration(holder)

# steffis kosmetische update funktion für die policy

print(perfect_policy[0])
print()
print(perfect_policy[1])
