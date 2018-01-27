#!/usr/local/bin/python3
# coding: utf-8


import argparse
import numpy as np


###############################
###Define Classes and Functions
###############################

######
# IO #
######

def getgridfromgridpath(gridpath):
    """
    Read Grid-File line-by-line from provided path
    Input:
        gridpath: Path of raw gridfile
    Return:
        grid: Contains line-by-line read in gridfile
    Terminate:
        If file at gridpath cannot be opened
    """
    
    grid = []
    try:
        with open(gridpath) as gridfile:
            grid = [line.split() for line in gridfile]
    except Exception as error:
        print("Error opening grid from gridpath! Please specify appropriate grid.")
        exit()
    print("Read grid from " + gridpath + ".\n")
    return grid

def printgrid(grid):
    """
    Prints raw gridpattern if appropriate grid is specified. Do not print otherwise.
    Input:
        grid: Raw line-by-line grid.
    """
    
    try:                                    #Check that grid is two-dimensional
        len(grid)                           #Grid has elements
        len(grid[0])                        #Grid is list of lists
        if (isinstance(grid[0][0], list)):  #No more than 2 dimensions
            raise Exception
    except:
        print("Error printing grid. No appropriate format. Not printing grid.\n")
        return
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            print(grid[i][j], end=' ')
        print()
    print() #Formatting, empty line after print-out

def beautyMode(policy, field):
    """
    Exchange ugly vector representation with beautiful arrows
    
    Input:
        Policy as zipped array or 3D array
        Field : Gridworld  to get shape
    """
    
    M,N = field.shape
    beauty = np.asarray(field)
    
    #go over policy and insert arrows in the beauty array for the different vectors
    for x in range(M):
        for y in range(N):
            if perfect_policy[0][x,y] == -1:
                beauty[x,y] = '↑'
            if perfect_policy[0][x,y] == 1:
                beauty[x,y] = '↓'
            if perfect_policy[1][x,y] == 1:
                beauty[x,y] = '→'
            if perfect_policy[1][x,y] == -1:
                beauty[x,y] = '←'
            print(beauty[x,y], end =' ') #Print immediately
        print()
    print()





##################
# MDP_PLUS CLASS #
##################
class MDP_PLUS:
        """
        Create Marcov Decision Processes
        Input:
            State = array with Gridworlds (Strings)
            probability = probability of intended state change
            reward = standard short-term reward
            discount_factor = discount factor
            iterations = number of evaluation steps
        """
 
        def __init__(self, State, probability, reward, discount_factor, iterations = 1):
            """
            call to get Gridworld
            """
            self.S = State
            self.p = probability
            self.r = reward
            self.its = iterations
            self.gamma = discount_factor
 
        def get_state(self):
            return self.S
        
        def get_iterations(self):
            return self.its
        
        def get_prob(self):
            return self.p
        
        def get_reward(self):
            return self.r
        
        def get_gamma(self):
            return self.gamma



##################
#   EVALUATION   #
##################
def old_value(field,new_state,old,state):
    """
    If the new state is an obstacle we do not move
    and return the value of the current state
    Input:
        field = grid wold as list
        new_state = coordinates we move to
        old = value function of t-1
        state = state to check
    Return:
        value of new state
    
    """
    
    i = (int)(new_state[0])
    j = (int)(new_state[1])
    
    old_i = (int)(state[0])
    old_j = (int)(state[1])
    
    M,N = field.shape
    
    #check if move possible and either return value state at t-1
    if i < 0 or i >= M or j < 0 or j >= N or field[i][j] == 'O':
        return old[old_i,old_j]
   
    #or return value of state at t (after taking an action)
    return (float)(old[i,j])


def v(state, old, mdp, zip_policy):
    """
    calculate value of state following policy
    Input:
        state =  current state (tuple)
        old = value function of t-1
        mdp as MDP_PLUS object which holds field, probability, reward, discount_factor
        zip_policy = tuple of 2 2D-arrays = action/ direction to move
    """
    
    x_policy, y_policy = zip_policy
    
    field = mdp.get_state()
    i,j = state

    # retrieve action from policy
    x,y = x_policy[state[0]][state[1]],y_policy[state[0]][state[1]]
    
    # calculate new state
    # add x,y for intended new state
    # add y,x for moving to the right of intended
    # add -y,-x for moving to the left of intended
    state_1 = (i + x, j + y)
    state_2 = (i + y, j + x)
    state_3 = (i - y, j - x)
    
    # probability of moving in an unintended direction
    prob = round(mdp.get_prob(), 3)
    un_prob = round((1 - mdp.get_prob())/2, 3)

    # formula from slides with old_value function
    value_1 = prob * old_value(field,state_1,old,state)
    value_2 = un_prob * old_value(field,state_2,old,state)
    value_3 = un_prob * old_value(field,state_3,old,state)
    
    #calculate value function as seen on the slides
    value = round((mdp.get_reward() + mdp.get_gamma() * (value_1 + value_2 + value_3)), 3)
    
    return value

def evaluation(mdp,zip_policy):
    """
    policy evaluation
    
    Input:
    mdp = field original grid world, probability as float, reward as float, discount_factor as float
    zip_policy = tuple of 2 2D-arrays = action/ direction to move
    
    Return:
        v_matrix = evaluated policy as value function
    """
    
    # get original grid world
    field = mdp.get_state()
    M,N = field.shape
    
    # create a 2d array which is going to hold the previous value matrix for comparison
    old = np.zeros((M,N))
    v_matrix = np.zeros((M,N))
    
    max_diff = 999999
    threshold = 0.01
    
    
    # evaluate policy until change is neglectable
    #while(abs(max_diff) > threshold):
    for _ in range(mdp.get_iterations()):
        # new value matrix is all zeros
        v_matrix = np.zeros((M,N))
        
        exit = []
        pit = []

        # iterate over each and every state and perform updates
        for i in range(M):
            for j in range(N):
                
                # if obstible, value = 0
                if field[i,j] == 'O':
                    v_matrix[i,j] = None
                    old[i,j] = None
                
                # if state is exit, value = 1
                if field[i,j] == 'E':
                    exit.append((i,j))
                
                # if state is pitfall, value = -1
                if field[i,j] == 'P':
                    pit.append((i,j))
                
                # if state is normal field, calculate new value
                if field[i,j] == 'F':
                    v_matrix[i,j] = v((i,j), old, mdp, zip_policy)
        
        
        
        # difference matrix
        for cords in exit:
            v_matrix[cords] = 1
        for cords in pit:
            v_matrix[cords] = -1
        
        # calculated value matrix is now old matrix
        old = np.copy(v_matrix)

    # return value function
    return v_matrix

########################
#   POLICY ITERATION   #
########################
def _iterate(v, mdp, policy):
    """
    Iterate over every State once
    
    Input:
        v = value function 2D-Array of expectred reward at each state
        mdp = MDP_PLUS-Object (Markov-Decision-Processes + discount factor) for which the optimal policy should be found
        policy = tuple of 2 2D-arrays = action/ direction to move
    Return:
        v , policy
    """
    v = evaluation(mdp, policy)
    x_policy, y_policy = policy
    M, N = mdp.get_state().shape
    
    # obstacle padding
    bigger_state =[]
    
    #create a List that has Obstacles on all 4 sides
    for i in range(M+2):
        sublist = []
        for j in range(N+2):
            sublist.append('O')
        bigger_state.append(sublist)
    
    bigger_state = np.asarray(bigger_state)
    
    #insert grid world so it is surrounded by Obstacles
    bigger_state[1:M+1,1:N+1] = mdp.get_state()
    
    #Also pad the valuefunction so it does not want to go on the boarder
    bigger = np.ones((M+2,N+2)) * -9999999999
    bigger[1:M+1,1:N+1] = v
    v = bigger
    
    #update policy in a greedy manner
    #go over every field 'F'
    for i,j in np.argwhere(bigger_state == 'F'):
        
            elem_list = []
            cords_list = []
            
            #get values of 4-Neighbourhood
            elem_list.append(old_value(bigger_state,(i-1,j),v,(i,j)))
            cords_list.append((i-1,j))

            elem_list.append(old_value(bigger_state,(i,j-1),v,(i,j)))
            cords_list.append((i,j-1))

            elem_list.append(old_value(bigger_state,(i+1,j),v,(i,j)))
            cords_list.append((i+1,j))

            elem_list.append(old_value(bigger_state,(i,j+1),v,(i,j)))
            cords_list.append((i,j+1))
            
            #take the greedy action
            max_point =  cords_list[elem_list.index(max(elem_list))]
            
            #get action by substracting point with highest reward and state's posittion
            x,y = tuple(np.subtract(max_point,(i,j)))
            
            #put coordinates in non-padded Policy
            x_policy[i-1,j-1] = x
            y_policy[i-1,j-1] = y
    
    #rezip policy
    policy = (x_policy,y_policy)
    
    #return value function in non padded area and policy
    return v[1:M+1,1:N+1],policy


def _policyIteration(mdp, zip_policy):
    """
    Find optimal policy by iterating over the policy until stopping-condition
    is met either until converges or amount of steps reached

    Input:
        mdp = MDP_PLUS-Object (Markov-Decision-Processes) for which the optimal policy should be found
        zip_policy = tuple of 2 2D-arrays = action/ direction to move
        iterations = number of calls, not stated do until policy converges

    Return:
        Optimal policy as 2D-array
    """
    
    
    #Initialize value function and set starting policy
    M, N = mdp.get_state().shape
    v_function = np.zeros((M,N))
    
    #iterate over policy until value function converges
    diff = np.ones((M,N))
    while(np.max(np.absolute(diff) > 0)):
        
        v_function, zip_policy = _iterate(v_function, mdp, zip_policy)
        diff = v_function - evaluation(mdp, zip_policy)
        
    return zip_policy



########
### MAIN
########


###Start Program here
print()
print("-------------------------------------------")
print("| Methods of AI - Markov Decision Process |")
print("-------------------------------------------")
print()


###Evaluate input parameters

#Create Argparser with options p, e, g
#Gridpath is necessary
#If evalsteps or gammavalue provided, other is needed
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--gridpath", help="Specify path of gridfile. Needed.")
parser.add_argument("-e", "--evalsteps", help="Define number of evaluation steps n. Needed for automatic iteration.")
parser.add_argument("-g", "--gammavalue", help="Define Gamma for weighting of rewards. Needed for automatic iteration.")
args = parser.parse_args()


#Gridpath must be provided, check for correctness later
if ((args.gridpath is None)): #Args are None, if not specified
    parser.error('Please specify a gridpath.')

#If -e then also -g necessary, if -g then also -e necessary
if ((args.evalsteps is not None and args.gammavalue is None) or
    (args.evalsteps is None and args.gammavalue is not None)):
    parser.error('Please specify both, evalsteps and gammavalue.')

#Evalsteps must be int, gamma must be float; set gridpath (string)
evalsteps = None
gamma = None
if ((args.evalsteps is not None)):
    try:
        evalsteps = int(args.evalsteps)
        gamma = float(args.gammavalue)
    except:
        parser.error('Please specify valid evalsteps and gammavalue.')
gridpath = args.gridpath



###Read in grid
#Try to open from gridpath, if not provided, terminate
grid = getgridfromgridpath(gridpath) #Exits program, if gridpath not appropriate
print("Raw read-in grid: ")
printgrid(grid) #Print grid


###Start Computation of Policy

field = np.asarray(grid)
M,N = field.shape

#initialize policy
y_policy = np.ones((M,N))
x_policy = np.zeros((M,N))

#set Obstacle, End and Pit with none in the policy so it does not change over the iteration
obs = np.argwhere(field == 'O')
ext = np.argwhere(field == 'E')
pit = np.argwhere(field == 'P')

for [x,y] in obs:
    x_policy[x,y] = None
    y_policy[x,y] = None

for [x,y] in ext:
    x_policy[x,y] = None
    y_policy[x,y] = None

for [x,y] in pit:
    x_policy[x,y] = None
    y_policy[x,y] = None
zip_policy = (x_policy,y_policy)

#Initialize MDP plus discount factor
#             (State, probability, reward, discount_factor, iterations = 1)
if evalsteps == None: #Provided only gridpath, automatic iteration
    mdp = MDP_PLUS(field,0.8,-0.04,1)
else:                 #Provided more parameters
    mdp = MDP_PLUS(field,float(gamma),-0.04,int(evalsteps))

# up =  -1 0
# right = 0 1
# left = 0 -1
# down = 1 0

#start Policy Iteration
perfect_policy = _policyIteration(mdp, zip_policy)

#print out our sexy Policy
print("Generated Policy:")
beautyMode(perfect_policy, field)

