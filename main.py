import inspect, os, sys, argparse  #Imports


###Define Functions

#Read Grid-File from 'gridpath' into 2-dimensional array 'grid'
#Take 'gridpath'
#Terminate, if 'gridpath' is not appropriate
#Return 'grid'
def getgridfromgridpath(gridpath):
    grid = []
    try:
        with open(gridpath) as gridfile:
            grid = [line.split() for line in gridfile]
    except Exception as error:
        print("Error opening grid from gridpath! Please specify appropriate grid.")
        exit()
    print("Read grid from " + gridpath + ".\n")
    return grid

#Print grid-pattern from 'grid'
#Take 'grid'
def printgrid(grid):
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
#---------------------------------------

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
parser.add_argument("-p", "--gridpath", help="Specify path of gridfile.")
parser.add_argument("-e", "--evalsteps", help="Define number of evaluation steps n. Needed for automatic iteration.")
parser.add_argument("-g", "--gammavalue", help="Define Gamma for weighting of rewards.")
args = parser.parse_args()
evalsteps = args.evalsteps #Args are None, if not specified
gridpath = args.gridpath
gamma = args.gammavalue

#Gridpath must be provided, check for correctness later
if ((args.gridpath is None)):
    parser.error('Please specify a gridpath.')

#If -e then also -g necessary, if -g then also -e necessary
if ((args.evalsteps is not None and args.gammavalue is None) or
    (args.evalsteps is None and args.gammavalue is not None)):
    parser.error('Please specify both, evalsteps and gammavalue.')




#Read in grid

#Try to open from gridpath, if not provided, terminate
grid = getgridfromgridpath(gridpath) #Exits program, if gridpath not appropriate
print("Read-in grid: ")
printgrid(grid) #Print grid











