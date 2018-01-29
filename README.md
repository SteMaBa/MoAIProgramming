# MoAI Programming Task 1 - Markov Decision Process

One Paragraph of project description goes here

## Getting Started

These 

### Prerequisites

Start the script with a **Python 3 interpreter**.
Necessary additional libraries include **numpy**.


```
Give examples
```

### Parameter Setting

Use the -h flag to show information about the possible parameters:


```
  -h, --help            show this help message and exit
  -m, --manual          Enable step-by-step evaluation with option for
                        intermediate input.
  -p GRIDPATH, --gridpath GRIDPATH
                        Specify path of gridfile. Needed.
  -e EVALSTEPS, --evalsteps EVALSTEPS
                        Define number of evaluation steps n. Needed for
                        automatic iteration.
  -g GAMMAVALUE, --gammavalue GAMMAVALUE
                        Define Gamma for weighting of rewards. Default value
                        1.
  -r REWARD, --reward REWARD
                        Define expected reward for actions in non-terminal
                        states. Default value -0.04.
  -d DESIREDPROB, --desiredprob DESIREDPROB
                        Define probability of performing desired action.
                        Default value 0.8.
  -a, --animate         Show intermediate steps during automatic processing.
```


## Run the script

### Automatic processing

An example run for automatic iteration looks like this:

```
python3 main.py -p *PATH_TO_GRID* -e 100
```

Alternatively, if python3 is located at /usr/local/bin/python3, make main.py executable and run directly from shell.

Output for above command and grid 5by10.grid yields:

```

-------------------------------------------
| Methods of AI - Markov Decision Process |
-------------------------------------------

Read grid from /User/moai/5by10.grid.

Raw read-in grid: 
F F F F F F F F F F 
F O O O O O F O O F 
F O F F O P F O E F 
F O O F O O O O O F 
F F F F F F F F F F 

Parameters:
Probability performing desired action: 0.8  Short-term reward: -0.04  Gamma-value: 1  Evaluation-Steps: 100  
Do manual iteration: False  Show intermediate steps during automatic processing: False 


Generating Policy...

Generated Policy Evaluation:
+0.37 +0.42 +0.47 +0.53 +0.57 +0.62 +0.68 +0.73 +0.78 +0.83 
+0.32                               +0.62             +0.89 
+0.27       +0.37 +0.42       -1.00 +0.40       +1.00 +0.94 
+0.32             +0.47                               +0.89 
+0.37 +0.42 +0.47 +0.53 +0.58 +0.63 +0.68 +0.73 +0.78 +0.83 

Arrow-Representation of generated Policy:
→ → → → → → → → → ↓ 
↑           ↑     ↓ 
↑   → ↓     ↑     ← 
↓     ↓           ↑ 
→ → → → → → → → → ↑ 

```


### Manual processing

This is an example command for manual processing: 

```
python3 main.py -p *PATH_TO_GRID* -m 
``` 

## Authors

Group 20
