#!/usr/bin/env python
# coding: utf-8

# Diese Version (mit b) soll dienen, um ein Video zu rendern.<br><br>
# TODO:
# - Forderstes Auto soll Rot sein und dessen Netzwerk soll angezeigt werden. Die anderen sollen grau sein.
# - Bei der Bahn soll im 'thick' Modus die Eckdarstellung zu wirklichen Ecken gemacht werden.

# In[1]:


import numpy as np
#import matplotlib.pyplot as plt
#from IPython.display import display, clear_output
import sys

if len(sys.argv) < 3:
    param_div = 1
    param_mod = 0
else:
    param_div = int(sys.argv[1])
    param_mod = int(sys.argv[2])
    
assert param_div != 0
print('Runnig with div=%d and mod=%d'%(param_div, param_mod))


from pylonen_utils_v5 import *
from plot_utils import *

_route = None


# ### Selbst fahren in Environment

# In[3]:


#import keyboard

env = EnvironmentVid()
env.delt = 0.15
env.render()


# ### Evolutionäres Lernen des Environments
# Sowohl sexuelle als auch nicht-sexuelle Reproduktion können ausprobiert werden. Beide schneiden in etwa gleich gut ab.<br>
# Eine Normalverteilung zur Initialisierung erwies sich als besser, als eine Gleichmäßige Verteilung.
# Am Schnellsten geht es (bei 30er Population) mit 3 reproduzierten Exemplaren.

# In[4]:

print('Importing genetic_evolution')
from genetic_evolution import *


# In[5]:


PLT_XY_EXCHANGED = True
    
def plt_plot(xs, ys, **args):
    if not PLT_XY_EXCHANGED:
        plt.plot(xs, ys, **args)
    else:
        plt.plot(ys, xs, **args)

def plt_scatter(xs, ys, **args):
    if not PLT_XY_EXCHANGED:
        plt.scatter(xs, ys, **args)
    else:
        plt.scatter(ys, xs, **args)

def show_network(Ws):
    'Shows the network architecture'
    for k in range(len(Ws)):
        W = Ws[k]
        ioff = W.shape[0]/2 - 0.5
        joff = W.shape[1]/2 - 0.5
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                s = W[i,j] >= 0
                plt.plot([i-ioff, j-joff], [-k, -k-1], lw=5*abs(W[i,j]), c=(s,0,1-s))
    plt.xticks([])
    plt.yticks([])
        
def show_network_act(Ws, bs, fs, input, actmax=1, scales=None):
    # Version ohne "3D"
    '''Shows the Network architecture and its activations.
    actmax: clipping of all activations is [0, actmax]
    scales: die darstellung eines layers auseinanderziehen. Indexversatz, da der input als activation angezeigt wird'''
    
    COLOR_ACT_OFF = np.array( (0., 0., 0.))
    COLOR_ACT_ON  = np.array( (0., 1., 0.))
    COLOR_ACT_ON  = np.array( (1.0, 0.925, 0.))
    LW_FACTOR = 5
    
    def _plot_activation(act, k):
        act = np.clip(act / actmax, 0., 1.)
        for j in range(act.size):
            c = COLOR_ACT_ON*act[j] + COLOR_ACT_OFF*(1-act[j])
            sc = scales[k] if scales else 1
            plt_scatter([sc * (j-act.size/2+0.5)], [k], c=['black'], linewidth=12, zorder=3)
            plt_scatter([sc * (j-act.size/2+0.5)], [k], c=[c], linewidth=10, zorder=3)
    
    # Plot Inputs as activation
    act = input
    _plot_activation(act, 0)
    
    for k in range(len(Ws)):
        W = Ws[k]
        b = bs[k]
        ioff = W.shape[0]/2 - 0.5
        joff = W.shape[1]/2 - 0.5
        
        # Weights are lines
        scale0 = scales[k] if scales else 1.
        scale1 = scales[k+1] if scales else 1.
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                s = 1.*(W[i,j] >= 0)
                plt_plot([scale0*(i-ioff), scale1*(j-joff)], [k, k+1], lw=LW_FACTOR*abs(W[i,j]), c=(s,0,1-s))
        
        # Activations are points
        act = np.matmul(act,W)+b
        act = fs[k](act)
        _plot_activation(act, k+1)
            
    plt.xticks([])
    plt.yticks([])
    plt.gca().invert_yaxis()
    
    


def evaluate(i, chrm, records=None):
    
    print("Starting with", i)
    
    env = EnvironmentVid()
    env.delt = DELT
    env.maxsteps = 40 / DELT
    

    obs = env.reset()
    env.count_collisions = 0
    done = False
    step = 0

    while not done:
        step += 1
        obs = np.array(obs)
        action = chrm.act(obs)
        action = action[0] - action[1]
        obs, reward, done, info = env.step(action)
        obs = np.ones(5)-obs

        records['car'][i].append(copy.deepcopy(env.car))
        records['obs'][i].append(obs.copy())
        records['raymarkers'][i].append(env.ray_markers.copy())
        records['progress'][i].append(env.progress)

        if env.count_collisions > 0:
            done = True
            break
        if step > 1000:
            done = True

    print(igen, i, env.steps)
    fitness = env.progress
    
    return fitness



import pickle ##
import copy
from time import sleep

DELT = 0.05
env = EnvironmentVid()
env.delt = DELT
env.maxsteps = 40 / DELT

print(f'delt = {env.delt}')
print(f'maxsteps = {env.maxsteps}')

print('=== Starting Loop ===')

for iseed in range(param_mod, 9999, param_div): ##
    
    print()
    print()
    print('Seed =', iseed)
    
    seed_records = dict() ##
    np.random.seed(iseed)  ##

    fitnesses = [] ##
    
    file_k = 0

    #plt.figure(figsize=(16, 6))
    gpool = genpool()

    for igen in range(NUM_GENERATIONS):
        records = { 'car': [[] for _ in range(NUM_POPULATION)],
                    'obs':  [[] for _ in range(NUM_POPULATION)],
                    'raymarkers': [[] for _ in range(NUM_POPULATION)],
                    'progress': [[] for _ in range(NUM_POPULATION)],
                    'chromosomes': gpool.chromosomes }

        for i in range(NUM_POPULATION):
            gpool.fitness[i] = evaluate(i, gpool.chromosomes[i], records)

        winner = np.argmax(gpool.fitness)

        #render_population(records['car'], None, records['obs'], records['raymarkers'],
        #                  records['progress'], gpool.chromosomes, igen)

        seed_records[igen] = deepcopy(records) ##

        fitnesses.append(deepcopy(gpool.fitness)) ##
        if np.any(gpool.fitness >= env.maxsteps): ##
            break
            
            
        pickle.dump([seed_records, fitnesses], open('records/seed%03d.pkl'%iseed, 'wb')) ##

        gpool.reproduce()
        

    
# seed_records hat das Format:
# seed_records[generation]['car'/'obs'/'raymarkers'/'progress'][number_of_ind] = car/obs/raymarkers/progress