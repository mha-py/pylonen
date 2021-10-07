import numpy as np
import matplotlib.pyplot as plt

NUM_IN = 5
NUM_HIDDEN = [4, 3]
NUM_HIDDENLAYERS = None
NUM_OUT = 2

NUM_GENERATIONS = 50
NUM_POPULATION = 20
NUM_REPRODUCE = 5

KEEP = 0 # keep the best
CROSSOVER = 0.5

MUTATION_STEP = 0.1
MUTATION_RATE = 0.1
MUTRATE_CHANGES = False


import copy

def sigmoid(z):
    a = 1./(1.+np.exp(-z))
    return a
def relu(z):
    a = np.clip(z, 0., None)
    return a
def identity(z):
    return z
def stepfunction(z):
    return z>0

FUNCTIONS = [ relu, relu, identity ]

class chromosome():
    def __init__(self, nin, nhid, nout, fs):
        if type(nhid) not in [list, tuple]: nhid = [nhid]
        assert len(nhid)+1 == len(fs)
        self.Ws = []
        self.bs = []
        self.fs = fs
        for n0, n1 in zip([nin] + nhid, nhid + [nout]):
            self.Ws.append( np.random.normal(0., 0.5, (n0, n1)) )
            self.bs.append( np.random.normal(0., 0.5, n1) )
            
        self.colorcode = np.random.rand(1, 100, 5)
        self.colorcode = np.random.rand(1000)
            
        # TEST
        self.m_step = MUTATION_STEP
        self.m_rate = MUTATION_RATE
        # END TEST
            
    def similarity(self, other): # how similar to another chromosome? result is between 0 and 1
        SIMILARITY_THRESHOLD = 5e-2
        a, b = 0, 0
        for M, M1 in zip(self.Ws + self.bs, other.Ws + other.bs):
            a += np.sum( abs(M-M1) < SIMILARITY_THRESHOLD*(abs(M) + abs(M1)).mean()/2 )
            b += M.size
        return a/b
            
    def act(self, input):
        a = input
        for W, b, f in zip(self.Ws, self.bs, self.fs):
            a = f(a @ W + b)
        return a
    
    def cross(self, other):
        for M, M1 in zip(self.Ws + self.bs + [self.colorcode], other.Ws + other.bs + [other.colorcode]):
            t = np.random.rand(*M.shape) > 0.5 # CHOICE
            #t = np.random.rand(*M.shape)       # WEIGHTED MEAN
            #t = np.random.rand(*M.shape)*2-0.5 # WEIGHTED MEAN AND EXTRAPOLATION
            M *= t
            M += (1-t)*M1
            
        if np.random.rand()<0.5:
            self.m_step = other.m_step
        if np.random.rand()<0.5:
            self.m_rate = other.m_rate
    
    def mutate(self):
        if MUTRATE_CHANGES:
            self.m_step *= np.random.choice([1, 1, 1.77827941, 1/1.77827941])
            self.m_rate *= np.random.choice([1, 1, 1.77827941, 1/1.77827941])
            self.m_step = min(1, self.m_step) # hier stand max? komisch.
            self.m_rate = min(1, self.m_rate)
        for M in self.Ws + self.bs + [self.colorcode]:
            M += self.m_step * np.random.normal(size=M.shape) * (np.random.rand(*M.shape) > self.m_rate)
            M += self.m_step * np.random.normal(size=M.shape) * (np.random.rand(*M.shape) > self.m_rate)
        self.colorcode.clip(0,1)
        
    def visualize(self): # prints the colorcode
        NUM = 20
        cc = self.colorcode[:NUM*3].reshape((1,NUM,3)).clip(0,1)
        plt.imshow(cc)
        plt.axis('off')
        plt.show()
        return cc
        
    def visualize2(self): # prints the colorcode
        cs = self.colorcode[:60].reshape((1,20,3))
        xs = self.colorcode[100:120]
        ys = self.colorcode[200:220]
        plt.scatter(xs, ys, c=cs, marker='s', lw=8)
        plt.axis('off')
        plt.show()
        
    def visualize3(self): # prints the colorcode
        NUM = 50
        cc = self.colorcode[:NUM*3].reshape((1,NUM,3)).clip(0,1)
        plt.imshow(cc, aspect=5)
        plt.axis('off')
        plt.show()
        return cc
        
        
        
        
        
        

class genpool():
    def __init__(self, npop=NUM_POPULATION, nin=NUM_IN, nhid=NUM_HIDDEN, nout=NUM_OUT, fs=FUNCTIONS):
        self.npop = npop
        self.chromosomes = [ chromosome(nin, nhid, nout, fs) for i in range(npop) ]
        self.fitness = np.zeros(npop)
        # plot stuff
        self.igeneration = 0 # iterates the generations
        self.bests = []
        self.percentile0 = []
        self.percentile1 = []
        self.median = []
        self.bank = []
        
        # TEST
        for c in self.chromosomes:
            origin = dict()
            origin['kind'] = 'firstgen'
            self.push_into_bank(c, origin)
        
    def adjust_fitness(self): # TEST-Stadium
        '''Berücksichtigt die Ähnlichkeit und bestraft diese'''
        sim = np.zeros((NUM_POPULATION, NUM_POPULATION))
        for i in range(NUM_POPULATION):
            for j in range(i):
                sim[i, j] = self.chromosomes[i].similarity(self.chromosomes[j])
                sim[j, i] = sim[i, j]
        self.fitness /= (1 + sim.mean(0))
        print((1 + sim.mean(0)))
        
    def push_into_bank(self, c, origin=None, fitness=None, rank=None):
        '''Erlaubt es, chromosom Information aufzuheben. Erstellt eine deepcopy.'''
        cc = copy.deepcopy(c)
        if fitness: cc.fitness = fitness
        if origin:  cc.origin  = origin
        if rank: cc.rank = rank
        cc.igeneration = self.igeneration
        self.bank.append(cc)
        c.ind = len(self.bank)-1
        
    def reproduce_SICHERUNG(self):
        '''Hier wird geschaut, wie viele Exemplare sich vermehren, evtl. wer mit wem. Die Kopulation und Mutation
        findet in der chromosome-Klasse statt.'''
        self.igeneration += 1
        self.adjust_fitness()
        oldchromosomes = copy.deepcopy(self.chromosomes)
        table = sorted(range(self.npop), key=lambda i: self.fitness[i], reverse=True)
        for i in range(0, KEEP):
            j = table[i]
            self.chromosomes[i] = copy.deepcopy(self.chromosomes[j])
            origin = dict()
            origin['kind'] = 'identical'
            origin['mother'] = len(self.bank)
            self.push_into_bank(self.chromosomes[i], origin)
        for i in range(KEEP, self.npop):
            j, j1 = np.random.randint(NUM_REPRODUCE, size=(2,))
            j, j1 = table[j], table[j1]
            # Selection
            self.chromosomes[i] = copy.deepcopy(oldchromosomes[j])
            # Crossover
            if np.random.rand() < CROSSOVER:
                origin = dict()
                origin['kind'] = 'crossover'
                origin['mother'] = oldchromosomes[j].ind
                origin['father'] = oldchromosomes[j1].ind
                self.chromosomes[i].cross(oldchromosomes[j1])
                self.push_into_bank(self.chromosomes[i], origin)
            # Mutation
            self.chromosomes[i].mutate()
            origin = dict()
            origin['kind'] = 'mutation'
            origin['mother'] = len(self.bank)-1 # last one added
            self.push_into_bank(self.chromosomes[i], origin)
            
        try: print(self.chromosomes[0].m_rate, self.chromosomes[0].m_step)
        except: pass
                
        self.bests.append(self.fitness.max())
        self.percentile0.append(np.percentile(self.fitness, 95))
        self.percentile1.append(np.percentile(self.fitness, 90))
        self.median.append(np.percentile(self.fitness, 50))
        
    def reproduce(self):
        '''Hier wird geschaut, wie viele Exemplare sich vermehren, evtl. wer mit wem. Die Kopulation und Mutation
        findet in der chromosome-Klasse statt.'''
        self.igeneration += 1
        ##self.adjust_fitness()
        
        # Welche sollen sich vermehren?
        
        
        
        oldchromosomes = copy.deepcopy(self.chromosomes)
        table = sorted(range(self.npop), key=lambda i: self.fitness[i], reverse=True)
        for i in range(0, KEEP):
            j = table[i]
            self.chromosomes[i] = copy.deepcopy(self.chromosomes[j])
            origin = dict()
            origin['kind'] = 'identical'
            origin['mother'] = len(self.bank)
            self.push_into_bank(self.chromosomes[i], origin)
        for i in range(KEEP, self.npop):
            j, j1 = np.random.randint(NUM_REPRODUCE, size=(2,))
            j, j1 = table[j], table[j1]
            # Selection
            self.chromosomes[i] = copy.deepcopy(oldchromosomes[j])
            # Crossover
            if np.random.rand() < CROSSOVER:
                origin = dict()
                origin['kind'] = 'crossover'
                origin['mother'] = oldchromosomes[j].ind
                origin['father'] = oldchromosomes[j1].ind
                self.chromosomes[i].cross(oldchromosomes[j1])
                self.push_into_bank(self.chromosomes[i], origin)
                imother = len(self.bank)-1 # last one added
            else:
                imother = j
            # Mutation
            self.chromosomes[i].mutate()
            origin = dict()
            origin['kind'] = 'mutation'
            origin['mother'] = imother
            self.push_into_bank(self.chromosomes[i], origin)
            
        try: print(self.chromosomes[0].m_rate, self.chromosomes[0].m_step)
        except: pass
                
        self.bests.append(self.fitness.max())
        self.percentile0.append(np.percentile(self.fitness, 95))
        self.percentile1.append(np.percentile(self.fitness, 90))
        self.median.append(np.percentile(self.fitness, 50))
        
    def plot(self):
        plt.plot(self.bests, '-', c='r', label='best')
        plt.plot(self.median, '-', c='b', label='Median')
        plt.plot(self.percentile0, '--', c='b', label='Percentile90&95')
        plt.plot(self.percentile1, '--', c='b')
        plt.title(f'Generation {self.igeneration}')
        plt.legend()
        #plt.show()