



#### NUR FÜR EINPARKEN NÖTIG!! NICHT FÜRS HAUPTPROJEKT! ####




import numpy as np
import matplotlib.pyplot as plt
import copy


VERBOSE = False # | True


if VERBOSE:
    print_ = print
else:
    print_ = lambda *args: None

    
    
    
# Chromosome Prototyp-Klasse

class chromosome_():
    def __init__(self, nin, nhid, nout, fs):
        if type(nhid) not in [list, tuple]: nhid = [nhid]
        assert len(nhid)+1 == len(fs)
        self.Ws = []
        self.bs = []
        self.fs = fs
        for n0, n1 in zip([nin] + nhid, nhid + [nout]):
            self.Ws.append( np.random.normal(0., 0.5, (n0, n1)) )
            self.bs.append( np.random.normal(0., 0.5, n1) )
            
    def act(self, input):
        a = input
        for W, b, f in zip(self.Ws, self.bs, self.fs):
            a = f(a @ W + b)
        return a
    
    def cross(self, other):
        for M, M1 in zip(self.Ws + self.bs, other.Ws + other.bs):
            #t = np.random.rand(*M.shape) > 0.5 # CHOICE
            t = np.random.rand(*M.shape)       # WEIGHTED MEAN
            #t = np.random.rand(*M.shape)*2-0.5 # WEIGHTED MEAN AND EXTRAPOLATION
            M *= t
            M += (1-t)*M1
    
    def mutate(self):
        for M in self.Ws + self.bs:
            M += MUTATION_STEP * np.random.normal(size=M.shape) * (np.random.rand(*M.shape) > MUTATION_RATE)

            
            
# Genpool-Klasse
            

class genpool():
    def __init__(self, chrom_class, num_pop, num_reproduce, num_keep=0, crossover_rate=0.5, new_rate=0.1, num_species=1,
                                                                     intersp_cross_over_rate=0.1):
        '''Genpool class for reproducing chromosomes (and plotting their fitnesses)
        chrom_class: The class of chromosomes, must have a mutate function (and crossover if wished).
        num_pop: Size of the population.
        num_reproduce: Number of individuals which survive in each epoch due to their fitness
        num_keep: Number of best individuals which are just copied without any mutation (don´t throw best solution away).
        crossover_rate: Rate of how many individuals are created by crossover (of those who arent kept or made new)
        num_species: Number of species whose individuals interact primarily with themselves
        '''
        self.chrom_class = chrom_class
        self.num_pop     = num_pop
        self.num_keep    = num_keep
        self.num_reproduce = num_reproduce
        self.num_species = num_species
        self.crossover_rate = crossover_rate
        self.new_rate = new_rate
        self.intersp_cross_over_rate = intersp_cross_over_rate
        self.per_species = int(self.num_pop // self.num_species)
        assert self.per_species*self.num_species == self.num_pop, 'Make sure num_pop is dividible by num_species'
        assert crossover_rate==0. or intersp_cross_over_rate < crossover_rate, 'You probably don´t wanna absord the intra species co rate into inter co!'
        
        self.chromosomes = [ chrom_class() for i in range(num_pop) ]
        self.fitness = np.zeros(num_pop)
        self.bank = []
        
        # plot stuff
        self.igeneration = 0 # iterates the generations
        self.bests = []
        self.percentiles = []
        self.median = []
        
        if self.bank is not None:
            for c in self.chromosomes:
                origin = {'kind': 'new'}
                self.push_into_bank(c, origin)
            
        
    def push_into_bank(self, c, origin=None):
        '''Erlaubt es, chromosom Information aufzuheben. Erstellt eine deepcopy.'''
        cc = c # keine deep copy notwendig?
        #cc = copy.deepcopy(c)
        if origin:  cc.origin  = origin
        cc.igeneration = self.igeneration
        self.bank.append(cc)
        c.ind = len(self.bank)-1
        
        
    def set_fitness(self, i, fit):
        'Sets the fitness of the i´th chromosomes'
        self.fitness[i] = fit
        if self.bank:
            k = self.chromosomes[i].ind
            self.bank[k].fitness = fit
        
        
    def reproduce_SAVED(self):
        '''Hier wird geschaut, wie viele Exemplare sich vermehren, evtl. wer mit wem. Die Kopulation und Mutation
        findet in der chromosome-Klasse statt.'''
            
        # statistics
        self.bests.append(self.fitness.max())
        self.percentiles.append([ np.percentile(self.fitness, 5*k) for k in range(0, 20) ])
        self.median.append(np.percentile(self.fitness, 50))
        
        # sort
        self.igeneration += 1
        oldchromosomes = copy.deepcopy(self.chromosomes)
        table = sorted(range(self.num_pop), key=lambda i: self.fitness[i], reverse=True) # table[i] ist der i´t-beste Index.
        num_new = round_prob(self.num_pop * self.new_rate)
        
        # New Generation
        
        # Individuals which are kept
        for i in range(0, self.num_keep):
            print_(i, 'kept')
            j = table[i]
            self.chromosomes[i] = copy.deepcopy(oldchromosomes[j])
            if self.bank is not None:
                origin = {'kind': 'identical', 'mother': oldchromosomes[j].ind}
                self.push_into_bank(self.chromosomes[i], origin)
                
        # Individuals crossover and mutation
        for i in range(self.num_keep, self.num_pop-num_new):
            k0, k1 = np.random.randint(self.num_reproduce, size=(2,))
            j0, j1 = table[k0], table[k1]
            # Selection
            c = copy.deepcopy(oldchromosomes[j0])
            print_(i, 'taken from', j0)
            # Crossover
            if np.random.rand() < self.crossover_rate:
                print_(i, 'crossover with', j1)
                c.cross(oldchromosomes[j1])
                if self.bank is not None:
                    origin = {'kind': 'crossover', 'mother': oldchromosomes[j0].ind, 'father': oldchromosomes[j1].ind}
                    self.push_into_bank(c, origin)
            # Mutation
            print_(i, 'mutates')
            self.chromosomes[i].mutate()
            if self.bank is not None:
                origin = {'kind': 'mutation', 'mother': len(self.bank)-1}
                self.push_into_bank(c, origin)
            self.chromosomes[i] = c
            
        # Individuals creation, new ones
        for i in range(self.num_pop-num_new, self.num_pop):
            print_(i, 'created new')
            self.chromosomes[i] = self.chrom_class()
            if self.bank is not None:
                origin = {'kind': 'new'}
                self.push_into_bank(self.chromosomes[i], origin)
        
        
        
    def reproduce(self):
        '''Hier wird geschaut, wie viele Exemplare sich vermehren, evtl. wer mit wem. Die Kopulation und Mutation
        findet in der chromosome-Klasse statt.'''
            
        # statistics
        ##self.bests.append(self.fitness.max())
        ##self.percentiles.append([ np.percentile(self.fitness, 5*k) for k in range(0, 20) ])
        ##self.median.append(np.percentile(self.fitness, 50))
        
        
        # statistics with species
        self.bests.append([self.fitness[self.per_species * i : self.per_species * (i+1)].max()
                           for i in range(self.num_species) ])
        self.percentiles.append([ np.percentile(self.fitness, 5*k) for k in range(0, 20) ])
        self.median.append(np.percentile(self.fitness, 50))
        
        # sort
        self.igeneration += 1
        oldchromosomes = copy.deepcopy(self.chromosomes)
        # For only one species
        ##table = sorted(range(self.num_pop), key=lambda i: self.fitness[i], reverse=True) # table[i] ist der i´t-beste Index.
        
        table = [ sorted(range(self.per_species * i, self.per_species * (i+1)), key=lambda i: self.fitness[i], reverse=True) 
                                                   for i in range(self.num_species) ] # table[i] ist der i´t-beste Index.
        
        print_('Best tables:')
        for s in range(self.num_species):
            for i in range(self.per_species):
                j = table[s][i]
                print_(j, self.fitness[j])
        
        # New Generation
        for s in range(self.num_species):
            print_('---- Species #%s ----'%s)
            offset = s * self.per_species
            num_new = round_prob(self.num_pop/self.num_species * self.new_rate)
            # Individuals which are kept
            for i in range(offset, offset + self.num_keep):
                j = table[s][i-offset]
                print_(i, 'kept', j)
                self.chromosomes[i] = copy.deepcopy(oldchromosomes[j])
                if self.bank is not None:
                    origin = {'kind': 'identical', 'mother': oldchromosomes[j].ind}
                    self.push_into_bank(self.chromosomes[i], origin)

            # Individuals crossover and mutation
            for i in range(offset + self.num_keep, offset + self.per_species - num_new):
                k0 = np.random.randint(self.num_reproduce // self.num_species)
                j0 = table[s][k0]
                # Selection
                c = copy.deepcopy(oldchromosomes[j0])
                print_(i, 'taken from', j0)
                # Crossover
                rnd = np.random.rand()
                if rnd < self.crossover_rate:
                    if rnd < self.intersp_cross_over_rate and self.num_species > 1:
                        s1 = np.random.choice([ ss for ss in range(self.num_species) if ss!=s ])
                        k1 = np.random.randint(self.num_reproduce)
                        j1 = table[s1][k1]
                        print_(i, 'interspecies crossover with', j1, 'of species', s1)
                    else:
                        s1 = s # same species
                        k1 = np.random.randint(self.num_reproduce)
                        j1 = table[s1][k1]
                        print_(i, 'crossover with', j1)
                        
                    c.cross(oldchromosomes[j1])
                    if self.bank is not None:
                        origin = {'kind': 'crossover', 'mother': oldchromosomes[j0].ind, 'father': oldchromosomes[j1].ind}
                        self.push_into_bank(c, origin)
                # Mutation
                print_(i, 'mutates')
                self.chromosomes[i].mutate()
                if self.bank is not None:
                    origin = {'kind': 'mutation', 'mother': len(self.bank)-1}
                    self.push_into_bank(c, origin)
                self.chromosomes[i] = c

            # Individuals creation, new ones
            for i in range(offset + self.per_species-num_new, offset + self.per_species):
                print_(i, 'created new')
                self.chromosomes[i] = self.chrom_class()
                if self.bank is not None:
                    origin = {'kind': 'new'}
                    self.push_into_bank(self.chromosomes[i], origin)
        
        
    def plot(self, show=True):
                
        # test:
        bests = np.array(self.bests)
        plt.plot(bests, '-', c='r', label='best')
                          
        # ohne test:
        #plt.plot(self.bests, '-', c='r', label='best')
                          
                          
        percentiles = np.array(self.percentiles)
        #plt.plot(self.median, '-', c='b', label='Median')
        plt.plot(percentiles[:,50//5], '-', c='b', label='Median')
        plt.plot(percentiles[:,95//5], '--', c='b', label='Percentile90&95')
        plt.plot(percentiles[:,90//5], '--', c='b')
        plt.title(f'Generation {self.igeneration}')
        plt.legend()
        if show:
            plt.show()
            
            
def round_prob(x):
    'Rounds x to an integer n such that E(n) = x.'
    n = int(x//1)
    t = x-n # between 0 and 1
    if np.random.rand() < t:
        n+=1
    return n