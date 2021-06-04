import numpy as np
from random import choices 
import time
class DifferentialEvolution:
    def __init__(self, function = None, lower_bound = -100, upper_bound = 100, dim = 10, max_eval = 200000,
                 F = 0.8, CR = 0.7, pop_size = 100):
        '''
        Current class optimizes a function using Differential Evolutionary Algorithms.
        '''
        self.function = function # objective function
        self.lower_bound = lower_bound #lower value that each variable can have
        self.upper_bound = upper_bound #greatest value that each variable can have
        self.dim = dim # number of variables to consider
        self.max_eval = max_eval # maximum number of evaluations of objective function 
        self.F = F # scaling factor in mutation strategy
        self.CR = CR # crossover rate
        self.pop_size = pop_size # population size
        
    def __initialize(self):
        '''
        Initializes the population randomly
        '''
        self.__eval_counter = 0 # counter of evaluations in objective function
        self.__population = np.array([np.random.uniform(self.lower_bound, self.upper_bound, size = self.dim)\
            for _ in range(self.pop_size)]) # matrix with rows as random vector of given dimension within domain
    
    def __mutate(self, x , index):
        '''
        Determine donor vector as follows (rand/1):
        v = x_r1 + F(x_r2 - x_r3)
        '''
        random_ix = choices(range(0, self.pop_size), k = 4) #selecting 4 random indices
        
        #first, verify that given index is not in random_ix
        while index in random_ix and self.pop_size > 3:
            random_ix = choices(range(0, self.pop_size), k = 4) #selecting 4 random indices
        
        r1, r2, r3, delta = random_ix
        
        # Donor Vector
        
        v = self.__population[r1] + self.F*(self.__population[r2] - self.__population[r3])
        
        u = self.__binomial_crossover(v, x, delta)
        #verifying that trial vector doesn't violate boundaries:
        for i in range(self.dim):
            if u[i] < self.lower_bound:
                u[i] = self.lower_bound
            elif u[i] > self.upper_bound:
                u[i] = self.upper_bound
                
        return u
        
    def __binomial_crossover(self, v, x, delta):
        '''
        Returns trial vector ussing Binomial Crossover strategy
        '''
        
        u = []
        random_prob = np.random.uniform(0,1, size = self.dim) #random probabilities to evaluate crossover rate
        for i in range(self.dim):
            if random_prob[i] <= self.CR or i == delta:
                u.append(v[i])
            else:
                u.append(x[i])
        return np.array(u)
        
        
        
        
    def solve(self):
        self.__initialize() # initializing population
        shit = 0
        while shit < self.max_eval/(self.pop_size): #self.__eval_counter < self.max_eval:
            shit += 1
            #mutate individuals of current population
            trial_vectors =[self.__mutate(self.__population[i], i) for i in range(self.pop_size)]       
            
            #evaluate mutations respect to current population
            for i in range(self.pop_size):
                if self.function(trial_vectors[i]) < self.function(self.__population[i]):
                    self.__eval_counter += 2
                    self.__population[i] = trial_vectors[i] # natural selection of the best individual
                                                            # for the next generation
                else:
                    self.__population[i] = self.__population[i]
        
        # selecting the best of final population
        best_ix = np.argmin([self.function(self.__population[i]) for i in range(self.pop_size)])
        self.__eval_counter += 2
        #print(f"""
        #Best solution reached at x = {self.__population[best_ix]}.
        #Minimum value: {self.function(self.__population[best_ix])}.
        #""")
        return self.function(self.__population[best_ix]) # minimum value
        
             

        
    

