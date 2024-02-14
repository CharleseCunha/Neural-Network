from random import randrange, shuffle, choice






class gene_alg:


    def __init__(self, freq, iterations, obj_fun, parameters, dof, arguments=None, num_individ=8):

        # See if my genetic algorithm works
        self.size = len(parameters)
        self.num_individ = num_individ
        self.generations = [[],[]]
        self.freq = freq
        self.iterations = iterations
        self.obj = obj_fun
        self.parameters = parameters
        self.dof = dof
        self.arguments = arguments


    def objective_fun(self, genes, last=0):
        input_parm = []
        
        #print("Size of parameters is ",len(self.parameters))
        #print("Size of genes is ",len(genes))
        
        for i in range(self.size):
            #print("genes[%d] = %d" % (i, genes[i]))
            #print("len of parameters[%d] = %d" % (i, len(self.parameters[i])))
            input_parm.append(self.parameters[i][genes[i]])
        
        if(self.arguments is None):
            return self.obj(input_parm, last)
        else:
            return self.obj(self.arguments, input_parm, last)
        
        
        
    def print_generation(self, pop):

        for i in range(self.num_individ):
            print(pop[i])



    # Create first generation population
    def first_pop(self):
        # First create the list that will store the population
        population = []

        # Populate it with 8 random individuals
        for i in range(self.num_individ):

            l = []

            # Create the random individual
            for i in range(self.size):
                l.append(randrange(0, self.dof))
                
            # Add the individual
            population.append(l.copy())
        
        return population


        
    # Go through the individuals and automatically bring the fittest
    # To the next generation A.K.A. Elitism
    def find_elite(self, pop, choice):
        storage = []

        for i in range(self.num_individ):
            storage.append(self.objective_fun(pop[i]))
            
        best = 0

        for i in range(self.num_individ):
            if(storage[i] > storage[best]):
                best = i
        
        if(choice == 1):
            return pop[best]
        else:
            return best
        
        

    # Perform tournament styled brackets to select the best individuals
    def selection(self, pop):

        # Assuming that the population is a power of 2 greater than 2
        num_iter = int(self.num_individ / 2)
        
        # List to store the fittest
        fittest = []
        
        # Loop counter
        x = 0
        
        # Loop through the pop. to perform selections
        for i in range(num_iter):
            if(self.objective_fun(pop[x]) > self.objective_fun(pop[x + 1])):
                fittest.append(pop[x])
            else:
                fittest.append(pop[x + 1])
            
            # Increment the counter
            x += 2

        return fittest
        
        
        
    def mutation(self, pop):
        
        #print("---Before mutation:---")
        #print_generation(pop)
        
        # Find the indices of the best individual
        ind = self.find_elite(pop, 2)
        
        for i in range(self.freq):
        
            # Introduce genetic diversity
            rand_ind = randrange(0,self.num_individ)
            rand_gene = randrange(0,self.size)
            
            # Correct mutation if the elite is being touched
            if(rand_ind == ind):
                #print("We saved the elite!")
                rand_ind = (rand_ind + 1) % self.num_individ
            
            # Create the mutant gene
            mutant = randrange(-10, 10)
            
            #print("rand_ind = %d" % rand_ind)
            #print("rand_gene = %d" % rand_gene)
            #self.print_generation(pop)
            
            pop[rand_ind][rand_gene] = ((pop[rand_ind][rand_gene] + mutant) % self.dof)
                
        #print("---After mutation:---")
        #print_generation(pop)
            
        return pop
        
        
    # Generate new individuals from the fittest
    def procreate(self, pop):

        # Loop counter
        x = 0
        
        # Define the new population
        new_pop = []
        
        for i in range(int(self.num_individ / 4)):
            
            new_individ1 = []
            new_individ2 = []
            
            # Copy first half of the genes from parent1
            for i in range(int(self.size / 2)):
                new_individ1.append(pop[x][i])
                
            # Copy second half of the genes from parent2
            for i in range(int(self.size / 2)):
                new_individ1.append(pop[x + 1][i + int(self.size / 2)])
                
            # Copy first half of genes from parent2
            for i in range(int(self.size / 2)):
                new_individ2.append(pop[x + 1][i])

            # Copy second half of genes from parent1
            for i in range(int(self.size / 2)):
                new_individ2.append(pop[x][i + int(self.size / 2)])
        
            # Add the new individuals into the population
            new_pop.append(new_individ1)
            new_pop.append(new_individ2)
            
        new_gen = pop + new_pop
            
        return new_gen
        
        
        
    def find_sol(self):
        # Loop for evolution: Will stop iterating until the iterative convergence is met.

        # Create the first population
        self.generations[0] = self.first_pop()
        print("First population looks like:")
        self.print_generation(self.generations[0])
        print("best for the first gen. is %d", self.objective_fun(self.find_elite(self.generations[0], 1)))
        print("\n")
        x = 0
        
        for i in range(self.iterations):
            
            # Create second generation
            self.generations[1] = self.mutation(self.procreate(self.selection(self.generations[0])))
            shuffle(self.generations[1])
            
            # Add the new gen to the list
            self.generations[0] = self.generations[1]
            
            """
            print("***************************************")
            print_generation(generations[0])
            print("***************************************")
            """
            if(x == 500):
                print("Best obj: %f" % self.objective_fun(self.find_elite(self.generations[0], 1)))
                x = 0
                
            if(i == self.iterations - 1):
                print("Best obj: %f" % self.objective_fun(self.find_elite(self.generations[0], 1), 1))
               
            x += 1

        self.print_generation(self.generations[0])
        
        




if __name__ == "__main__":

    print("Oh Happy Days!")