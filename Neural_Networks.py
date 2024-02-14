from random import randrange
import numpy as np
from Genetic_Algorithm import gene_alg

# Create a simple neural network to model y = x^2

# If you change the number of nodes in each hidden layer you have to change the format of changing the weights
# Of the nodes in the Nodez_in_output list inside the cost method. Also the number of generations used for the solver
# Has to be a multiple of four. The number of parameters has to even.

class Node:

    def __init__(self, input_sz, output_layer=False):
    
        self.input_sz = input_sz
        self.weights = []
        self.create()
        self.bias = 0
        self.output_layer = output_layer
        
        
    def __str__(self):
        return "This is a node with %d inputs." % self.input_sz
        
        
    def create(self):
        
        # Populate initial weights with 1
        for i in range(self.input_sz):
            self.weights.append(1)
            
            
    def change(self, weightz, bias):
        
        # Change weights for learning
        self.weights = weightz
        self.bias = bias
        
        
    def activation_funct(self, x):
    
        # Use a ReLU as the act_funct
        if(x <= 0):
            return (-(x**2))
        else:
            return (x**2)
            
            
    def evaluate(self, inputs):
        
        # Method to calculate the nodal function
        # Inputs parameter must be a list
        
        weighted_sum = 0
        
        #print("\n")
        #print("Weights are ... ", self.weights)
        #print("Inputs are ... ", inputs)
        #print("Bias is ... ", self.bias)
        #print("\n")
        
        # Calculate the weighted_sum
        for i in range(self.input_sz):
            weighted_sum += self.weights[i] * inputs[i]
        
        # Add the bias
        weighted_sum += self.bias
        
        # Return a value
        if(self.output_layer is False):
            return self.activation_funct(weighted_sum)
        elif(self.output_layer is True):
            return weighted_sum
        
        





class Neural_Network:

    def __init__(self, param_sz, output_sz, hidden_layers, num_nodes):
        
        self.param_sz = param_sz                    # Number of inputs
        self.output_sz = output_sz                  # Number of outputs
        self.output_nodes = []                      # List containing all of the output nodes
        self.outputs_f = []                         # List containing the  outputs from the output nodes
        self.hidden_layers = hidden_layers          # Number of hidden layers
        self.num_nodes = num_nodes                  # Number of nodes in each hidden layer
        self.first_layer = []                       # List containing the nodes of the first hidden layer
        self.activations_list = []                  # List containing the activations (node outputs) from the hidden layers
        self.hidden_layers_list = []                # List containing lists of the nodes in each hidden layer
        self.Nodez_in_layers = []                   # List containing all nodes in the hidden layers
        self.Nodez_in_output = []                   # List containing all the nodes in the output



    def setup_hidden_layers(self):

        # Layers list represents lists that represent hidden_layers
        # That are not part of the first hidden layer
        
        for i in range(self.hidden_layers):
            
            # Add a hidden layer
            self.hidden_layers_list.append([])
            
            for j in range(self.num_nodes):
                
                # Add the nodes into that hidden layer
                self.hidden_layers_list[i].append(Node(self.num_nodes))
                
        # Add hidden nodes to the Nodez list to access weights and biases
        for i in range(self.hidden_layers):
            for j in range(self.num_nodes):
                self.Nodez_in_output.append(self.hidden_layers_list[i][j])
            


    def first_hidden_layer(self):
    
        # Method to initialize the first hidden layer separate
        # From the others since these nodes will have inputs from the parameters

        # The first layer list represents the first hidden layer
        for i in range(self.num_nodes):

            # Populate the first layer with nodes
            self.first_layer.append(Node(self.param_sz))
        
        # Create List to have access to the Nodes
        for i in range(self.num_nodes):
            self.Nodez_in_layers.append(self.first_layer[i])
           
        #print("First hidden layer is ... ", self.first_hidden_layer)
        
        
        
    def create_output_nodes(self):
    
        # Method to initialize the output nodes
    
        # Create the list for the output nodes
        for i in range(self.output_sz):
            self.output_nodes.append(Node(self.num_nodes, True))
            self.Nodez_in_output.append(self.output_nodes[i])
        

        
    def print_nodes(self):
    
        # Method to print out nodes in a given layer
        
        # Takes a hidden layer list of nodes
        for i in range(self.num_nodes):
            print(self.first_layer[i])



    def calculate_first_layer(self, parameters):

        # Input the parameters into the nodes of the first layer
        # And then store their output into the activations list to be used
        # Either for the next hidden layer or the outputs
        l = []
        
        for i in range(self.num_nodes):
            l.append(self.first_layer[i].evaluate(parameters))
            
        self.activations_list.append(l)
        #print("Self activations are ... --- ", self.activations_list)



    def calculate_hidden_layer(self):
    
        # Method to calculate the hidden layers
        # And store the outputs into the activation list
        l = []
        
        # Loop to go index through all the hidden layers
        for i in range(self.hidden_layers):
        
        # Loop to index through all of the nodes in a hidden layer
            for j in range(self.num_nodes):
                l.append(self.hidden_layers_list[i][j].evaluate(self.activations_list[-1]))
                
        # Add l to the activations list
        self.activations_list.append(l)



    def function_output(self):

        # Will return a list with the activations of the output layer
        
        for i in range(self.output_sz):
            self.outputs_f.append(self.output_nodes[i].evaluate(self.activations_list[-1]))
            
        self.activations_list.clear()   # Clear the previous function scratch work for other evaluations
        #print("Output_f is ... ", self.outputs_f)
        output_list = self.outputs_f.copy()
        self.outputs_f.clear()
        
        return output_list           # Returning a list of the final outputs



    def initialize(self):
    
        # Method to initialize the neural network

        # Initialize the first layer
        self.first_hidden_layer()
        
        # Initialize the other hidden_layers if applicable
        self.setup_hidden_layers()
        
        # Initialize output layers
        self.create_output_nodes()



    def crunch(self, parameters):
        
        # Method to handle initialization and calculation parameters all in one
        
        # Calculate the first hidden_layer
        self.calculate_first_layer(parameters)
        
        # Calculate other hidden layers
        if(self.hidden_layers != 0):
            self.calculate_hidden_layer()
        
        # Calculate the neural network output
        return self.function_output()
        
        
        
    def print_weights_and_biases(self):
    
        # Method to print out the weights and biases
        for i in range(len(self.Nodez_in_layers)):
            print("Weights are is ", self.Nodez_in_layers[i].weights, " Bias is ", self.Nodez_in_layers[i].bias)
           
        print("\n")  
          
        for i in range(len(self.Nodez_in_output)):
            print("Weights are is ", self.Nodez_in_output[i].weights, " Bias is ", self.Nodez_in_output[i].bias)
        


    def cost_function(self, data, gene_param, last):

        # The last parameter is for printing the calculations
        # For the last population in the genetic algorithm
    
        # The objective function that we want to minimize
        funct_values = []
        total_cost = 0
        
        counter = 0
        
        # Loop to change weights and biases in the hidden layers
        for i in range(len(self.Nodez_in_layers)):
            self.Nodez_in_layers[i].change([gene_param[counter]], gene_param[counter + 1])
            counter += 2
        
        # Loop to change weights and biases in the output layers
        for i in range(len(self.Nodez_in_output)):
            self.Nodez_in_output[i].change([gene_param[counter], gene_param[counter+1], gene_param[counter+2]],
                                            gene_param[counter+3])
                                            #gene_param[counter+6], gene_param[counter+7], gene_param[counter+8],
                                            #gene_param[counter+9]], gene_param[counter+10])
            counter += 4
        
        # Loop to create list storing the output of the neural network with the
        # Specified training data values provided from the user.
        for i in range(data[1]):
            if last:
                print("Crunch value index %d = %f" % (i, self.crunch([data[0][i][0]])[0]))
            funct_values.append(self.crunch([data[0][i][0]]).copy())
            
        # Loop to add the square of the differences to the running total_cost
        for i in range(data[1]):
            add = (funct_values[i][0] - data[0][i][1])**2
            if last:
                print("(funct_val index %d: %f - data_val: %f)**2 = %f" % (i, funct_values[i][0], data[0][i][1], add))
            total_cost += add 
            
        # Calculate the mean square root
        average_cost = total_cost / data[1]
        if last:
            print("Average cost is ... ", average_cost)
            print("total_cost is %f / number of points %f = %f" % (total_cost, data[1], average_cost))
        return (-1 * average_cost)






resolution = 200
amp = 5

weights = []

for i in range(10):
    weights.append(np.linspace(-amp, amp, resolution))

parameterz = []

for i in range(10):
    parameterz.append(weights[i])


Training_set = []
for i in range(11):
    x_val = -5 + i
    y_val = x_val * x_val
    Training_set.append((x_val, y_val))


data = [Training_set, 11]


# Create the neural network instance
NN = Neural_Network(1, 1, 0, 3)
NN.initialize()


print(NN.crunch([1]))
print(NN.crunch([2]))
print(NN.crunch([3]))
print(NN.crunch([4]))
print(NN.crunch([5]))
print(NN.crunch([6]))
print(NN.crunch([7]))
print(NN.crunch([8]))
print(NN.crunch([9]))
print(NN.crunch([-3]))


# Create the solver object instance
solver = gene_alg(7, 100000, NN.cost_function, parameterz, resolution, data, 8) # freq=50 48 individuals worked well
solver.find_sol()

print("\n")

print("Crunch(-7) is ... ",NN.crunch([-7]))
print("Crunch(-6) is ... ",NN.crunch([-6]))
print("Crunch(-5) is ... ",NN.crunch([-5]))
print("Crunch(-4) is ... ",NN.crunch([-4]))
print("Crunch(-3.5) is ... ",NN.crunch([-3.5]))
print("Crunch(-3) is ... ", NN.crunch([-3]))
print("Crunch(-2.5) is ... ",NN.crunch([-2.5]))
print("Crunch(-2) is ... ",NN.crunch([-2]))
print("Crunch(-1) is ... ",NN.crunch([-1]))
print("Crunch(0) is ... ",NN.crunch([0]))
print("Crunch(1) is ... ",NN.crunch([1]))
print("Crunch(2) is ... ",NN.crunch([2]))
print("Crunch(2.5) is ... ",NN.crunch([2.5]))
print("Crunch(3) is ... ",NN.crunch([3]))
print("Crunch(-3.5) is ... ",NN.crunch([3.5]))
print("Crunch(4) is ... ",NN.crunch([4]))
print("Crunch(5) is ... ",NN.crunch([5]))
print("Crunch(6) is ... ",NN.crunch([6]))
print("Crunch(7) is ... ",NN.crunch([7]))

NN.print_weights_and_biases()