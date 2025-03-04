#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch, torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import itertools
from itertools import product
import matplotlib.pyplot as plt
import networkx as nx


# In[2]:


import random

def individual(bits=4):
    bases = 'AGCU'  # These are the bases we will use to generate our individuals
    return ''.join(random.choice(bases) for _ in range(bits))
    
def generator(size,bits):
    population = []
    for _ in range(size):
        combination = individual(bits)
        population.append(combination)
    return population

bases = ['A', 'U', 'G', 'C']
DOS = [''.join(p) for p in product(bases, repeat=3)]


# In[3]:


#It slices each individual in an Open Reading Frame so that it can be decodified.
def ORF_indv(individual):
    try:
        start_indices = [i for i in range(len(individual)) if individual.startswith('AUG', i)]
    except AttributeError as e:
        if str(e).startswith("'list' object has no attribute 'startswith'"):
            raise Exception("ORF_indv is for an individual, if you are trying on a list (population) please use ORF")
    else:
        sliced = []
        transcription_factors=[]
        start_indices = [i for i in range(len(individual)) if individual.startswith('AUG', i)] #AUG es el codón de inicio
        for start_index in start_indices:
            end_index = len(individual)
            for stop_codon in ['UAA', 'UAG', 'UGA', "AUG"]:   #Estos son los codones de STOP donde para la "transcripción"
                stop_index = individual.find(stop_codon, start_index + 3)
                if stop_index != -1 and stop_index < end_index:
                    end_index = stop_index + 3  # Include the stop codon
            sliced_individual = individual[start_index:end_index]
            sliced.append(sliced_individual)    
            #print("codon", individual[0:3])
        if individual[0:3] in DOS[0:8]:
            second_order=False
        else: 
            second_order=True
        transcription_factors.append(second_order)           
        #sliced corresponds to the scliced individuals
        return sliced, transcription_factors         
    
def ORF(population):        
    decodified_individuals=[]
    for individual in population:     #given each individual
        codons=[]
        second_order=[]
        for codon in ORF_indv(individual): #retrive the ORF from each individual AKA the parameters for the SNN.
            codons.append(codon) #This includes both the slices of the individuals plsu the transcription_factor
            
        decodified_individuals.append(codons)

    return decodified_individuals


# In[89]:


def mRNA(decodified_individuals=list):
    parameters = {}
    alpha = 0.9
    beta = 0.8
    bases = ['A', 'U', 'G', 'C']
    network_type=0
    
    for index, dna_strip in enumerate(decodified_individuals):
        indv=dna_strip[0]
        second_order=dna_strip[1]
        num_neurons = []
        neurons = []
        neuro_types = []
        for codon in indv:
            neurons.append(len(codon) * 4)  #número de neuronas por capa
            if len(codon) > 6:
                capa = codon[3:6]
                DOS = [''.join(p) for p in product(bases, repeat=3)]
                match capa:
                    case item if item in DOS[0:16]:
                        neuro_types.append("Alpha")
                    case item if item in DOS[16:32]:
                        neuro_types.append("Lapicque")
                    case item if item in DOS[32:48]:
                        neuro_types.append("Synaptic")
                    case item if item in DOS[48:65]:
                        neuro_types.append("Leaky")
                if codon[3:6] in DOS[0:32]:
                    network_type="Conv"
                else:             
                    network_type="Recurrente"
                parameters[index] = (neuro_types, neurons, len(neuro_types), network_type)
    return parameters


# In[82]:


class GAnet(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        neuro_type= parameters[0]
        num_neurons= parameters[1]
        num_layers= parameters[2] 
        network_type=parameters[3] ## Tipo de red convolucionada o linear
        ########Esto lo tengo que poner para que lo infiera de acuerdo al data set :S
        num_inputs = 784  # number of inputs
        num_outputs = 10  # number of classes
        N_in= 64
        in_channels=1
        H_in= 28
        W_in= 28
        ########
        self.num_layers= num_layers
        kernel_size= 5
        beta = 0.5
        beta1 = 0.8  # global decay rate for all leaky neurons in layer 1
        beta2 = torch.rand((num_outputs), dtype=torch.float)  # independent decay rate for each leaky neuron in layer 2: [0, 1)
        alpha = 0.999
        R = 1
        C = 1.44    
        V1 = 0.5 # shared recurrent connection
        V2 = torch.rand(num_outputs) # unshared recurrent connections
        spike_grad_lstm = surrogate.straight_through_estimator()
        spike_grad = surrogate.fast_sigmoid() # surrogate gradient
        
        
######## Recurrente
        if network_type == "Recurrente": #Si es un tipo de red neuronal lineal / recurrente
            # Creamos nuestra lista dinamica de capas
            self.layers = []  
            for i in range(num_layers):
######## ULTIMA CAPA                     
                if i == num_layers - 1: #Si es la última capa
                    if num_layers==1:  #Si solo hay una capa
                        self.layers.append(nn.Flatten())
                        self.layers.append(nn.Linear(num_inputs, num_outputs))
                    else: #Capa final con más de una capa 
                        self.layers.append(nn.Linear(num_neurons[i-1], num_outputs))  
                    match neuro_type[i]:
                        case "Alpha":
                            self.layers.append(snn.Alpha(alpha=alpha, beta=beta1, init_hidden=True, output=True))                                  
                        case "Leaky":
                            self.layers.append(snn.Leaky(beta=beta1, init_hidden=True, output=True))
                        case "Synaptic":
                            self.layers.append(snn.Synaptic(alpha=alpha, beta=beta1, init_hidden=True, output=True))
                        case "Lapicque":
                            self.layers.append(snn.Lapicque(R=R, C=C,init_hidden=True, output=True))      
                        case _:
                            self.layers.append(snn.Leaky(beta=beta1, init_hidden=True, output=True))
####### PRIMERA CAPA
                elif i == 0:
                    self.layers.append(nn.Flatten())
                    self.layers.append(nn.Linear(num_inputs, num_neurons[i]))
                    match neuro_type[i]:
                        case "Alpha":
                            self.layers.append(snn.Alpha(alpha=alpha, beta=beta1, init_hidden=True))
                        case "Synaptic":
                            self.layers.append(snn.Synaptic(alpha=alpha, beta=beta1, init_hidden=True))
                        case "Leaky":
                            self.layers.append(snn.Leaky(beta=beta1, learn_beta=True, init_hidden=True))
                        case "Lapicque":
                            self.layers.append(snn.Lapicque(beta=beta, init_hidden=True))
                        case "LeakyParallel":
                            self.layers.append(snn.LeakyParallel(input_size=num_inputs, hidden_size=num_hidden))
                        case _:
                            self.layers.append(snn.Leaky(beta=beta1, learn_beta=True, init_hidden=True))            
    
###### CAPA DE ENMEDIO        
                else:
                    self.layers.append(nn.Linear(num_neurons[i-1], num_neurons[i]))
                    match neuro_type[i]:
                        case "Synaptic":
                            self.layers.append(snn.Synaptic(alpha=alpha, beta=beta1, init_hidden=True))
                        case "Alpha":
                            self.layers.append(snn.Alpha(alpha=alpha, beta=beta2, init_hidden=True))
                        case "Leaky":
                            self.layers.append(snn.Leaky(beta=0.9, learn_beta=True, init_hidden=True))
                        case "Lapicque":
                            self.layers.append(snn.Lapicque(beta=beta, init_hidden=True)) 
                        case _:
                            self.layers.append(snn.Leaky(beta=beta1, learn_beta=True, init_hidden=True))
##############################################################################################################

        elif network_type == "Conv": #Si es una red convolucionada
            self.layers = []  
            for i in range(num_layers):
######## ULTIMA CAPA 
                print("CAPA NUMERO", i, "CAPAS:", num_layers )
                if i == num_layers - 1: #Si es la última capa
                    if num_layers==1: #Si solo hay una capa         
                        self.layers.append(nn.Conv2d(in_channels, num_neurons[i-1], kernel_size))
                        image_size= H_in - kernel_size +1
                        self.layers.append(nn.MaxPool2d(2))
                        image_size = image_size / 2  
                    else: #Capa final con más de una capa 
                        self.layers.append(nn.Flatten()),
                        image_size=int(round(image_size))
                        self.layers.append(nn.Linear(num_neurons[i-1] * (image_size * image_size), num_outputs))
                        print("pana", neuro_type, i)                             
                        match neuro_type[i]:
                            case "Alpha":
                                self.layers.append(snn.Alpha(alpha=alpha, beta=beta1, init_hidden=True, output=True))                                  
                            case "Leaky":
                                self.layers.append(snn.Leaky(beta=beta1, init_hidden=True, output=True))
                            case "Synaptic":
                                self.layers.append(snn.Synaptic(alpha=alpha, beta=beta1, init_hidden=True, output=True))
                            case "Lapicque":
                                self.layers.append(snn.Lapicque(R=R, C=C,init_hidden=True, output=True))      
                            case _:
                                self.layers.append(snn.Leaky(beta=beta1, init_hidden=True, output=True))


####### PRIMERA CAPA                
                elif i == 0:
                    self.layers.append(nn.Conv2d(in_channels, num_neurons[i], kernel_size))
                    image_size= H_in - kernel_size +1
                    self.layers.append(nn.MaxPool2d(2))
                    image_size = image_size / 2 
                    match neuro_type[i]:
                        case "Alpha":
                            self.layers.append(snn.Alpha(alpha=alpha, beta=beta1, init_hidden=True))
                        case "Synaptic":
                            self.layers.append(snn.Synaptic(alpha=alpha, beta=beta1, init_hidden=True))
                        case "Leaky":
                            self.layers.append(snn.Leaky(beta=beta1, learn_beta=True, init_hidden=True))
                        case "Lapicque":
                            self.layers.append(snn.Lapicque(beta=beta, init_hidden=True))
                        case "LeakyParallel":
                            self.layers.append(snn.LeakyParallel(input_size=num_inputs, hidden_size=num_hidden))
                        case _:
                            self.layers.append(snn.Leaky(beta=beta1, learn_beta=True, init_hidden=True))

                    
###### CAPA DE ENMEDIO        
                else:
                    self.layers.append(nn.Conv2d(num_neurons[i-1], num_neurons[i],  kernel_size)),
                    image_size= image_size - kernel_size +1
                    self.layers.append(nn.MaxPool2d(2)),
                    image_size = image_size / 2 
                    match neuro_type[i]:
                        case "Synaptic":
                            self.layers.append(snn.Synaptic(alpha=alpha, beta=beta1, init_hidden=True))
                        case "Alpha":
                            self.layers.append(snn.Alpha(alpha=alpha, beta=beta2, init_hidden=True))
                        case "Leaky":
                            self.layers.append(snn.Leaky(beta=0.9, learn_beta=True, init_hidden=True))
                        case "Lapicque":
                            self.layers.append(snn.Lapicque(beta=beta, init_hidden=True)) 
                        case _:
                            self.layers.append(snn.Leaky(beta=beta1, learn_beta=True, init_hidden=True))


# In[8]:


cache = {}
def accuracy_OF(population):
    print(population)
    funcion_obj=[]
    if isinstance(population, list): ##si es una lista pasalo así normal :)
        for i in range(len(population)):
            if population[i] in cache:
                funcion_obj.append(cache[population[i]] )
                continue
            else:
                try:
                    red=(GAnet(mRNA(ORF(population))[i]))
                    net=nn.Sequential(*red.layers)
                    res=(entrenar(net,printear=False))
                    cache[population[i]] = res
                    funcion_obj.append(res)
                except:
                    funcion_obj.append(0)
                    continue
    else:       ##Si es un individuo
        if population in cache:  ##Revisa si ya está en caché
                return cache[population] 
        try:   ##Mide el accuracy 
            red=(GAnet(mRNA(ORF(population))))
            net=nn.Sequential(*red.layers)
            res=(entrenar(net,printear=False))
            cache[population[i]] = res
            funcion_obj.append(res)
            
        except:
            funcion_obj.append(0)
    print("Accuracy:", funcion_obj)
            
    return funcion_obj

def One_Max(c):
    value=sum(c)
    return value
