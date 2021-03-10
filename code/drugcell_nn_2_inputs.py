import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import argparse
import time

import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag


class drugcell_nn(nn.Module):
    ###TO DO: update train function to include extra arguments
    ###

    # Need to add terms for number of features, each feature input
    def __init__(self, term_size_map, term_direct_gene_map, dG, nfeatures,
                 ngene, root, num_hiddens_feature, num_hiddens_genotype,
                 num_cancer_types):

        super(drugcell_nn, self).__init__()

        self.root = root
        self.num_hiddens_genotype = num_hiddens_genotype  # = 6
        # self.num_hiddens_drug = num_hiddens_drug
        self.num_hiddens_feature = num_hiddens_feature

        # dictionary from terms to genes directly annotated with the term
        self.term_direct_gene_map = term_direct_gene_map

        # calculate the number of values in a state (term): term_size_map is the number of all genes annotated with the term
        self.cal_term_dim(term_size_map)

        # ngenes, gene_dim are the number of all genes
        # nfeatures is the number of biological features connected to each gene
        self.gene_dim = ngene
        # self.drug_dim = ndrug
        self.feat_dim = nfeatures

        # add modules for neural networks to process genotype features
        self.construct_direct_biofeature_layer()
        self.contruct_direct_gene_layer()
        self.construct_NN_graph(dG)

        # add modules for neural networks to process drugs	
        # self.construct_NN_drug()

        # add modules for final layer
        final_input_size = num_hiddens_genotype
        self.add_module('final_linear_layer',
                        nn.Linear(final_input_size, num_cancer_types))
        self.add_module('final_softmax_layer', nn.Softmax(num_cancer_types))

    # calculate the number of values in a state (term)
    # self.derm_dim_map = {GO_term: #_hidden_units}
    # Only for GO terms connected to genes
    # #_hidden_units is a hyperparam, default is set to 6
    def cal_term_dim(self, term_size_map):

        self.term_dim_map = {}

        for term, term_size in term_size_map.items():
            num_output = self.num_hiddens_genotype

            # log the number of hidden variables per each term
            num_output = int(num_output)
            self.term_dim_map[term] = num_output


    # TODO: Update dimensions to take feature inputs
    # Build a layer for forwarding muations that are directly annotated with a gene
    def construct_direct_biofeature_layer(self):
        """ Iterate through term_direct_gene_map.items() to get gene indices
            Add linear layer for each gene instance that takes in features
        """

        for term, gene_set in self.term_direct_gene_map.items():
            if len(gene_set) == 0:
                print(
                    f'There are no directed asscoiated features for {term} {gene_set}')
                sys.exit(1)

            for gene in gene_set:
                self.add_module(term + str(gene) + '_direct_feature_layer',
                                nn.Linear(self.feat_dim,
                                          self.num_hiddens_feature))
                # produces array [1, n_hiddens] for each mutation/feature

    # build a layer for forwarding gene that are directly annotated with the term
    def contruct_direct_gene_layer(self):
        """ 
                    
        """
        for term, gene_set in self.term_direct_gene_map.items():
            if len(gene_set) == 0:
                print('There are no directed asscoiated genes for', term)
                sys.exit(1)

            for gene in gene_set:
                # if there are some genes directly annotated with the term
                # add a layer taking in all genes and forwarding out only those genes 		
                self.add_module(term + str(gene) + '_direct_gene_layer',
                                nn.Linear(self.num_hiddens_feature,
                                          self.num_hiddens_genotype))
                self.add_module(term + str(gene) + '_batchnorm_layer',
                                nn.BatchNorm1d(self.num_hiddens_genotype))
                self.add_module(term + str(gene) + '_aux_linear_layer1',
                                nn.Linear(self.num_hiddens_genotype, 1))
                self.add_module(term + str(gene) + '_aux_linear_layer2',
                                nn.Linear(1, 1))

    # start from bottom (leaves), and start building a neural network using the given ontology
    # adding modules --- the modules are not connected yet
    def construct_NN_graph(self, dG):

        self.term_layer_list = []  # term_layer_list stores the built neural network
        self.term_neighbor_map = {}

        # term_neighbor_map records all children of each term	
        # Root node is hierarchy parent
        # dG.neighbors returns adjacent SUCCESSOR GO terms (not predecessor) for digraph
        # gives you downstream terms of current GO term
        for term in dG.nodes():
            self.term_neighbor_map[term] = []
            for child in dG.neighbors(term):
                self.term_neighbor_map[term].append(child)

        while True:
            leaves = [n for n in dG.nodes() if dG.out_degree(n) == 0]
            # leaves = [n for n,d in dG.out_degree().items() if d==0]
            # leaves = [n for n,d in dG.out_degree() if d==0]

            if len(leaves) == 0:
                break

            self.term_layer_list.append(leaves)

            # leaves are only terms connected to genes
            for term in leaves:

                # input size will be #chilren + #genes directly annotated by the term
                # 
                input_size = 0

                for child in self.term_neighbor_map[term]:
                    input_size += self.term_dim_map[child]

                # Multiply number_genes * hidden_dim for total feature length 
                if term in self.term_direct_gene_map:
                    input_size += len(self.term_direct_gene_map[
                                          term]) * self.num_hiddens_genotype

                # term_hidden is the number of the hidden variables in each state
                term_hidden = self.term_dim_map[term]

                self.add_module(term + '_linear_layer',
                                nn.Linear(input_size, term_hidden))
                self.add_module(term + '_batchnorm_layer',
                                nn.BatchNorm1d(term_hidden))
                self.add_module(term + '_aux_linear_layer1',
                                nn.Linear(term_hidden, 1))
                self.add_module(term + '_aux_linear_layer2', nn.Linear(1, 1))

            # remove leaves, next layer down now has out_degree == 0
            # iterator will repeat until rood node
            # breaks loop when no nodes left
            dG.remove_nodes_from(leaves)

    # definition of forward function

    def forward(self, x, y):
        """ Need to: 
                * add addn'l feature input
                * forward gene features into feature map
                * pass feature map activations to term+_direct_gene_layer
                
                x = input gene feature 1
                y = input gene feature 2
        """
        # x = input features from util.build_input_vector()
        # x = torch.Tensor.cuda() dims [batch_size, 5000], each row corresponds to a cell line

        # gene_input = Tensor of gene mutations [batch_size, 3008] each row is cell line
        # drug_input = Tensor of drug represen. [batch_size, 2000] each row is cell line

        # define forward function for GENOTYPE dcell FEATURES-FEATURES-FEATURES-FEATURES-FEATURES-FEATURES-FEATURES-

        # Nested dict to store{GO_term:{gene_ind: out_layer}, ...} 
        # term_gene_feature_out_map['GO_term'][gene_ind] accesses the stored activation array
        term_gene_feature_out_map = {}

        for term, gene in self.term_direct_gene_map.items():
            gene_input_1 = x[:, gene]
            gene_input_2 = y[:, gene]

            feat_input = torch.cat([gene_input_1, gene_input_2], dim=1)

            term_gene_feature_out_map[term] = {}
            term_gene_feature_out_map[term][gene] = self._modules[
                term + gene + '_direct_feature_layer'](feat_input)

        # define forward function for GENOTYPE dcell GENE-GENE-GENE-GENE-GENE-GENE-GENE-GENE-GENE-GENE-GENE-GENE-GENE-

        # Dict stores {GO_term: [linear_result1, ...]} for each gene connected to GO_term
        term_gene_out_map = {}
        gene_aux_map = {}

        #       Mutation Input   x          Layer dims
        # [batch_size, gene_dim] x [gene_dim, num_connected_genes] --> [batch_size, num_connected_genes]
        # generates {GO_term: [linear output tensor]}
        for term, gene_dict in self.term_gene_feature_out_map.items():

            for gene, activation in gene_dict.items():
                gene_layer_output = self._modules[
                    term + gene + '_direct_gene_layer'](activation)
                tanh_out = torch.tanh(gene_layer_output)
                term_gene_out_map[term] = self._modules[
                    term + gene + '_batchnorm_layer'](tanh_out)
                aux_layer1_out = torch.tanh(
                    self._modules[term + gene + '_aux_linear_layer1'](
                        term_gene_out_map[term]))
                gene_aux_map[term] = self._modules[
                    term + gene + '_aux_linear_layer2'](aux_layer1_out)

        # define forward function for ONTOLOGY dcell ONTOLOGY-ONTOLOGY-ONTOLOGY-ONTOLOGY-ONTOLOGY-ONTOLOGY-ONTOLOGY-

        term_NN_out_map = {}
        aux_out_map = {}

        for i, layer in enumerate(self.term_layer_list):

            for term in layer:

                child_input_list = []

                # Appends any existing direct child outputs from previous forward pass
                for child in self.term_neighbor_map[term]:
                    child_input_list.append(term_NN_out_map[child])

                if term in self.term_direct_gene_map:
                    child_input_list.append(term_gene_out_map[term])

                # Add layer before this for gene term
                child_input = torch.cat(child_input_list, 1)

                term_NN_out = self._modules[term + '_linear_layer'](child_input)

                Tanh_out = torch.tanh(term_NN_out)
                term_NN_out_map[term] = self._modules[
                    term + '_batchnorm_layer'](Tanh_out)
                aux_layer1_out = torch.tanh(
                    self._modules[term + '_aux_linear_layer1'](
                        term_NN_out_map[term]))
                aux_out_map[term] = self._modules[term + '_aux_linear_layer2'](
                    aux_layer1_out)

        # connect two neural networks at the top #################################################
        final_input = term_NN_out_map[self.root]

        out = self._modules['final_softmax_layer'](
            self._modules['final_linear_layer'](final_input))
        term_NN_out_map['final'] = out

        # aux_layer_out = torch.tanh(self._modules['final_aux_linear_layer'](out))
        aux_out_map['final'] = out

        return aux_out_map, term_NN_out_map
