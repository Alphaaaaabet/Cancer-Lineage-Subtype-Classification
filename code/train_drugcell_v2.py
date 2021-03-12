import sys
import os
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from util import *
from drugcell_nn_2_inputs import *
import argparse
import numpy as np
import time


# build mask: matrix (nrows = number of relevant gene set, ncols = number all genes)
# elements of matrix are 1 if the corresponding gene is one of the relevant genes
def create_term_mask(term_direct_gene_map, gene_dim):
    term_mask_map = {}

    for term, gene_set in term_direct_gene_map.items():

        print("gene_dim type:", type(gene_dim))
        print("gene_dim: ", gene_dim)

        mask = torch.zeros(len(gene_set), gene_dim)

        for i, gene_id in enumerate(gene_set):
            mask[i, gene_id] = 1

        mask_gpu = torch.autograd.Variable(mask.cuda(CUDA_ID))

        term_mask_map[term] = mask_gpu

    return term_mask_map

def accuracy(output, label):
    total = output.shape[0]
    preds = torch.argmax(output, dim=1).T
    acc = torch.eq(preds, label).sum().item()/total
    return acc

# TODO: cell_features_1 and cell_features_2 are not used. Modity this function to use them.
def train_model(root, term_size_map, term_direct_gene_map, dG, train_data,
                nfeatures, gene_dim, model_save_folder, train_epochs,
                batch_size, learning_rate, num_hiddens_genotype,
                cell_features_1, cell_features_2, num_cancer_types):
    epoch_start_time = time.time()
    
    best_model_idx = 0
    
    train_loss_list_for_graphing = [] 
    test_loss_list_for_graphing = [] 
    test_acc_list_for_graphing = []

    # dcell neural network
    model = drugcell_nn(term_size_map, term_direct_gene_map, dG, nfeatures,
                        gene_dim, root, num_hiddens_features,
                        num_hiddens_genotype, num_cancer_types)
    
    best_model = model

    train_feature, train_label, test_feature, test_label = train_data

    model.cuda(CUDA_ID)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 betas=(0.9, 0.99), eps=1e-05)
    #term_mask_map = create_term_mask(model.term_direct_gene_map, gene_dim)

    optimizer.zero_grad()

    for name, param in model.named_parameters():
        # Removed masking as no longer needed, keeping weight reduction
        #term_name = name.split('_')[0]

        #if '_direct_gene_layer.weight' in name:
        #    param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
        #else:
        #    param.data = param.data * 0.1
        param.data = param.data * 0.1

    train_loader = du.DataLoader(du.TensorDataset(train_feature, train_label),
                                 batch_size=batch_size, shuffle=False)
    test_loader = du.DataLoader(du.TensorDataset(test_feature, test_label),
                                batch_size=batch_size, shuffle=False)

    best_loss = 10.0
    loss = nn.CrossEntropyLoss()
    
    for epoch in range(train_epochs):

        # Train
        model.train()
        train_predict = torch.zeros(0, 0).type(torch.FloatTensor).cuda(CUDA_ID)
        total_loss = 0
        total_test_loss = 0

        for i, (inputdata, labels) in enumerate(train_loader):

            # Convert torch tensor to Variable
            # features = [batch_size, 5000] feature tensor with cell/drug features
            # Convert torch tensor to Variable
            #print("features_1:", len(cell_features_1))
            features_1 = build_input_vector(inputdata, cell_features_1)
            #print("features_2:", len(cell_features_2))
            features_2 = build_input_vector(inputdata, cell_features_2)
            labels = torch.flatten(labels).type(torch.LongTensor)
            
            # cuda_features 
            cuda_features_1 = torch.autograd.Variable(features_1.cuda(CUDA_ID))
            cuda_features_2 = torch.autograd.Variable(features_2.cuda(CUDA_ID))
            cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer

            # Here term_NN_out_map is a dictionary
            aux_out_map, term_NN_out_map = model(cuda_features_1, cuda_features_2)

            if train_predict.size()[0] == 0:
                train_predict = term_NN_out_map['final'].data
            else:
                train_predict = torch.cat(
                    [train_predict, term_NN_out_map['final'].data], dim=0)

            train_loss = 0
            for name, output in term_NN_out_map.items():
                if name == 'final':
                    train_loss += loss(output, cuda_labels)

            train_loss.backward()

            total_loss += train_loss / len(train_loader)

            optimizer.step()

        if (epoch+1) % 10 == 0:
            torch.save(model,
                       model_save_folder + '/model_' + str(epoch+1) + '.pt')

        # Test: random variables in training mode become static
        model.eval()

        test_predict = torch.zeros(0, 0).cuda(CUDA_ID)

        acc = 0.0
        for i, (inputdata, labels) in enumerate(test_loader):
            # Convert torch tensor to Variable
            features_1 = build_input_vector(inputdata, cell_features_1)
            features_2 = build_input_vector(inputdata, cell_features_2)
            labels = torch.flatten(labels).type(torch.LongTensor)

            # cuda_features 
            cuda_features_1 = torch.autograd.Variable(features_1.cuda(CUDA_ID))
            cuda_features_2 = torch.autograd.Variable(features_2.cuda(CUDA_ID))
            cuda_labels = Variable(labels.cuda(CUDA_ID))

            aux_out_map, term_NN_out_map = model(cuda_features_1, cuda_features_2)

            if test_predict.size()[0] == 0:
                test_predict = term_NN_out_map['final'].data
            else:
                test_predict = torch.cat(
                    [test_predict, term_NN_out_map['final'].data], dim=0)

            test_loss = 0
            a = 0.0
            for name, output in term_NN_out_map.items():
                if name == 'final':
                    test_loss += loss(output, cuda_labels)
                    a += accuracy(output, cuda_labels)

            acc += a / len(test_loader)
            total_test_loss += test_loss / len(test_loader)

        epoch_end_time = time.time()
        print(
            "epoch\t%d\tcuda_id\t%d\ttotal_loss\t%.6f\ttest_loss\t%.6f\taccuracy\t%.6f\telapsed_time\t%s" % (
            epoch, CUDA_ID, total_loss, total_test_loss, acc, 
            epoch_end_time - epoch_start_time))
        
        train_loss_list_for_graphing.append(total_loss) 
        test_loss_list_for_graphing.append(total_test_loss)
        test_acc_list_for_graphing.append(acc)
    
        epoch_start_time = epoch_end_time

        if best_loss > total_test_loss:
            best_model_idx = epoch
            best_model = model
            best_loss = total_test_loss

    torch.save(best_model, model_save_folder + '/model_final.pt')

    print("Best performed model (epoch)\t%d" % best_model_idx)
    return train_loss_list_for_graphing, test_loss_list_for_graphing, test_acc_list_for_graphing


parser = argparse.ArgumentParser(description='Train dcell')
parser.add_argument('-onto',
                    help='Ontology file used to guide the neural network',
                    type=str)
parser.add_argument('-train', help='Training dataset', type=str)
parser.add_argument('-test', help='Validation dataset', type=str)
parser.add_argument('-epoch', help='Training epochs for training', type=int,
                    default=300)
parser.add_argument('-lr', help='Learning rate', type=float, default=0.01)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=5000)
parser.add_argument('-modeldir', help='Folder for trained models', type=str,
                    default='MODEL/')
parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)
parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str)
parser.add_argument('-drug2id', help='Drug to ID mapping file', type=str)
parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str)

parser.add_argument('-feature_hiddens',
                    help='Mapping for the number of neurons in each term in genotype feature inputs',
                    type=int, default=6)
parser.add_argument('-genotype_hiddens',
                    help='Mapping for the number of neurons in each term in genotype parts',
                    type=int, default=6)
parser.add_argument('-drug_hiddens',
                    help='Mapping for the number of neurons in each layer',
                    type=str, default='100,50,6')
parser.add_argument('-final_hiddens',
                    help='The number of neurons in the top layer', type=int,
                    default=6)
parser.add_argument('-num_cancer_types', help='The number of cancer types',
                    type=int, default=92)

parser.add_argument('-genotype_1', help='Mutation information for cell lines',
                    type=str)
parser.add_argument('-genotype_2', help='Mutation information for cell lines',
                    type=str)
parser.add_argument('-fingerprint',
                    help='Morgan fingerprint representation for drugs',
                    type=str)

# call functions
opt = parser.parse_args()
torch.set_printoptions(precision=5)

# load input data
train_data, cell2id_mapping = prepare_train_data(opt.train, opt.test,
                                                 opt.cell2id)
gene2id_mapping = load_mapping(opt.gene2id)

# load cell/drug features
# cell features = [cell_line, gene] mutation array
# drug features = [cell_line, drug] fingerprint array
cell_features_1 = np.genfromtxt(opt.genotype_1, delimiter=',')
cell_features_2 = np.genfromtxt(opt.genotype_2, delimiter=',')

num_cells = len(cell2id_mapping)
num_genes = len(gene2id_mapping)

# load ontology
dG, root, term_size_map, term_direct_gene_map = load_ontology(opt.onto,
                                                              gene2id_mapping)

# load the number of hiddens #######
num_hiddens_genotype = opt.genotype_hiddens
num_cancer_types = opt.num_cancer_types
num_hiddens_final = opt.final_hiddens

num_hiddens_features = opt.feature_hiddens
#####################################

CUDA_ID = opt.cuda

# TODO: Check this. Not sure what should gene_dim be. Need to figure out the correct value.
gene_dim = 6

# TODO: Make sure the arguments are in correct order.
train_loss, test_loss, acc = train_model(root, term_size_map, term_direct_gene_map, dG, train_data,
            2, num_genes, opt.modeldir, opt.epoch, opt.batchsize, opt.lr,
            num_hiddens_features, cell_features_1, cell_features_2, num_cancer_types)

plt.figure(figsize=(7,5), dpi=250)
plt.plot(range(len(train_loss)), train_loss, c='b', label='Train Loss')
plt.plot(range(len(train_loss)), test_loss, c='orange', label='Val. Loss')
plt.title('Train vs. Val. Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Average Loss', fontsize=16)
plt.ylim(0, max(test_loss) + .25)
plt.legend()
plt.savefig(model_save_folder+'train_val_loss', dpi=1200)

plt.figure(figsize=(7,5), dpi=250)
plt.plot(range(len(train_loss)), acc, c='orange', label='Acc.')
plt.title('Accuracy', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.ylim(0, max(acc) + .25)
plt.savefig(model_save_folder+'test_acc', dpi=1200)