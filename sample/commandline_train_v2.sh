#!/bin/bash
inputdir="../data/"
gene2idfile=$inputdir"gene2ind.txt"
cell2idfile=$inputdir"cell2ind.txt"
drug2idfile=$inputdir"drug2ind.txt"
traindatafile=$inputdir"targets_train.txt"
valdatafile=$inputdir"targets_test.txt"
ontfile=$inputdir"drugcell_ont.txt"

mutationfile_1=$inputdir"cell2mutation.txt"
mutationfile_2=$inputdir"cell2mutation.txt"
drugfile=$inputdir"drug2fingerprint.txt"

cudaid=0

modeldir=model_sample
rm -rf $modeldir
mkdir $modeldir #PW CHANGE commented out

#source activate pytorch3drugcell #PW CHANGE commented out

python -u ../code/train_drugcell_v2.py -onto $ontfile -gene2id $gene2idfile -drug2id $drug2idfile -cell2id $cell2idfile -train $traindatafile -test $valdatafile -model $modeldir -cuda $cudaid -genotype_1 $mutationfile_1 -genotype_2 $mutationfile_2 -fingerprint $drugfile -feature_hiddens 6 -genotype_hiddens 6 -drug_hiddens '100,50,6' -final_hiddens 6 -epoch 100 -batchsize 5000 -num_cancer_types 92