#!/bin/bash
inputdir="../data/"
gene2idfile=$inputdir"gene2ind.txt"
cell2idfile=$inputdir"cell2ind.txt"
drug2idfile=$inputdir"drug2ind.txt"
traindatafile=$inputdir"drugcell_test.txt"
valdatafile=$inputdir"drugcell_val.txt"
ontfile=$inputdir"drugcell_ont.txt"

mutationfile_1=$inputdir"cell2mutation.txt"
mutationfile_2=$inputdir"cell2mutation.txt"
drugfile=$inputdir"drug2fingerprint.txt"

cudaid=0

modeldir=Model_sample
mkdir $modeldir #PW CHANGE commented out

#source activate pytorch3drugcell #PW CHANGE commented out

python -u ../code/train_drugcell.py -onto $ontfile -gene2id $gene2idfile -drug2id $drug2idfile -cell2id $cell2idfile -train $traindatafile -test $valdatafile -model $modeldir -cuda $cudaid -genotype_1 $mutationfile_1 -genotype_2 $mutationfile_2 -fingerprint $drugfile -feature_hiddens 6 -genotype_hiddens 6 -drug_hiddens '100,50,6' -final_hiddens 6 -epoch 100 -batchsize 5000 #> train_sample.log