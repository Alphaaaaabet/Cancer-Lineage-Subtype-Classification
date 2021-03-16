#!/bin/bash

# Changed to drugcell_nn_v2 with feature layer nonlinear activation
# Changed feature hiddens to 18
# Changed Genotype Hiddens to 22
# Changed lr = 0.006

inputdir="../data/"
gene2idfile=$inputdir"gene2ind.txt"
cell2idfile=$inputdir"cell2ind.txt"
drug2idfile=$inputdir"drug2ind.txt"
traindatafile=$inputdir"xval/targets_train_${1}.txt"
valdatafile=$inputdir"xval/targets_test_${1}.txt"
ontfile=$inputdir"drugcell_ont.txt"

mutationfile_1=$inputdir"Vest_full_vector.txt"
mutationfile_2=$inputdir"chasmplus_v2_vector.txt"
drugfile=$inputdir"drug2fingerprint.txt"

cudaid=0

modeldir=2in_ft_xval_$1
#rm -rf $modeldir
mkdir -p xval/$modeldir #PW CHANGE commented out

source activate pytorch3drugcell #PW CHANGE commented out

python -u ../code/train_drugcell_v2_ft_act.py -onto $ontfile -gene2id $gene2idfile -drug2id $drug2idfile -cell2id $cell2idfile -train $traindatafile -test $valdatafile -model $modeldir -cuda $cudaid -genotype_1 $mutationfile_1 -genotype_2 $mutationfile_2 -fingerprint $drugfile -feature_hiddens 18 -genotype_hiddens 22 -drug_hiddens '100,50,6' -final_hiddens 6 -epoch 100 -batchsize 5000 -lr 0.006 > xval/$modeldir/2in_ft_xval_${1}.log
