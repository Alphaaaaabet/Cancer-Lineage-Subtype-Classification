#!/bin/bash
inputdir="../data/"
gene2idfile=$inputdir"gene2ind.txt"
cell2idfile=$inputdir"cell2ind.txt"
drug2idfile=$inputdir"drug2ind.txt"
testdatafile=$inputdir"drugcell_test.txt"

mutationfile_1=$inputdir"cell2mutation.txt"
mutationfile_2=$inputdir"chasmplus_v2_vector.txt"
drugfile=$inputdir"drug2fingerprint.txt"

modelfile="./Model_sample/model_final.pt"
resultdir="Result_sample"
hiddendir="Hidden_sample"

mkdir $resultdir
mkdir $hiddendir

#source activate pytorch3drugcellcpu

python -u ../code/predict_drugcell_cpu.py -gene2id $gene2idfile -cell2id $cell2idfile -drug2id $drug2idfile -genotype_1 $mutationfile_1 -genotype_2 $mutationfile_2 -fingerprint $drugfile -hidden $hiddendir -result $resultdir -predict $inputdir/drugcell_test.txt -load $modelfile > test_sample.log
