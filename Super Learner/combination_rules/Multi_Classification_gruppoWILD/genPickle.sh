#!/bin/bash

mkdir -p ./output ./temp
rm -f ./CSVs/CICIDS201*
rm -f ./CSVs/.C*
rm -f ./CSVs/.~*


echo "Down Sampling..."
python3 ./script/down_sampling.py
echo "Down Sampling done!"
rm -f ./temp/temp_*

printf "\n\n"
echo "Merging of files:"
python3 ./script/mergeDataset_script.py
echo "Merge done!"

#rm -f ./temp/final_*
rm -f -r ./temp

printf "\n\n"
echo "Deleting rows..."
python3 ./script/delRow.py
echo "Delete done!"
printf "\n\n"
echo "Deleting columns..."
cut -d',' -f1-7,62 --complement ./CSVs/CICIDS2017_edit.csv > ./CSVs/CICIDS2017_corrected.csv
echo "Delete done!"

rm -f ./CSVs/CICIDS2017_edit.csv ./CSVs/CICIDS2017.csv

printf "\n\n"
echo "Generating pickle..."
python3 ./script/dataset_preparation.py ./CSVs/CICIDS2017_corrected.csv 76 76 ./output
echo "Pickle generated!"
printf "\n\n"

rm -f ./CSVs/CICIDS2017_corrected.csv

#echo "Executing RandomForest Algorithm..."
#python3 ./script/machine_learning_architectures.py 1 ./output/CICIDS2017_corrected_76.pickle 1 4 ./output/

