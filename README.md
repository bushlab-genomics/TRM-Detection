# TRM-Detection
The code and files in this repo correspond to the paper titled **Detecting global influence of transcription factor interactions on gene expression in lymphoblastoid cells using neural network models**. Here, we have provided test files and code for creating MLPU models described in the paper using multi-omics TF-based regulatory features derived from PANDA GRNs. We have also provided scripts for computing interaction effects for TF regulatory modules(TRMs) and main effects for individual TFs from the learned MLPU models.

## Requirements(software)
python3

## python packages
numpy(1.17.2)\
pandas(0.25.1)\
scipy(1.4.1)\
scikit-learn(0.21.2)\
Keras(2.3.1)\
tensorflow(1.14.0)

### Tutorial
There are four scripts in the **Code** folder. Below we have provided the detailed instructions on how to run them. These scripts should be run in the order mentioned below.

1) **build_mlpu_models.py**: This script uses a GRN based input feature matrix to build MLPU models to predict TG expression. The output from this script consists of the trained MLPU models and the evaluation metrics file containing PCC (Pearsons Correlation Coefficient). The models(format: output prefix + '\_\' + *random iteration* + '\_\' + *fold number* + "_mlpu_model.h5" and the output file(format: output prefix + "_pred_eval.csv") will be produced in the "pred_results" directory by default, which will be created while the script is being run if it is not already present in the repo directory. 
* -i, --input: The input GRN based feature matrix required to build the MLPU models. We have provided an example "toy_grn_reg_net.tsv" in the **Data** folder. In order to generate a custom feature matrix, please use our code from the TF_GRN repo(https://github.com/bushlab-genomics/TF_GRN).
* -o, --output: The output prefix to be used for the ouput file and the models. 
* -x, --expression: The TG expression file to be used as output for training the MLPU models. The example expression file "toy_pred_expression.csv" is provided in the **Data** folder.
* -r, --random: Number of random iterations/states to be used for training the MLPU models(default = 5). For each iteration, the MLPU models are further trained using 5 fold inner cross-validation and corresponding MLPU models are saved for each fold within each iteration. 
* -p, --dropout: Dropout probabilities to be used for training the MLPU models(default = [0.2,0.3,0.5]).
* -u, --denseu: Number of dense units to be used for each of the three layers of the univariate MLPs of the MLPU models(default = [10,10,10]).
* -m, --densem: Number of dense units to be used for each of the three layers of the traiditional MLP of the MLPU models(default = [100,200,300]).
* -e, --epochs: Number of epochs to be used for training the MLPU models(default = 50). 
* -b, --batchsize: Batchsize of the TG set to be used for training the MLPU models(default = 32). 

**example run: python Code/build_mlpu_models.py -i toy_grn_reg_net.tsv -x toy_pred_expression.csv -r -b -o toy_grn -e 10**

2) **get_nid_scores.py**: This script uses the trained MLPU models to detect TRMs and calculate their interaction effects using the NID algorithm. Significant portion of this script is derived from the original NID repo(https://github.com/mtsang/neural-interaction-detection). The output files(format: output prefix + "_" + random iteration + "_model_nid_scores.csv") will be produced in the "pred_results" directory by default. 
* -o, --output: The output prefix to be used for the files containing NID scores calculated using MLPU models for each random iteration. This prefix must match the output prefix used for running the **build_mlpu_models.py** exactly. 
* -m, --multiprocess: Select this flag in order to run the script as a parallel process. 
* -n, --cpu: Use this option along with the -m flag to specify the number of cores/cpus to be used for parallel processing.

**example run: python Code/get_nid_scores.py -o toy_grn**

3) **postprocess_nid_scores.py**: This script postprocesses the file containing NID scores calculated using the **get_nid_scores.py**. Specifically, it merges the NID scores from all the random iterations and computes average NID score for each TRM followed by Log2 normalization of the NID scores. It also uses a threshold, derived from the median Log2NID scores of the TRM set discovered in the paper to filter out the insignificant TRMs. The final file containing Log2NID scores(format: output prefix + "_nid_scores_postprocessed.csv") will be produced in the "pred_results" directory by default. 
* -o, --output: The output prefix used in the previous scripts, which will be also present in the final postprocessed file. 
* -t, --threshold: The threshold to be used for Log2 NID scores to filter out TRMs not significantly impacting TG expression(default= 2.25 derived from the median Log2NID scores of the TRMs discovered in the paper). 

**example run: python Code/postprocess_nid_scores.py -o toy_grn**

4) **get_main_effects.py**: This script computes main effects of individual TFs present in the input feature matrix from the univariate MLPs of the MLPU models. The output file(format: output prefix + "_univariate_main_effects.csv")
* -o, --output: The output prefix used in the previous scripts, which will be also present in the final file containing main effects.
* -m, --multiprocess: Select this flag in order to run the script as a parallel process. 
* -n, --cpu: Use this option along with the -m flag to specify the number of cores/cpus to be used for parallel processing.

**example run: python Code/get_main_effects.py -o toy_grn**



