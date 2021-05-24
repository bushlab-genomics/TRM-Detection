import os
#os.environ["KERAS_BACKEND"]="theano"
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import itertools as itr
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint
from functools import partial
from keras.layers import Layer, Input, Dense, add,Dropout
from keras.models import load_model,Model, Sequential
import argparse 


parser = argparse.ArgumentParser()

parser.add_argument("-i","--input", help="please provide a file name containing PANDA GRN features", action = "store") 
parser.add_argument("-o","--output", help="please provide a prefix for the output files and models", action = "store") 
parser.add_argument("-x","--expression", help="please provide the file name containing expression fpkm values", action = "store") 
parser.add_argument("-r","--random", help="please provide the number of random instances for which the MLPU models will be run",type = int, nargs = "?", const = 5, action = "store")
parser.add_argument("-p","--dropout", help = "please provide the dropout probabiliies for the MLP model layers(default = [0.2,0.3,0.5])", type = float, nargs = "+",default = [0.2,0.3,0.5], action = "store") 
parser.add_argument("-u","--denseu", help = "please provide the number of dense units for each of the three model layers for the univariate models(default = [10,10,10])", type = int, nargs = "+",default = [10,10,10], action = "store")
parser.add_argument("-m","--densem", help = "please provide the number of dense units for each of the three MLP model layers for the MLPU models(default = [100,200,300])", type = int, nargs = "+",default = [100,200,300], action = "store")
parser.add_argument("-e","--epochs", help="please provide the number of epochs for fitting the MLPU models(default = 50)", action = "store", type = int, nargs = "?", const = 50)
parser.add_argument("-b","--batchsize", help="please provide the batchsize for fitting the MLPU models(default = 32)",type = int, nargs= "?",const = 32, action = "store")
args = parser.parse_args()

input_file_path  = "Data/"
out_path =  "pred_results/"
if not os.path.exists(out_path):
        os.mkdir(out_path)

print("Reading the input files.")
expr_df  =pd.read_csv(input_file_path + args.expression,header = None)
panda_scores_df =pd.read_csv(args.input, sep = "\t")
panda_scores_t = panda_scores_df.T
panda_scores_t.insert(0,"hgnc_symbol", list(panda_scores_t.index))
expr_df.columns = ["hgnc_symbol", "lg_fpkm"]
mrg_df = expr_df.merge(panda_scores_t, on = "hgnc_symbol", how = "inner")
#np_scores = np.array(mrg_df.iloc[:,2:])
#np_scores_scaled  = MinMaxScaler().fit_transform(np_scores)
#np_expr = np.array(mrg_df.iloc[:,1])
#np_expr_scaled = StandardScaler().fit_transform(np_expr.reshape((-1,1)))
ind_list = list(mrg_df.index)
tf_names = list(mrg_df)[2:]
epochs = args.epochs
mlp_dense_units = args.densem
mlpu_dense_units = args.denseu
drop_probs = args.dropout
batch_size = args.batchsize 
tf_df = pd.DataFrame(tf_names)
tf_df.to_csv(out_path + "tf_list.txt",index = False, header = None)

print("Building the MLPU models using " + str(mrg_df.shape[0]) + " genes and " + str(len(tf_names)) + " TFs for " + str(epochs) +" epochs.")


def get_lists(x):
        ind_train,ind_test = train_test_split(ind_list,test_size = 0.2, random_state = x)
        rr = len(ind_test)*4 - len(ind_train)
        ind_i =  random.sample(range(len(ind_test)),rr)
        ind_tr = ind_train + list(np.array(ind_train)[ind_i])
        ind_rest = list(zip(*[iter(ind_tr)]*len(ind_test)))
        test_ind_list = [ind_test, ind_rest[0], ind_rest[1] , ind_rest[2], ind_rest[3]]
        train_ind_list = [list(set.difference(set(ind_list),set(i))) for i in test_ind_list]
        train_scores = []
        train_expr =[]
        test_scores = []
        test_expr = []
        gene_test = []
        tf_names= list(mrg_df.columns)[2:]
        for i in range(len(test_ind_list)):
                np_test_scores = np.array(mrg_df.ix[test_ind_list[i],2:])
                np_train_scores = np.array(mrg_df.ix[train_ind_list[i],2:])
                np_test_scaled  = MinMaxScaler().fit_transform(np_test_scores)
                np_train_scaled  = MinMaxScaler().fit_transform(np_train_scores)
                np_test_expr = np.array(mrg_df.ix[test_ind_list[i],1])
                np_train_expr = np.array(mrg_df.ix[train_ind_list[i],1])
                np_test_expr_scaled = StandardScaler().fit_transform(np_test_expr.reshape((-1,1)))
                np_train_expr_scaled = StandardScaler().fit_transform(np_train_expr.reshape((-1,1)))
                train_scores.append(np_train_scaled)
                test_scores.append(np_test_scaled)
                train_expr.append(np_train_expr_scaled)
                test_expr.append(np_test_expr_scaled)
                gene_test.append(list(mrg_df.ix[test_ind_list[i],0]))
        return(train_scores,test_scores,train_expr,test_expr,gene_test,tf_names)

mlp_model = Sequential()
mlp_model.add(Dense(mlp_dense_units[0], input_dim = len(tf_names),activation = "relu"))
mlp_model.add(Dropout(drop_probs[0]))
mlp_model.add(Dense(mlp_dense_units[1], activation = "relu"))
mlp_model.add(Dropout(drop_probs[1]))
mlp_model.add(Dense(mlp_dense_units[2], activation = "relu"))
mlp_model.add(Dropout(drop_probs[2]))
mlp_model.add(Dense(1))
mlpu_model_list =[]
for i in range(len(tf_names)):
    inp  = Input((1,))
    d1 = Dense(mlpu_dense_units[0], activation = "relu")(inp)
    d2 = Dense(mlpu_dense_units[1], activation = "relu")(d1)
    d3 = Dense(mlpu_dense_units[2], activation = "relu")(d2)
    out = Dense(1)(d3)
    mdl  = Model(inp,out)
    new_layer_names = ["".join(list(itr.chain.from_iterable([tf_names[i],layer.name.split("_")]))) for layer in mdl.layers]
    for i in range(len(mdl.layers)):
        mdl.layers[i].name = new_layer_names[i]
    mlpu_model_list.append(mdl)
all_out = [mdl.output for mdl in mlpu_model_list]
all_out.append(mlp_model.output)
sum_out  = add(all_out)
in_layers = [mdl.input for mdl in mlpu_model_list]
in_layers.append(mlp_model.input)
mlpu_model = Model(in_layers, output = sum_out)
mlpu_model.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ["mae"])

def pred_mlpu_models(n):
	train_score_list,test_score_list,train_expr_list,test_expr_list,gene_list,tf_names =  get_lists(n)
	out_mdl_list = [out_path + args.output  + "_" + str(n) + "_" + str(i) + "_mlpu_model.h5" for i in range(len(train_score_list))]
	r2_list = []
	for i in range(len(train_score_list)):
		out_mdl = out_mdl_list[i]
		mc = ModelCheckpoint(out_mdl, monitor='val_loss', verbose=1, save_best_only=True)
		es = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
		np_array_list = [train_score_list[i][:,k] for k in range(len(tf_names))]
		np_array_list.append(train_score_list[i])
		np_array_list_test = [test_score_list[i][:,k] for k in range(len(tf_names))]
		np_array_list_test.append(test_score_list[i])
		mlpu_model.fit(np_array_list,train_expr_list[i], batch_size = batch_size, epochs = epochs, callbacks = [es,mc], validation_split = 0.1)
		pred_mlp  = mlpu_model.predict(np_array_list_test)
		corr_mlp  =np.corrcoef(test_expr_list[i].reshape(-1,),pred_mlp.reshape(-1,))[1,0]
		r2_list.append(corr_mlp)
	mean_r2 = np.mean(r2_list)
	res_list = [n, args.output, mean_r2]		
	return(res_list)


random_states_f = random.sample(range(1000,10000000), args.random)
pred_df = pd.DataFrame(list(map(pred_mlpu_models, random_states_f)))
pred_df.columns = ["Random_State","Output_prefix", "Mean_PCC"]
pred_df.to_csv(out_path + args.output + "_pred_eval.csv", index = False)









