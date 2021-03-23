import scipy
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import itertools as itr
import collections
import bisect
import operator
from keras.models import load_model
import multiprocessing as mp
import glob as gl
from functools import partial
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("-o","--output", help="provide the output prefix", action = "store")
parser.add_argument("-m","--multiprocess", help = "select this flag along with all flag to do multiprocessing", action = "store_true")
parser.add_argument("-n", "--cpu",type = int, help = "number of cpu cores to use", action = "store")

args = parser.parse_args()

models_path = "pred_results/"
tf_df = pd.read_csv(models_path + "tf_list.txt", header = None)
tf_names = list(tf_df[0])

def preprocess_weights_tf(tf_layers):
    tf_weights = [np.transpose(layer.get_weights()[0]) for layer in tf_layers]
    w_later = np.abs(tf_weights[-1])
    w_input = tf_weights[0]
    for i in range(len(tf_weights) - 2, 0, -1):
        w_later = np.matmul(w_later, np.abs(tf_weights[i]))
    return w_input, w_later

def get_tf_layers(x,layers):
    tf_layers= []
    for layer in layers:
        layer_tf_name = layer.name.split("dense")[0]
        if layer_tf_name == x:
            tf_layers.append(layer)
    return(tf_layers)

def get_main_effects_model(model):
    mdl_load = model
    #mdl_load = load_model(model, compile = False)
    u_layers =[layer for layer in mdl_load.layers if layer.name.find("_") == -1]
    u_layers = [layer for layer in u_layers if layer.name.find("dense") !=-1]
    mean_scores = []
    for i in tf_names:
        tf_layers = get_tf_layers(i,u_layers)
        p_in,p_later = preprocess_weights_tf(tf_layers)
        tf_score = np.mean(p_in*np.sum(p_later))
        mean_scores.append((i,tf_score))
    mean_scores_df = pd.DataFrame(mean_scores,columns = ["TF_Name","Main_Effect"]) 
    return(mean_scores_df)

def get_main_effects(x):
    fl_names = [models_path+ args.output + "_"+str(x)+"_"+str(i)+"_mlpu_model.h5" for i in range(5)]
    model_tf_df_list = []
    for i in fl_names:
        nm = i
        model_opn = load_model(nm,compile = False)
        tf_effect_df = get_main_effects_model(model_opn)
        model_tf_df_list.append(tf_effect_df)
    model_tf_df = pd.concat(model_tf_df_list)
    model_tf_df1 = model_tf_df.groupby("TF_Name").mean()
    model_tf_df1.insert(0,"TF_Name", list(model_tf_df1.index))
    model_tf_df1 = model_tf_df1.reset_index(drop = True)
    return(model_tf_df1)

 
if args.multiprocess:
    pool = mp.Pool(args.cpu)
    random_file_list = glob.glob(models_path + "*h5")
    random_states_f = []
    for i in random_file_list:
        k1= i.split("/")
        k2 = int(k1[len(k1) -1].split("_")[2])
        random_states_f.append(k2)
    random_states_f = sorted(set(random_states_f))
    t1 = list(pool.map(get_main_effects,random_states_f))
else:            
    random_file_list = glob.glob(models_path + "*h5")
    random_states_f = []
    for i in random_file_list:
        k1= i.split("/")          
        k2 = int(k1[len(k1) -1].split("_")[2])
        random_states_f.append(k2)
    random_states_f = sorted(set(random_states_f))
    t1 = list(map(get_main_effects,random_states_f))

t1_df = pd.concat(t1)
t1_df1 = t1_df.groupby("TF_Name").mean()
t1_df1.insert(0, "TF_Name", list(t1_df1.index))
t1_df1.to_csv(models_path + args.output + "_univariate_main_effects.csv", index = False)
