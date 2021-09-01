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
tf_list_enum =  list(enumerate(tf_df[0]))

def get_weights(x):
    lyr_weight_list = []
    fl_names = [models_path+ args.output + "_"+str(x)+"_"+str(i)+"_mlpu_model.h5" for i in range(5)]
    for i in fl_names:
        nm = i
        model_opn = load_model(nm,compile = False)
        mlp_layers = [layer for layer in model_opn.layers if layer.name.find("_") != -1 and layer.name.find("dense") != -1]
        mlp_layers = [layer for layer in mlp_layers if layer.name.find("input")  == -1]
        uni_mlp_weights = [np.transpose(layer.get_weights()[0]) for layer in mlp_layers] 
        lyr_weight_list.append(uni_mlp_weights)
    lyr1_mat_lst = [i[0] for i in lyr_weight_list]
    lyr2_mat_lst = [i[1] for i in lyr_weight_list]
    lyr3_mat_lst = [i[2] for i in lyr_weight_list]
    lyr_out_lst = [i[3] for i in lyr_weight_list]
    lyr_1_f = np.zeros(lyr1_mat_lst[0].shape)
    lyr_2_f = np.zeros(lyr2_mat_lst[0].shape)
    lyr_3_f = np.zeros(lyr3_mat_lst[0].shape)
    lyr_out_f = np.zeros(lyr_out_lst[0].shape)
    for k in range(len(lyr1_mat_lst)):
        lyr_1_f = np.add(lyr1_mat_lst[k],lyr_1_f)
        lyr_2_f = np.add(lyr2_mat_lst[k],lyr_2_f)
        lyr_3_f = np.add(lyr3_mat_lst[k],lyr_3_f)
        lyr_out_f = np.add(lyr_out_lst[k],lyr_out_f)
    return([lyr_1_f/len(lyr_weight_list),lyr_2_f/len(lyr_weight_list), lyr_3_f/len(lyr_weight_list),lyr_out_f/len(lyr_weight_list)])


def preprocess_weights(weights):
    w_later = np.abs(weights[-1])
    w_input = np.abs(weights[0])

    for i in range(len(weights) - 2, 0, -1):
        w_later = np.matmul(w_later, np.abs(weights[i]))

    return w_input, w_later


def make_one_indexed(interaction_ranking):
    return [(tuple(np.array(i) + 1), s) for i, s in interaction_ranking]

def interpret_interactions(w_input, w_later, get_main_effects=False):
    interaction_strengths = {}
    for i in range(w_later.shape[1]):
        sorted_hweights = sorted(
            enumerate(w_input[i]), key=lambda x: x[1], reverse=True
        )
        interaction_candidate = []
        candidate_weights = []
        for j in range(w_input.shape[1]):
            bisect.insort(interaction_candidate, sorted_hweights[j][0])
            candidate_weights.append(sorted_hweights[j][1])

            if not get_main_effects and len(interaction_candidate) == 1:
                continue
            interaction_tup = tuple(interaction_candidate)
            if interaction_tup not in interaction_strengths:
                interaction_strengths[interaction_tup] = 0
            interaction_strength = (min(candidate_weights)) * (np.sum(w_later[:, i]))
            interaction_strengths[interaction_tup] += interaction_strength

    interaction_ranking = sorted(
        interaction_strengths.items(), key=operator.itemgetter(1), reverse=True
    )

    return interaction_ranking

def interpret_pairwise_interactions(w_input, w_later):
    p = w_input.shape[1]

    interaction_ranking = []
    for i in range(p):
        for j in range(p):
            if i < j:
                strength = (np.minimum(w_input[:, i], w_input[:, j]) * w_later).sum()
                interaction_ranking.append(((i, j), strength))

    interaction_ranking.sort(key=lambda x: x[1], reverse=True)
    return interaction_ranking

def prune_redundant_interactions(interaction_ranking, max_interactions=100000):
    interaction_ranking_pruned = []
    current_superset_inters = []
    for inter, strength in interaction_ranking:
        set_inter = set(inter)
        if len(interaction_ranking_pruned) >= max_interactions:
            break
        subset_inter_skip = False
        update_superset_inters = []
        for superset_inter in current_superset_inters:
            if set_inter < superset_inter:
                subset_inter_skip = True
                break
            elif not (set_inter > superset_inter):
                update_superset_inters.append(superset_inter)
        if subset_inter_skip:
            continue
        current_superset_inters = update_superset_inters
        current_superset_inters.append(set_inter)
        interaction_ranking_pruned.append((inter, strength))

    return interaction_ranking_pruned

def get_interactions(weights, pairwise=False, one_indexed=False):
    w_input, w_later = preprocess_weights(weights)

    if pairwise:
        interaction_ranking = interpret_pairwise_interactions(w_input, w_later)
    else:
        interaction_ranking = interpret_interactions(w_input, w_later)
        interaction_ranking = prune_redundant_interactions(interaction_ranking)

    if one_indexed:
        return make_one_indexed(interaction_ranking)
    else:
        return interaction_ranking

def get_h_tf_rankings_model(x):
    mdl_weights = get_weights(x)
    h_gn_rankings = get_interactions(mdl_weights)
    act_ind = []
    scr_list = []
    for i in range(len(h_gn_rankings)):
        a_n1 = []
        for k in range(len(tf_list_enum)):
            for n in range(len(h_gn_rankings[i][0])):
                if h_gn_rankings[i][0][n] == tf_list_enum[k][0]:
                    a_n1.append(tf_list_enum[k][1])
        act_ind.append(a_n1)
    for i in range(len(h_gn_rankings)):
        kn1 = "_".join(act_ind[i])
        kn_scr = h_gn_rankings[i][1]
        scr_list.append((kn1,kn_scr))
    scr_df = pd.DataFrame(scr_list)
    scr_df.columns = ["Interactions", "Score"]
    #return(scr_df)
    scr_df.insert(0,"Random_State", [x]*len(scr_df))
    nm =  models_path + args.output + "_" + str(x) + "_model_nid_scores.csv"
    scr_df.to_csv(nm, index = False)
    print(nm + " is done.")

if args.multiprocess:
    pool = mp.Pool(args.cpu)
    random_file_list = glob.glob(models_path + "*h5")
    random_states_f = []
    for i in random_file_list:
        k1= i.split("/")
        k2 = int(k1[len(k1) -1].split("_")[-4])
        random_states_f.append(k2)
    random_states_f = sorted(set(random_states_f))
    tl = list(pool.map(get_h_tf_rankings_model,random_states_f))
else:
    pool = mp.Pool(args.cpu)
    random_file_list = glob.glob(models_path + "*h5")
    random_states_f = []
    for i in random_file_list:
        k1= i.split("/")
        k2 = int(k1[len(k1) -1].split("_")[-4])
        random_states_f.append(k2)
    random_states_f = sorted(set(random_states_f))
    tl = list(map(get_h_tf_rankings_model,random_states_f))
