import pandas as pd
import numpy as np
import glob
import multiprocessing as mp
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-o","--output", help="provide the output prefix", action = "store")
parser.add_argument("-t","--threshold", help="provide the output threshold for log2nid scores to be used to filter out non-significant interactions(default = 2.25)",nargs = "?", action = "store", type = float, const = 2.25)

args = parser.parse_args()

results_path = "pred_results/"
score_file_list = glob.glob(results_path + args.output + "*nid_scores.csv")
df_list = pd.concat([pd.read_csv(i) for i in score_file_list])
df_list = df_list.drop("Random_State", axis = 1)
df_list_grouped = df_list.groupby("Interactions").mean()
df_list_grouped.insert(0, "Interactions", df_list_grouped.index)
df_list_grouped["Log2_NID_Score"] = np.log2(df_list_grouped.Score)
df_list_grouped.columns = ["Interactions", "Mean_NID_Score", "Log2NID_Score"]
df_list_grouped = df_list_grouped.loc[df_list_grouped.Log2NID_Score >= args.threshold]
df_list_grouped.to_csv(results_path + args.output + "_nid_scores_postprocessed.csv", index = False)

