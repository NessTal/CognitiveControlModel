import pandas as pd
import numpy as np
import itertools
import multiprocessing as mp
import pickle
import glob
import random
import math
import plotly.express as px
from scipy import stats
#import time

import model as m

file = 'data.csv'
df = pd.read_csv(file)

# parameters
RepresetationsDecayRate = 0.01           # The number of activation units lost every iteration (from each Stroop/LangKnow node)
CognitiveControlDecayRate = 0.005        # The number of activation units lost every iteration (from each CC node)
ActivationRate = 0.1                     # The relation between unit activation and the activation it exerts through excitatory connections
MonitorBiasActivationRate = 1            # The relation between unit activation and the activation it exerts through excitatory connections
InhibitionRate = 0.1                     # The relation between unit activation and the inhibition it exerts through inhibitory connections
BiasingMult = 0.00002                    # The relation between Biasing unit activation level and how strongly it activates Stroop/Lang high-level nodes (in proportion to their activation levels)
MaxIter = 2000                           # Maximum number of iterations (per trial)
ActivationThreshold = 1000               # Threshold activation level for a lower-level unit to be cosidered the final interpratation
BetweenTrialsInterval = 1.5              # Time (in seconds) between consecutive trials
CongruentStroop = (0,0,0,10,0,15)        # Activation of font_color that supports Blue & text_blue that supports Blue
IncongruentStroop = (0,0,0,0,10,15)      # Activation of font_color that supports Blue & text_red that supports red
CongruentSentence = (15,0,10,0,0,0)      # Activation of wk that supports SubjIsTheme & MSK_ed that supports SubjIsTheme
AnomalousSentence = (15,10,0,0,0,0)      # Activation of wk that supports SubjIsTheme & MSK_ing that supports SubjIsAgent

params = (RepresetationsDecayRate, CognitiveControlDecayRate, ActivationRate, MonitorBiasActivationRate, InhibitionRate, BiasingMult, MaxIter, ActivationThreshold, BetweenTrialsInterval, CongruentStroop, IncongruentStroop, CongruentSentence, AnomalousSentence)

possible_values = {
    'RepresetationsDecayRate': [0.001,0.005,0.01,0.05,0.1],
    'CognitiveControlDecayRate': [0.0001,0.0005,0.001,0.0025,0.005,0.0075,0.01],
    'ActivationRate': [0.005,0.01,0.05,0.1,0.15,0.2],
    'MonitorBiasActivationRate': [0.1,0.5,1],
    'InhibitionRate': [0.005,0.01,0.05,0.1],
    'BiasingMult': [0.000001,0.000005,0.00001,0.000015,0.00002,0.00005,0.0001,0.0005],
    'ActivationThreshold': [100,1000]
}


def run_sim_for_human_data(df,params,save_csv=False):
    participant_trials_dict = df.groupby('Participant')['Trial'].apply(tuple).to_dict()
    practice_trials = [CongruentStroop, IncongruentStroop, CongruentSentence, AnomalousSentence, CongruentStroop, CongruentSentence]

    output = pd.DataFrame(columns=['Participant','Trial','Simulated RT'])
    for p in participant_trials_dict.keys():
        trials = practice_trials+[globals()[trial] for trial in participant_trials_dict[p]]
        results = m.RunTrialSequence(trials,params)
        simulated_RTs = [t[0] for t in results]
        simulated_RTs = simulated_RTs[6:]
        df_p = pd.DataFrame({'Participant':p,'Trial':trials[6:],'Simulated RT':simulated_RTs})
        output = pd.concat([output, df_p], axis=0)

    if save_csv == True:
        output.to_csv('output.csv')
    return output


def find_best_params_helper(args):
    df, params, col_name = args
    try:
        simulated_rt = list(run_sim_for_human_data(df, params)['Simulated RT'])
        return col_name, simulated_rt
    except:
        return col_name, None


def find_best_params(df,possible_values):
    combinations = list(itertools.product(*possible_values.values()))
    num_combs = len(combinations)

    pool = mp.Pool(processes=22)

    #start = time.time()
    failed = []
    results = []
    
    # Prepare a list of tuples to be passed to the pool
    for i, current_params in enumerate(combinations):
        RepresetationsDecayRate, CognitiveControlDecayRate, ActivationRate, MonitorBiasActivationRate, InhibitionRate, BiasingMult, ActivationThreshold = current_params
        params = (RepresetationsDecayRate, CognitiveControlDecayRate, ActivationRate, MonitorBiasActivationRate, InhibitionRate, BiasingMult, MaxIter, ActivationThreshold, BetweenTrialsInterval, CongruentStroop, IncongruentStroop, CongruentSentence, AnomalousSentence)
        col_name = "_".join([f"{param}={val}" for param, val in zip(possible_values.keys(), current_params)])
        results.append((df, params, col_name))

    i = 1
    # Parallel processing
    for col_name, simulated_rt in pool.imap_unordered(find_best_params_helper, results,chunksize=50): ########@
        if simulated_rt is None:
            failed.append(col_name)
            df[col_name] = np.nan
            print('Failed running', col_name)
            i += 1
        else:
            df[col_name] = simulated_rt
            print(f'Finished running {i} of {num_combs}')
            i += 1
        if i % 2500 == 0:
            pd.to_pickle(df,f'find_best_params_df_{i}.pkl')
            with open(f'failed_{i}.pkl', 'wb') as file:
                pickle.dump(failed, file)
            df = df[['Participant', 'Trial', 'Avg']]
            failed = []
    
    #end = time.time()
    #print(end - start)
    pd.to_pickle(df,f'find_best_params_df__{i}.pkl')

    with open(f'failed_{i}.pkl', 'wb') as file:
        pickle.dump(failed, file)
    
    #df = df[['Participant', 'Trial', 'Avg']]

    pool.close()
    pool.join()
    
    return df


file_names = 'find_best_params_df_*.pkl'

def combine_dfs_from_pickles(file_names):
    # Find all .pkl files starting with 'find_best_params_df' in the current directory
    file_paths = glob.glob(file_names)

    # Initialize an empty list to store DataFrames
    dfs = []
    i = 1

    # Iterate over each file path
    for file_path in file_paths:
        # Read the .pkl file into a DataFrame
        df = pd.read_pickle(file_path)

        # from all datasets exept for the first one, remove the columns 'Participant', 'Trial', 'Avg'
        if i > 1:
            df = df.drop(['Participant', 'Trial', 'Avg'], axis=1)
        i += 1
        dfs.append(df)

    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(dfs,axis=1)

    # Alert the user if an column name exists in more than one DataFrame
    unique_columns = combined_df.columns.unique()
    duplicate_columns = set()
    
    for column in unique_columns:
        column_counts = sum(column in df.columns for df in dfs)
        if column_counts != 1:
            duplicate_columns.add(column)

    if duplicate_columns:
        print("Warning: Column name conflicts detected in the following columns:")
        for column in duplicate_columns:
            print(column)

    # Write the combined DataFrame to a .pkl file
    combined_df.to_pickle('find_best_params_df.pkl')
    return combined_df


df_file_name = 'find_best_params_df.pkl'
test_proportion = 0.25
def split_participants(df_file_name,test_proportion, seed=123):
    df_all = pd.read_pickle(df_file_name)
    participants = df_all['Participant'].unique()
    random.seed(seed)
    test_participants = random.sample(list(participants), k = math.ceil(test_proportion*len(participants)))
    # split df_all such that df_test contains all rows with Participant in test_participants, and df_train contains the rest
    df_test = df_all[df_all['Participant'].isin(test_participants)]
    df_train = df_all[~df_all['Participant'].isin(test_participants)]
    return df_train, df_test

#df_train, df_test = split_participants(df_file_name,test_proportion)

def correlate_with_avg(df,correlations_file_name):
    df_critical_trials = df.dropna(subset=['Avg'])
    df_4corr = df_critical_trials.iloc[:, 3:]
    correlations = df_4corr.corrwith(df_critical_trials['Avg']).sort_values(ascending=False)
    correlations.to_pickle(correlations_file_name)
    return correlations

#correlations_file_name = 'correlations_train.pkl'
#correlations_train = correlate_with_avg(df_train,correlations_file_name)
#winning_params = correlations_train.index[0]

#correlations_file_name = 'correlations_test.pkl'
#correlation_test = correlate_with_avg(df_test,correlations_file_name)[winning_params]

def plot_correlation(df,winning_params):
    df_critical_trials = df.dropna(subset=['Avg'])
    plt = px.scatter(df_critical_trials, x=winning_params, y="Avg",
                     trendline="ols",trendline_color_override="black")
    plt.update_layout(xaxis_title='Simulated (iterations)', yaxis_title='P600 amplitude (uV)')
    plt.show()
    return plt


def dict_winning_params(correlations_file_name):
    correlations = pd.read_pickle(correlations_file_name)
    top_10 = []
    for col_name in correlations.index[:10]:
        params = {}
        pairs = col_name.split("_")
        for pair in pairs:
            param, val = pair.split("=")
            params[param] = val
        top_10.append(params)
    return top_10



if __name__ == "__main__":
    find_best_params(df,possible_values)


#with open('failed.pkl', 'rb') as file:
#    failed = pickle.load(file)

#correlations = pd.read_pickle('correlations.pkl')
#top_10 = []
#for col_name in correlations.index[:10]:
#    params = {}
#    pairs = col_name.split("_")
#    for pair in pairs:
#        param, val = pair.split("=")
#        params[param] = val
#    top_10.append(params)


# df_all = pd.read_pickle('find_best_params_df.pkl')
# def find_missing_params(df,possible_values):
#     combinations = list(itertools.product(*possible_values.values()))
#     col_names_all_combs = []
#     for i, current_params in enumerate(combinations):
#         col_name = "_".join([f"{param}={val}" for param, val in zip(possible_values.keys(), current_params)])
#         col_names_all_combs.append(col_name)

#     col_names_in_df = list(df.columns)
#     missing_cols = list(set(col_names_all_combs) - set(col_names_in_df))
    
#     missing_params = []
#     for col_name in missing_cols:
#         params = []
#         pairs = col_name.split("_")
#         for pair in pairs:
#             param, val = pair.split("=")
#             params.append(float(val))
#         missing_params.append(tuple(params))

#     #######################################@
#     for file in ['find_best_params_df_1000.pkl','find_best_params_df_500_c.pkl','find_best_params_df_1000_c.pkl']:
#         df_part = pd.read_pickle(file)
#         df_part_params = []
#         for col_name in df_part.columns:
#             if col_name in ['Participant', 'Trial', 'Avg']:
#                 continue
#             params = []
#             pairs = col_name.split("_")
#             for pair in pairs:
#                 param, val = pair.split("=")
#                 params.append(float(val))
#             df_part_params.append(tuple(params))
#         missing_params = list(set(missing_params) - set(df_part_params))
#     #######################################@

#     return missing_cols, missing_params

# missing_cols, missing_params = find_missing_params(df_all,possible_values)
# with open('missing_params.pkl', 'wb') as file:
#     pickle.dump(missing_params, file)

# with open('missing_params.pkl', 'rb') as file:
#     missing_params = pickle.load(file)


with open('find_best_params_df.pkl', 'rb') as file:
    df_all = pickle.load(file)

with open('correlations_test.pkl', 'rb') as file:
    correlations_test = pickle.load(file)

with open('correlations_train.pkl', 'rb') as file:
    correlations_train = pickle.load(file)

df_train, df_test = split_participants(df_file_name,test_proportion)

winning_params = correlations_train.index[0]
correlation_test = correlations_test[winning_params]
df_critical_trials = df_all.dropna(subset=['Avg'])
df_critical_trials_test = df_test.dropna(subset=['Avg'])
fig = plot_correlation(df_critical_trials,winning_params)
stat_result_test = stats.pearsonr(df_critical_trials_test[winning_params],df_critical_trials_test['Avg'])
print(f"Pearson correlation on test trials: {stat_result_test[0]:.3f}, p-value: {stat_result_test[1]:.3f}")
stat_result_all = stats.pearsonr(df_critical_trials[winning_params],df_critical_trials['Avg'])
print(f"Pearson correlation on all trials: {stat_result[0]:.3f}, p-value: {stat_result[1]:.3f}")
#print(f"Correlation between {winning_params} and Avg: {stat_result[0]:.3f}, p-value: {stat_result[1]:.3f}")