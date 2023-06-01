import pandas as pd
import numpy as np
import itertools
import multiprocessing as mp
import pickle
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
MaxIter = 5000                           # Maximum number of iterations (per trial)
ActivationThreshold = 1000               # Threshold activation level for a lower-level unit to be cosidered the final interpratation
BetweenTrialsInterval = 1.5              # Time (in seconds) between consecutive trials
CongruentStroop = (0,0,0,10,0,15)        # Activation of font_color that supports Blue & text_blue that supports Blue
IncongruentStroop = (0,0,0,0,10,15)      # Activation of font_color that supports Blue & text_red that supports red
CongruentSentence = (15,0,10,0,0,0)      # Activation of wk that supports SubjIsTheme & MSK_ed that supports SubjIsTheme
AnomalousSentence = (15,10,0,0,0,0)      # Activation of wk that supports SubjIsTheme & MSK_ing that supports SubjIsAgent

params = (RepresetationsDecayRate, CognitiveControlDecayRate, ActivationRate, MonitorBiasActivationRate, InhibitionRate, BiasingMult, MaxIter, ActivationThreshold, BetweenTrialsInterval, CongruentStroop, IncongruentStroop, CongruentSentence, AnomalousSentence)


def run_sim_for_human_data(df,params):
    participant_trials_dict = df.groupby('Participant')['Trial'].apply(tuple).to_dict()

    output = pd.DataFrame(columns=['Participant','Trial','Simulated RT'])
    for p in participant_trials_dict.keys():
        trials = [globals()[trial] for trial in participant_trials_dict[p]]
        results = m.RunTrialSequence(trials,params)
        simulated_RTs = [t[0] for t in results]
        df_p = pd.DataFrame({'Participant':p,'Trial':trials,'Simulated RT':simulated_RTs})
        output = pd.concat([output, df_p], axis=0)

    output.to_csv('output.csv')
    return output

possible_values = {
    'RepresetationsDecayRate': [0.001,0.005,0.01,0.05,0.1],
    'CognitiveControlDecayRate': [0.0001,0.0005,0.001,0.0025,0.005,0.0075,0.01],
    'ActivationRate': [0.005,0.01,0.05,0.1,0.15,0.2],
    'MonitorBiasActivationRate': [0.1,0.5,1],
    'InhibitionRate': [0.005,0.01,0.05,0.1],
    'BiasingMult': [0.000001,0.000005,0.00001,0.000015,0.00002,0.00005,0.0001,0.0005],
    'ActivationThreshold': [100,1000]
}


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

    combinations_divided = []
    for i in range((num_combs // 5000) + 1):
        start_index = i * 5000
        end_index = (i + 1) * 5000
        combinations_divided.append(combinations[start_index:end_index])

    failed = []

    pool = mp.Pool(processes=4)
    results = []

    batch_num = 1
 
    for combs in combinations_divided:
        #start = time.time()
        pool = mp.Pool(processes=4)
        results = []
        
        for i, current_params in enumerate(combs):
            RepresetationsDecayRate, CognitiveControlDecayRate, ActivationRate, MonitorBiasActivationRate, InhibitionRate, BiasingMult, ActivationThreshold = current_params
            params = (RepresetationsDecayRate, CognitiveControlDecayRate, ActivationRate, MonitorBiasActivationRate, InhibitionRate, BiasingMult, MaxIter, ActivationThreshold, BetweenTrialsInterval, CongruentStroop, IncongruentStroop, CongruentSentence, AnomalousSentence)

            col_name = "_".join([f"{param}={val}" for param, val in zip(possible_values.keys(), current_params)])
            results.append((df, params, col_name))

        # Parallel processing
        for col_name, simulated_rt in pool.map(find_best_params_helper, results):
            if simulated_rt is None:
                failed.append(col_name)
                df[col_name] = np.nan
                print('Failed running', col_name)
            else:
                df[col_name] = simulated_rt
                print('Finished running batch %2d' % (batch_num))

        pool.close()
        pool.join()

        batch_num += 1
        
        #end = time.time()
        #print(end - start)
        df = df.copy()
        pd.to_pickle(df,'find_best_params_df.pkl')

        with open('failed.pkl', 'wb') as file:
            pickle.dump(failed, file)
    
    return[df]


if __name__ == "__main__":
    df_all = find_best_params(df,possible_values)


#df_all = pd.read_pickle('find_best_params_df.pkl')
#with open('failed.pkl', 'rb') as file:
#    failed = pickle.load(file)