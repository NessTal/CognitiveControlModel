import pandas as pd
import model as m

file = 'data.csv'

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


df = pd.read_csv(file)
participant_trials_dict = df.groupby('Participant')['Trial'].apply(tuple).to_dict()


output = pd.DataFrame(columns=['Participant','Trial','Simulated RT'])
for p in participant_trials_dict.keys():
    trials = [globals()[trial] for trial in participant_trials_dict[p]]
    results = m.RunTrialSequence(trials,params)
    simulated_RTs = [t[0] for t in results]
    df_p = pd.DataFrame({'Participant':p,'Trial':trials,'Simulated RT':simulated_RTs})
    output = pd.concat([output, df_p], axis=0)

output.to_csv('output.csv')