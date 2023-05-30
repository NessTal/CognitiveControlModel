import plotly.express as px
import pandas as pd

from model import *

def CreateFig_WithWithoutCC_Stroop(params):
    RepresetationsDecayRate, CognitiveControlDecayRate, ActivationRate, MonitorBiasActivationRate, InhibitionRate, BiasingMult, MaxIter, ActivationThreshold, BetweenTrialsInterval, CongruentStroop, IncongruentStroop, CongruentSentence, AnomalousSentence = params
    params_no_CC = RepresetationsDecayRate, CognitiveControlDecayRate, ActivationRate, MonitorBiasActivationRate, InhibitionRate, 0, MaxIter, ActivationThreshold, BetweenTrialsInterval, CongruentStroop, IncongruentStroop, CongruentSentence, AnomalousSentence

    UserBiasingMult = BiasingMult
    # Congruent Stroop trial with BiasingMult = 0
    BiasingMult = 0
    i, Winner, CogCtrl, LingKnow, Stroop, Activations = RunTrial(CongruentStroop,params_no_CC)
    RT_CongStroop_NoCC = i
    print(i, Winner)

    # Congruent Stroop trial with BiasingMult = 1
    BiasingMult = UserBiasingMult
    i, Winner, CogCtrl, LingKnow, Stroop, Activations = RunTrial(CongruentStroop,params)
    RT_CongStroop_WithCC = i
    print(i, Winner)

    # Inongruent Stroop trial with BiasingMult = 0
    BiasingMult = 0
    i, Winner, CogCtrl, LingKnow, Stroop, Activations = RunTrial(IncongruentStroop,params_no_CC)
    RT_IncongStroop_NoCC = i
    print(i, Winner)

    # Incongruent Stroop trial with BiasingMult = 1
    BiasingMult = UserBiasingMult
    i, Winner, CogCtrl, LingKnow, Stroop, Activations = RunTrial(IncongruentStroop,params)
    RT_IncongStroop_WithCC = i
    print(i, Winner)

    # Figure
    df = pd.DataFrame({'CC':['Without CC','With CC','Without CC','With CC'],
                       'Trial':['Congruent','Congruent','Incongruent','Incongruent'],
                       'RT':[RT_CongStroop_NoCC,RT_CongStroop_WithCC,RT_IncongStroop_NoCC,RT_IncongStroop_WithCC]})

    fig = px.bar(df,x='Trial',y='RT',color='CC',barmode="group")
    return fig

def CreateFig_WithWithoutCC_Lang(params):
    RepresetationsDecayRate, CognitiveControlDecayRate, ActivationRate, MonitorBiasActivationRate, InhibitionRate, BiasingMult, MaxIter, ActivationThreshold, BetweenTrialsInterval, CongruentStroop, IncongruentStroop, CongruentSentence, AnomalousSentence = params
    params_no_CC = RepresetationsDecayRate, CognitiveControlDecayRate, ActivationRate, MonitorBiasActivationRate, InhibitionRate, 0, MaxIter, ActivationThreshold, BetweenTrialsInterval, CongruentStroop, IncongruentStroop, CongruentSentence, AnomalousSentence

    UserBiasingMult = BiasingMult
    # Control sentence trial with BiasingMult = 0
    BiasingMult = 0
    i, Winner, CogCtrl, LingKnow, Stroop, Activations = RunTrial(CongruentSentence,params_no_CC)
    RT_CongLang_NoCC = i
    print(i, Winner)

    # control sentence trial with BiasingMult = 1
    BiasingMult = UserBiasingMult
    i, Winner, CogCtrl, LingKnow, Stroop, Activations = RunTrial(CongruentSentence,params)
    RT_CongLang_WithCC = i
    ActivationsCong = Activations
    print(i, Winner)

    # Anomalous sentence trial with BiasingMult = 0
    BiasingMult = 0
    i, Winner, CogCtrl, LingKnow, Stroop, Activations = RunTrial(AnomalousSentence,params_no_CC)
    RT_AnomLang_NoCC = i
    print(i, Winner)

    # Anomalous sentence trial with BiasingMult = 1
    BiasingMult = UserBiasingMult
    i, Winner, CogCtrl, LingKnow, Stroop, Activations = RunTrial(AnomalousSentence,params)
    RT_AnomLang_WithCC = i
    ActivationsIncong = Activations
    print(i, Winner)

    # Figure
    df = pd.DataFrame({'CC':['Without CC','With CC','Without CC','With CC'],
                       'Trial':['Congruent','Congruent','Anomaly','Anomaly'],
                       'RT':[RT_CongLang_NoCC,RT_CongLang_WithCC,RT_AnomLang_NoCC,RT_AnomLang_WithCC]})

    fig_WithWithout = px.bar(df,x='Trial',y='RT',color='CC',barmode="group")

    fig_Cong_Bias = px.scatter(x=range(1,len(ActivationsCong['Biasing'])+1),y=ActivationsCong['Biasing'])
    fig_Cong_Agent = px.scatter(x=range(1,len(ActivationsCong['SubjIsAgent'])+1),y=ActivationsCong['SubjIsAgent'])
    fig_Cong_Theme = px.scatter(x=range(1,len(ActivationsCong['SubjIsTheme'])+1),y=ActivationsCong['SubjIsTheme'])
    fig_Incong_Bias = px.scatter(x=range(1,len(ActivationsIncong['Biasing'])+1),y=ActivationsIncong['Biasing'])
    fig_Incong_Agent = px.scatter(x=range(1,len(ActivationsIncong['SubjIsAgent'])+1),y=ActivationsIncong['SubjIsAgent'])
    fig_Incong_Theme = px.scatter(x=range(1,len(ActivationsIncong['SubjIsTheme'])+1),y=ActivationsIncong['SubjIsTheme'])

    return fig_WithWithout, fig_Cong_Bias,fig_Cong_Agent,fig_Cong_Theme, fig_Incong_Bias,fig_Incong_Agent,fig_Incong_Theme

def CreateFig_CrossTaskAdapt(params):
    RepresetationsDecayRate, CognitiveControlDecayRate, ActivationRate, MonitorBiasActivationRate, InhibitionRate, BiasingMult, MaxIter, ActivationThreshold, BetweenTrialsInterval, CongruentStroop, IncongruentStroop, CongruentSentence, AnomalousSentence = params

    # A Congruent Stroop -> Control Sentence sequence
    Results = RunTrialSequence((CongruentStroop,CongruentSentence),params)
    RT_CongCong = Results[1][0]
    print([(r[0],r[1]) for r in Results])

    # An Inongruent Stroop -> Control Sentence sequence
    Results = RunTrialSequence((IncongruentStroop,CongruentSentence),params)
    RT_IncongCong = Results[1][0]
    print([(r[0],r[1]) for r in Results])

    # A Congruent Stroop -> Anomalous Sentence sequence
    Results = RunTrialSequence((CongruentStroop,AnomalousSentence),params)
    RT_CongAnom = Results[1][0]
    print([(r[0],r[1]) for r in Results])

    # An Incongruent Stroop -> Anomalous Sentence sequence
    Results = RunTrialSequence((IncongruentStroop,AnomalousSentence),params)
    RT_IncongAnom = Results[1][0]
    print([(r[0],r[1]) for r in Results])

    # Figure
    df = pd.DataFrame({'PrevStroop':['Congruent','Incongruent','Congruent','Incongruent'],
                       'Trial':['Congruent','Congruent','Anomaly','Anomaly'],
                       'RT':[RT_CongCong,RT_IncongCong,RT_CongAnom,RT_IncongAnom]})

    fig = px.bar(df,x='PrevStroop',y='RT',color='Trial',barmode="group")
    return fig

