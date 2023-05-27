#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 21:37:24 2023
@author: talness
"""

import dash
import dash.dcc as dcc
import dash.html as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import statistics
import copy

# The model:
## Classes
class CognitiveControl(object):
    def __init__(self):
        self.ConflictMonitoring = 0
        self.Biasing = 0
 
    def Update(self,LingKnow,Stoop):
        # increase CM activity based on the ratio between SubjIsAgent & SubjIsTheme
        if LingKnow.SubjIsAgent+LingKnow.SubjIsTheme > 0:
            self.ConflictMonitoring += ActivationRate*1/(abs(LingKnow.SubjIsAgent-LingKnow.SubjIsTheme))
        # increase CM activity based on the ratio between Blue & Red
        if Stroop.Blue+Stroop.Red > 0:
            self.ConflictMonitoring += ActivationRate*1/(abs(Stroop.Blue-Stroop.Red))
        # increase B activation based on CM activation
        self.Biasing += MonitorBiasActivationRate*self.ConflictMonitoring       
        # decay
        self.ConflictMonitoring -= CognitiveControlDecayRate
        self.Biasing -= CognitiveControlDecayRate      
        # prevent negative activation values
        if self.ConflictMonitoring < 0:
            self.ConflictMonitoring = 0
        if self.Biasing < 0:
            self.Biasing = 0
    
    def BetweenTrialsDecay(self):
        self.ConflictMonitoring -= CognitiveControlDecayRate*BetweenTrialsInterval*1000
        self.Biasing -= CognitiveControlDecayRate*BetweenTrialsInterval*1000
        if self.ConflictMonitoring < 0:
            self.ConflictMonitoring = 0
        if self.Biasing < 0:
            self.Biasing = 0


            
class LinguisticKnowledge(object):
    def __init__(self, WorldAct=0, MorphSyntIngAct=0, MorphSyntEdAct=0):
        # Higher-Level
        self.WorldKnowledge = WorldAct
        self.MorphoSyntacticKnowledge_ing = MorphSyntIngAct
        self.MorphoSyntacticKnowledge_ed = MorphSyntEdAct
        # Lower-Level
        self.SubjIsAgent = 0
        self.SubjIsTheme = 0
        
    def InputAct(self,WK,MSK_ing,MSK_ed):
        self.WorldKnowledge += WK
        self.MorphoSyntacticKnowledge_ing += MSK_ing
        self.MorphoSyntacticKnowledge_ed += MSK_ed

    def Update(self,CogCtrl,Stoop):        
        # save initial activations for calculations (to make sure all updating happens "at once")
        WorldKnowledgeInitialAct = self.WorldKnowledge
        MorphoSyntacticKnowledgeIngInitialAct = self.MorphoSyntacticKnowledge_ing
        MorphoSyntacticKnowledgeEdInitialAct = self.MorphoSyntacticKnowledge_ed
        SubjIsAgentInitialAct = self.SubjIsAgent
        SubjIsThemeInitialAct = self.SubjIsTheme

        # increase WK & MSK activity based on CogCtrl.Biasing and their activation levels
        self.WorldKnowledge += (WorldKnowledgeInitialAct*CogCtrl.Biasing*BiasingMult)
        self.MorphoSyntacticKnowledge_ing += (MorphoSyntacticKnowledgeIngInitialAct*CogCtrl.Biasing*BiasingMult)        
        self.MorphoSyntacticKnowledge_ed += (MorphoSyntacticKnowledgeEdInitialAct*CogCtrl.Biasing*BiasingMult)        
        
        # increase SubjIsAgent & SubjIsTheme based on WK & MSK
        self.SubjIsAgent += (MorphoSyntacticKnowledgeIngInitialAct*ActivationRate) # SubjIsAgent is supported by MSK_ing
        self.SubjIsTheme += (MorphoSyntacticKnowledgeEdInitialAct*ActivationRate) # SubjIsTheme is supported by MSK_ed
        self.SubjIsTheme += (WorldKnowledgeInitialAct*ActivationRate) # SubjIsTheme is supported by WN
        
        # lateral inhibition between SubjIsAgent & SubjIsTheme
        self.SubjIsAgent -= (SubjIsThemeInitialAct*InhibitionRate)
        self.SubjIsTheme -= (SubjIsAgentInitialAct*InhibitionRate)
        
        # decay
        self.WorldKnowledge -= RepresetationsDecayRate
        self.MorphoSyntacticKnowledge_ing -= RepresetationsDecayRate
        self.MorphoSyntacticKnowledge_ed -= RepresetationsDecayRate
        self.SubjIsAgent -= RepresetationsDecayRate
        self.SubjIsTheme -= RepresetationsDecayRate
        # prevent negative activation values
        if self.WorldKnowledge < 0 :
            self.WorldKnowledge = 0
        if self.MorphoSyntacticKnowledge_ing <0:
            self.MorphoSyntacticKnowledge_ing = 0
        if self.MorphoSyntacticKnowledge_ed <0:
            self.MorphoSyntacticKnowledge_ed = 0
        if self.SubjIsAgent < 0:
            self.SubjIsAgent = 0
        if self.SubjIsTheme < 0:
            self.SubjIsTheme = 0
    
    def BetweenTrialsDecay(self):
        self.WorldKnowledge -= RepresetationsDecayRate*BetweenTrialsInterval*1000
        self.MorphoSyntacticKnowledge_ing -= RepresetationsDecayRate*BetweenTrialsInterval*1000
        self.MorphoSyntacticKnowledge_ed -= RepresetationsDecayRate*BetweenTrialsInterval*1000
        self.SubjIsAgent -= RepresetationsDecayRate*BetweenTrialsInterval*1000
        self.SubjIsTheme -= RepresetationsDecayRate*BetweenTrialsInterval*1000
        if self.WorldKnowledge < 0 :
            self.WorldKnowledge = 0
        if self.MorphoSyntacticKnowledge_ing <0:
            self.MorphoSyntacticKnowledge_ing = 0
        if self.MorphoSyntacticKnowledge_ed <0:
            self.MorphoSyntacticKnowledge_ed = 0
        if self.SubjIsAgent < 0:
            self.SubjIsAgent = 0
        if self.SubjIsTheme < 0:
            self.SubjIsTheme = 0


        
class StroopTaskRepresentation(object):
    def __init__(self, TextBlueAct=0, TextRedAct=0, FontColAct=0):
        # Higher-Level
        self.Text_blue = TextBlueAct
        self.Text_red = TextRedAct
        self.FontColor = FontColAct
        # Lower-Level
        self.Blue = 0
        self.Red = 0
            
    def InputAct(self,Text_blue,Text_red,FontColor):
        self.Text_blue += Text_blue
        self.Text_red += Text_red
        self.FontColor += FontColor
        

    def Update(self,CogCtrl,LingKnow):
        # save initial activations for calculations (to make sure all updating happens "at once")
        TextBlueInitialAct = self.Text_blue
        TextRedInitialAct = self.Text_red
        FontColorInitialAct = self.FontColor
        BlueInitialAct = self.Blue
        RedInitialAct = self.Red
        
        # increase Text & FontColor activity based on CogCtrl.Biasing and their activation levels
        self.Text_blue += (TextBlueInitialAct*CogCtrl.Biasing*BiasingMult)
        self.Text_red += (TextRedInitialAct*CogCtrl.Biasing*BiasingMult)
        self.FontColor += (FontColorInitialAct*CogCtrl.Biasing*BiasingMult)

        # increase Blue & Red based activity based on Text & FontColor
        self.Blue += (FontColorInitialAct*ActivationRate) # Blue is supported by FontColor
        self.Blue += (TextBlueInitialAct*ActivationRate) # Blue is supported by Text_blue
        self.Red += (TextRedInitialAct*ActivationRate) # Red is supported by Text_red

        # lateral inhibition between Blue & Red
        self.Blue -= (RedInitialAct*InhibitionRate)
        self.Red -= (BlueInitialAct*InhibitionRate)
        
        # decay
        self.Text_blue -= RepresetationsDecayRate
        self.Text_red -= RepresetationsDecayRate
        self.FontColor -= RepresetationsDecayRate
        self.Blue -= RepresetationsDecayRate
        self.Red -= RepresetationsDecayRate
        # prevent negative activation values
        if self.Text_blue < 0:
            self.Text_blue = 0
        if self.Text_red < 0:
            self.Text_red = 0
        if self.FontColor <0:
            self.FontColor = 0
        if self.Blue < 0:
            self.Blue = 0
        if self.Red < 0:
            self.Red = 0

    def BetweenTrialsDecay(self):
        self.Text_blue -= RepresetationsDecayRate*BetweenTrialsInterval*1000
        self.Text_red -= RepresetationsDecayRate*BetweenTrialsInterval*1000
        self.FontColor -= RepresetationsDecayRate*BetweenTrialsInterval*1000
        self.Blue -= RepresetationsDecayRate*BetweenTrialsInterval*1000
        self.Red -= RepresetationsDecayRate*BetweenTrialsInterval*1000
        if self.Text_blue < 0:
            self.Text_blue = 0
        if self.Text_red < 0:
            self.Text_red = 0
        if self.FontColor <0:
            self.FontColor = 0
        if self.Blue < 0:
            self.Blue = 0
        if self.Red < 0:
            self.Red = 0

        
## Functions
def UpdateAll(CogCtrl,LingKnow,Stroop):    
    CogCtrl_prev = copy.copy(CogCtrl)
    LingKnow_prev = copy.copy(LingKnow)
    Stroop_prev = copy.copy(Stroop)
    
    CogCtrl.Update(LingKnow_prev,Stroop_prev)
    LingKnow.Update(CogCtrl_prev,Stroop_prev)
    Stroop.Update(CogCtrl_prev,LingKnow_prev)
    return [CogCtrl, LingKnow, Stroop]


def RunTrial(InputAct:tuple,CC=None,LK=None,S=None):
    '''
    InputAct specifies the input activations for WK,MSK_ing,MSK_ed,Text_blue,Text_red,FontColor (in that order).
    '''
    i = 1
    ActivationLevels = {'ConfMon':[],'Biasing':[],
                        'WK':[],'MSK_ing':[],'MSK_ed':[],'SubjIsAgent':[],'SubjIsTheme':[],
                       'FontColor':[],'Text_blue':[],'Text_red':[],'Blue':[],'Red':[]}

    global CogCtrl
    global LingKnow
    global Stroop

    if CC == None:
        CogCtrl = CognitiveControl()
    else:
        CogCtrl = CC
        
    if LK == None:
        LingKnow = LinguisticKnowledge()
    else:
        LingKnow = LK
        
    if S == None:
        Stroop = StroopTaskRepresentation()
    else:
        Stroop = S
        

    LingKnow.InputAct(InputAct[0],InputAct[1],InputAct[2])
    Stroop.InputAct(InputAct[3],InputAct[4],InputAct[5])

    NodesDict = {'ConfMon':CogCtrl.ConflictMonitoring,'Biasing':CogCtrl.Biasing,
                        'WK':LingKnow.WorldKnowledge,'MSK_ing':LingKnow.MorphoSyntacticKnowledge_ing,'MSK_ed':LingKnow.MorphoSyntacticKnowledge_ed,'SubjIsAgent':LingKnow.SubjIsAgent,'SubjIsTheme':LingKnow.SubjIsTheme,
                       'FontColor':Stroop.FontColor,'Text_blue':Stroop.Text_blue,'Text_red':Stroop.Text_red,'Blue':Stroop.Blue,'Red':Stroop.Red}
    for key in ActivationLevels.keys():
        ActivationLevels[key].append(NodesDict[key])
    
    print('BiasingMult = %2d' % BiasingMult)

    while i <= MaxIter:
        UpdateAll(CogCtrl,LingKnow,Stroop)
        NodesDict = {'ConfMon':CogCtrl.ConflictMonitoring,'Biasing':CogCtrl.Biasing,
                        'WK':LingKnow.WorldKnowledge,'MSK_ing':LingKnow.MorphoSyntacticKnowledge_ing,'MSK_ed':LingKnow.MorphoSyntacticKnowledge_ed,'SubjIsAgent':LingKnow.SubjIsAgent,'SubjIsTheme':LingKnow.SubjIsTheme,
                       'FontColor':Stroop.FontColor,'Text_blue':Stroop.Text_blue,'Text_red':Stroop.Text_red,'Blue':Stroop.Blue,'Red':Stroop.Red}
        for key in ActivationLevels.keys():
            ActivationLevels[key].append(NodesDict[key])
        i += 1
        MaxAct = max(LingKnow.SubjIsAgent,LingKnow.SubjIsTheme,Stroop.Blue,Stroop.Red)
        if MaxAct > ActivationThreshold:
            break
    
    # determine the final interpratation
    if [LingKnow.SubjIsAgent,LingKnow.SubjIsTheme,Stroop.Blue,Stroop.Red].count(MaxAct) > 1:
        Winner = None
    if LingKnow.SubjIsAgent == MaxAct:
        Winner = 'SubjIsAgent'
    if LingKnow.SubjIsTheme == MaxAct:
        Winner = 'SubjIsTheme'
    if Stroop.Blue == MaxAct:
        Winner = 'Blue'
    if Stroop.Red == MaxAct:
        Winner = 'Red'
    
    return i, Winner, CogCtrl, LingKnow, Stroop, ActivationLevels


def RunTrialSequence(Trials:"list of tuples"):
    '''
    Trials is a list of tuples.
    Each tuple consists of 4 values that are the input activations of WK,MSK_ing,MSK_ed,Text_blue,Text_red,FontColor (in that order) in a specific trial.
    '''
    CogCtrl = CognitiveControl()
    LingKnow = LinguisticKnowledge()
    Stroop = StroopTaskRepresentation()
    
    Results = []
    for Trial in Trials:
        CogCtrl = CogCtrl.BetweenTrialsDecay()
        LingKnow = LingKnow.BetweenTrialsDecay()
        Stroop = Stroop.BetweenTrialsDecay()
        i, Winner, CogCtrl, LingKnow, Stroop, Activations = RunTrial(Trial,CogCtrl,LingKnow,Stroop)
        Results.append([i,Winner,Activations])
        
    return Results

# Initialize the data
data = {
    'iteration': [],
    'winner': [],
    'ConflictMonitoring': [],
    'Biasing': [],
    'WorldKnowledge': [],
    'MorphoSyntacticKnowledge_ing': [],
    'MorphoSyntacticKnowledge_ed': [],
    'SubjIsAgent': [],
    'SubjIsTheme': [],
    'Text_blue': [],
    'Text_red': [],
    'FontColor': [],
    'Blue': [],
    'Red': []
}

# Create initial objects
CogCtrl = None
LingKnow = None
Stroop = None

# Define the app layout
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Cognitive Control Simulation Dashboard"),
    html.Div([
        html.H3("Parameters"),
        html.Label("Represetations Decay Rate  "),
        dcc.Input(id='decay-rate', type='number', value=0.01),
        html.Span(style={'margin-right': '40px'}),
        html.Label("Cognitive Control Decay Rate  "),
        dcc.Input(id='control-decay-rate', type='number', value=0.0000001),
        html.Br(),
        html.Label("Activation Rate  "),
        dcc.Input(id='activation-rate', type='number', value=0.1),
        html.Span(style={'margin-right': '40px'}),
        html.Label("Monitoring to Biasing Activation Rate  "),
        dcc.Input(id='monitor-bias-activation-rate', type='number', value=1),
        html.Span(style={'margin-right': '40px'}),
        html.Label("Inhibition Rate  "),
        dcc.Input(id='inhibition-rate', type='number', value=0.1),
        html.Span(style={'margin-right': '40px'}),
        html.Label("Biasing Multiplier  "),
        dcc.Input(id='biasing-mult', type='number', value=0.01),
        html.Br(),
        html.Label("Maximum Iterations  "),
        dcc.Input(id='max-iter', type='number', value=10000),
        html.Span(style={'margin-right': '40px'}),
        html.Label("Activation Threshold  "),
        dcc.Input(id='activation-threshold', type='number', value=100),
        html.Span(style={'margin-right': '40px'}),
        html.Label("Between Trials Interval  "),
        dcc.Input(id='between-trials-interval', type='number', value=1.5),
    ]),
    html.Div([html.Br(),
        html.Label("Input Activations:"),
        html.Br(),
        html.Label("Congruent Stroop  "),
        dcc.Input(
            id='input-congruent-stroop',
            type='text',
            value='0,0,0,10,0,15'
        ),
        html.Span(style={'margin-right': '40px'}),
        html.Label("Incongruent Stroop  "),
        dcc.Input(
            id='input-incongruent-stroop',
            type='text',
            value='0,0,0,0,10,15'
        ),
        html.Span(style={'margin-right': '40px'}),
        html.Label("Congruent Sentence  "),
        dcc.Input(
            id='input-congruent-sentence',
            type='text',
            value='15,0,10,0,0,0'
        ),
        html.Span(style={'margin-right': '40px'}),
        html.Label("Anomalous Sentence  "),
        dcc.Input(
            id='input-anomalous-sentence',
            type='text',
            value='15,10,0,0,0,0'
        ),
    ]),
    html.Div([html.Br(),
        html.Button('Run Simulations', id='run-button', n_clicks=0),
    ]),
    html.Div([
        html.H3("Simulation Results"),
        html.Div([dcc.Graph(id='iteration-graph')],
                 style={'width': '50%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='accuracy-graph')],
                 style={'width': '50%', 'display': 'inline-block'}),
        html.Div(dcc.Graph(id='reaction-time-graph'),
                 style={'width': '50%', 'margin': '0 auto'}),
        html.Label("Activation plots - Congruent sentence:"),
        html.Br(),
        html.Div([dcc.Graph(id='activ-cong-bias-graph')],
                 style={'width': '33%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='activ-cong-agent-graph')],
                 style={'width': '33%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='activ-cong-theme-graph')],
                 style={'width': '33%', 'display': 'inline-block'}),
        html.Label("Activation plots - Incongruent sentence:"),
        html.Br(),
        html.Div([dcc.Graph(id='activ-incong-bias-graph')],
                 style={'width': '33%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='activ-incong-agent-graph')],
                 style={'width': '33%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='activ-incong-theme-graph')],
                 style={'width': '33%', 'display': 'inline-block'}),
    ])
])


# Define callback functions
@app.callback(
    [Output('iteration-graph', 'figure'),
     Output('accuracy-graph', 'figure'),
     Output('reaction-time-graph', 'figure'),
     Output('activ-cong-bias-graph', 'figure'),
     Output('activ-cong-agent-graph', 'figure'),
     Output('activ-cong-theme-graph', 'figure'),
     Output('activ-incong-bias-graph', 'figure'),
     Output('activ-incong-agent-graph', 'figure'),
     Output('activ-incong-theme-graph', 'figure')],
    [Input('run-button', 'n_clicks')],
    [State('decay-rate', 'value'),
     State('control-decay-rate', 'value'),
     State('activation-rate', 'value'),
     State('monitor-bias-activation-rate', 'value'),
     State('inhibition-rate', 'value'),
     State('biasing-mult', 'value'),
     State('max-iter', 'value'),
     State('activation-threshold', 'value'),
     State('between-trials-interval', 'value'),
     State('input-congruent-stroop', 'value'),
     State('input-incongruent-stroop', 'value'),
     State('input-congruent-sentence', 'value'),
     State('input-anomalous-sentence', 'value')])
def run_simulation(n_clicks, decay_rate, control_decay_rate, activation_rate, monitor_bias_activation_rate, inhibition_rate, biasing_mult,
                   max_iter, activation_threshold, between_trials_interval,
                   input_congruent_stroop, input_incongruent_stroop, input_congruent_sentence, input_anomalous_sentence):

    input_congruent_stroop = tuple(map(int,input_congruent_stroop.split(',')))
    input_incongruent_stroop = tuple(map(int,input_incongruent_stroop.split(',')))
    input_congruent_sentence = tuple(map(int,input_congruent_sentence.split(',')))
    input_anomalous_sentence = tuple(map(int,input_anomalous_sentence.split(',')))

    global RepresetationsDecayRate, CognitiveControlDecayRate, ActivationRate, MonitorBiasActivationRate, InhibitionRate, BiasingMult, MaxIter, ActivationThreshold, BetweenTrialsInterval, CongruentStroop, IncongruentStroop, CongruentSentence, AnomalousSentence
    RepresetationsDecayRate, CognitiveControlDecayRate, ActivationRate, MonitorBiasActivationRate, InhibitionRate, BiasingMult, MaxIter, ActivationThreshold, BetweenTrialsInterval, CongruentStroop, IncongruentStroop, CongruentSentence, AnomalousSentence = decay_rate, control_decay_rate, activation_rate, monitor_bias_activation_rate ,inhibition_rate, biasing_mult, max_iter, activation_threshold, between_trials_interval, input_congruent_stroop, input_incongruent_stroop, input_congruent_sentence, input_anomalous_sentence

    iteration_graph = CreateFig_WithWithoutCC_Stroop()
    accuracy_graph,act_cong_bias_graph,act_cong_agent_graph,act_cong_theme_graph,act_incong_bias_graph,act_incong_agent_graph,act_incong_theme_graph = CreateFig_WithWithoutCC_Lang()
    reaction_time_graph = CreateFig_CrossTaskAdapt()
    return iteration_graph, accuracy_graph, reaction_time_graph, act_cong_bias_graph,act_cong_agent_graph,act_cong_theme_graph,act_incong_bias_graph,act_incong_agent_graph,act_incong_theme_graph


def CreateFig_WithWithoutCC_Stroop():
    global BiasingMult
    UserBiasingMult = BiasingMult
    # Congruent Stroop trial with BiasingMult = 0
    BiasingMult = 0
    i, Winner, CogCtrl, LingKnow, Stroop, Activations = RunTrial(CongruentStroop)
    RT_CongStroop_NoCC = i
    print(i, Winner)

    # Congruent Stroop trial with BiasingMult = 1
    BiasingMult = UserBiasingMult
    i, Winner, CogCtrl, LingKnow, Stroop, Activations = RunTrial(CongruentStroop)
    RT_CongStroop_WithCC = i
    print(i, Winner)

    # Inongruent Stroop trial with BiasingMult = 0
    BiasingMult = 0
    i, Winner, CogCtrl, LingKnow, Stroop, Activations = RunTrial(IncongruentStroop)
    RT_IncongStroop_NoCC = i
    print(i, Winner)

    # Incongruent Stroop trial with BiasingMult = 1
    BiasingMult = UserBiasingMult
    i, Winner, CogCtrl, LingKnow, Stroop, Activations = RunTrial(IncongruentStroop)
    RT_IncongStroop_WithCC = i
    print(i, Winner)

    # Figure
    df = pd.DataFrame({'CC':['Without CC','With CC','Without CC','With CC'],
                       'Trial':['Congruent','Congruent','Incongruent','Incongruent'],
                       'RT':[RT_CongStroop_NoCC,RT_CongStroop_WithCC,RT_IncongStroop_NoCC,RT_IncongStroop_WithCC]})

    fig = px.bar(df,x='Trial',y='RT',color='CC',barmode="group")
    return fig

def CreateFig_WithWithoutCC_Lang():
    global BiasingMult
    UserBiasingMult = BiasingMult
    # Control sentence trial with BiasingMult = 0
    BiasingMult = 0
    i, Winner, CogCtrl, LingKnow, Stroop, Activations = RunTrial(CongruentSentence)
    RT_CongLang_NoCC = i
    print(i, Winner)

    # control sentence trial with BiasingMult = 1
    BiasingMult = UserBiasingMult
    i, Winner, CogCtrl, LingKnow, Stroop, Activations = RunTrial(CongruentSentence)
    RT_CongLang_WithCC = i
    ActivationsCong = Activations
    print(i, Winner)

    # Anomalous sentence trial with BiasingMult = 0
    BiasingMult = 0
    i, Winner, CogCtrl, LingKnow, Stroop, Activations = RunTrial(AnomalousSentence)
    RT_AnomLang_NoCC = i
    print(i, Winner)

    # Anomalous sentence trial with BiasingMult = 1
    BiasingMult = UserBiasingMult
    i, Winner, CogCtrl, LingKnow, Stroop, Activations = RunTrial(AnomalousSentence)
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

def CreateFig_CrossTaskAdapt():
    # A Congruent Stroop -> Control Sentence sequence
    Results = RunTrialSequence((CongruentStroop,CongruentSentence))
    RT_CongCong = Results[1][0]
    print([(r[0],r[1]) for r in Results])

    # An Inongruent Stroop -> Control Sentence sequence
    Results = RunTrialSequence((IncongruentStroop,CongruentSentence))
    RT_IncongCong = Results[1][0]
    print([(r[0],r[1]) for r in Results])

    # A Congruent Stroop -> Anomalous Sentence sequence
    Results = RunTrialSequence((CongruentStroop,AnomalousSentence))
    RT_CongAnom = Results[1][0]
    print([(r[0],r[1]) for r in Results])

    # An Incongruent Stroop -> Anomalous Sentence sequence
    Results = RunTrialSequence((IncongruentStroop,AnomalousSentence))
    RT_IncongAnom = Results[1][0]
    print([(r[0],r[1]) for r in Results])

    # Figure
    df = pd.DataFrame({'PrevStroop':['Congruent','Incongruent','Congruent','Incongruent'],
                       'Trial':['Congruent','Congruent','Anomaly','Anomaly'],
                       'RT':[RT_CongCong,RT_IncongCong,RT_CongAnom,RT_IncongAnom]})

    fig = px.bar(df,x='PrevStroop',y='RT',color='Trial',barmode="group")
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
