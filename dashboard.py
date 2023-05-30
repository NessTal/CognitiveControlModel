import dash
import dash.dcc as dcc
import dash.html as html
from dash.dependencies import Input, Output, State

from model import *
from figure_creation import *

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
        html.Label("For an explanation, Hover over a parameter's name"),
        html.Br(), 
        html.Label("Represetations Decay Rate  ", title="The number of activation units lost every iteration, from each Stroop/LanguageKnowledge node"),
        dcc.Input(id='decay-rate', type='number', value=0.01),
        html.Span(style={'margin-right': '40px'}),
        html.Label("Cognitive Control Decay Rate  ", title="The number of activation units lost every iteration, from each Cognitive Control node)"),
        dcc.Input(id='control-decay-rate', type='number', value=0.005),
        html.Br(),
        html.Label("Activation Rate  ", title="The relation between unit activation and the activation it exerts through excitatory connections"),
        dcc.Input(id='activation-rate', type='number', value=0.1),
        html.Span(style={'margin-right': '40px'}),
        html.Label("Monitoring to Biasing Activation Rate  ", title='The relation between the activation of Conflict Monitoring and the activation it exerts to the Biasing Unit (a value of 1 models a case in which Monitoring is "inside" Biasing)'),
        dcc.Input(id='monitor-bias-activation-rate', type='number', value=1),
        html.Span(style={'margin-right': '40px'}),
        html.Label("Inhibition Rate  ", title="The relation between unit activation and the inhibition it exerts through inhibitory connections"),
        dcc.Input(id='inhibition-rate', type='number', value=0.1),
        html.Span(style={'margin-right': '40px'}),
        html.Label("Biasing Multiplier  ", title="The relation between Biasing unit activation level and how strongly it activates Stroop/LanguageKnowledge high-level nodes in proportion to their activation levels"),
        dcc.Input(id='biasing-mult', type='number', value=0.00002),
        html.Br(),
        html.Label("Activation Threshold  ", title='When a lower-level unit\'s activation level reaches this threshold, it "wins" and is cosidered the final interpratation'),
        dcc.Input(id='activation-threshold', type='number', value=1000),
        html.Span(style={'margin-right': '40px'}),
        html.Label("Maximum Iterations  ", title="Maximum number of iterations per trial. If reached, the lower-level unit with highest activation level is cosidered the final interpratation"),
        dcc.Input(id='max-iter', type='number', value=5000),
        html.Span(style={'margin-right': '40px'}),
        html.Label("Between Trials Interval  ", title="Time (in seconds) between consecutive trials"),
        dcc.Input(id='between-trials-interval', type='number', value=1.5),
    ]),
    html.Div([html.Br(),
        html.Label("Input Activations:", title= "Input activations for each trial type (order: WK,MSK_ing,MSK_ed,Text_blue,Text_red,FontColor)"),
        html.Br(),
        html.Label("Congruent Stroop  ", title= "Input activations (order: WK,MSK_ing,MSK_ed,Text_blue,Text_red,FontColor). For a congruent Stroop, activation of FontColor that supports Blue & Text_blue that supports Blue"),
        dcc.Input(id='input-congruent-stroop', type='text', value='0,0,0,10,0,15'),
        html.Span(style={'margin-right': '40px'}),
        html.Label("Incongruent Stroop  ", title= "Input activations (order: WK,MSK_ing,MSK_ed,Text_blue,Text_red,FontColor). For an Incongruent Stroop, activation of FontColor that supports Blue & Text_red that supports red "),
        dcc.Input(id='input-incongruent-stroop', type='text', value='0,0,0,0,10,15'),
        html.Span(style={'margin-right': '40px'}),
        html.Label("Congruent Sentence  ", title= "Input activations (order: WK,MSK_ing,MSK_ed,Text_blue,Text_red,FontColor). For a Congruent Sentence, activation of WorldKnowledge that supports SubjIsTheme & MorphosyntacticKnowledge_ed that supports SubjIsTheme"),
        dcc.Input(id='input-congruent-sentence', type='text', value='15,0,10,0,0,0'),
        html.Span(style={'margin-right': '40px'}),
        html.Label("Anomalous Sentence  ", title= "Input activations (order: WK,MSK_ing,MSK_ed,Text_blue,Text_red,FontColor). For an Anomalous Sentence, activation of WorldKnowledge that supports SubjIsTheme & MorphosyntacticKnowledge_ing that supports SubjIsAgent"),
        dcc.Input(id='input-anomalous-sentence', type='text', value='15,10,0,0,0,0'),
    ]),
    html.Div([html.Br(),
        html.Button('Run Simulations', id='run-button', n_clicks=0),
    ]),
    html.Div([
        html.H3("Simulation Results"),
        html.Div([dcc.Graph(id='stroop-w\wo-CC-graph')],
                 style={'width': '50%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='language-w\wo-CC-graph')],
                 style={'width': '50%', 'display': 'inline-block'}),
        html.Div(dcc.Graph(id='cross-task-adaptation-graph'),
                 style={'width': '50%', 'margin': '0 auto'}),
        html.Label("Activation plots - Congruent sentence:"),
        html.Br(),
        html.Div([dcc.Graph(id='activations-cong-bias-graph')],
                 style={'width': '33%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='activations-cong-agent-graph')],
                 style={'width': '33%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='activations-cong-theme-graph')],
                 style={'width': '33%', 'display': 'inline-block'}),
        html.Label("Activation plots - Incongruent sentence:"),
        html.Br(),
        html.Div([dcc.Graph(id='activations-incong-bias-graph')],
                 style={'width': '33%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='activations-incong-agent-graph')],
                 style={'width': '33%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='activations-incong-theme-graph')],
                 style={'width': '33%', 'display': 'inline-block'}),
    ])
])


# Define callback functions
@app.callback(
    [Output('stroop-w\wo-CC-graph', 'figure'),
     Output('language-w\wo-CC-graph', 'figure'),
     Output('cross-task-adaptation-graph', 'figure'),
     Output('activations-cong-bias-graph', 'figure'),
     Output('activations-cong-agent-graph', 'figure'),
     Output('activations-cong-theme-graph', 'figure'),
     Output('activations-incong-bias-graph', 'figure'),
     Output('activations-incong-agent-graph', 'figure'),
     Output('activations-incong-theme-graph', 'figure')],
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
    params = (RepresetationsDecayRate, CognitiveControlDecayRate, ActivationRate, MonitorBiasActivationRate, InhibitionRate, BiasingMult, MaxIter, ActivationThreshold, BetweenTrialsInterval, CongruentStroop, IncongruentStroop, CongruentSentence, AnomalousSentence)
    iteration_graph = CreateFig_WithWithoutCC_Stroop(params)
    accuracy_graph,act_cong_bias_graph,act_cong_agent_graph,act_cong_theme_graph,act_incong_bias_graph,act_incong_agent_graph,act_incong_theme_graph = CreateFig_WithWithoutCC_Lang(params)
    reaction_time_graph = CreateFig_CrossTaskAdapt(params)
    return iteration_graph, accuracy_graph, reaction_time_graph, act_cong_bias_graph,act_cong_agent_graph,act_cong_theme_graph,act_incong_bias_graph,act_incong_agent_graph,act_incong_theme_graph

if __name__ == '__main__':
    app.run_server(debug=True)