import plotly.express as px
import pandas as pd

from model import *

def create_fig_with_withoutCC_stroop(params):
    represetations_decay_rate, cognitive_control_decay_rate, activation_rate, monitor_bias_activation_rate, inhibition_rate, biasing_mult, max_iter, activation_threshold, between_trials_interval, congruent_stroop, incongruent_stroop, congruent_sentence, anomalous_sentence = params
    params_no_CC = represetations_decay_rate, cognitive_control_decay_rate, activation_rate, monitor_bias_activation_rate, inhibition_rate, 0, max_iter, activation_threshold, between_trials_interval, congruent_stroop, incongruent_stroop, congruent_sentence, anomalous_sentence

    user_biasing_mult = biasing_mult
    # Congruent Stroop trial with biasing_mult = 0
    biasing_mult = 0
    i, winner, CogCtrl, LingKnow, Stroop, activations = run_trial(congruent_stroop,params_no_CC)
    rt_cong_stroop_no_CC = i
    print(i, winner)

    # Congruent Stroop trial with biasing_mult = 1
    biasing_mult = user_biasing_mult
    i, winner, CogCtrl, LingKnow, Stroop, activations = run_trial(congruent_stroop,params)
    rt_cong_stroop_with_CC = i
    print(i, winner)

    # Inongruent Stroop trial with biasing_mult = 0
    biasing_mult = 0
    i, winner, CogCtrl, LingKnow, Stroop, activations = run_trial(incongruent_stroop,params_no_CC)
    rt_incong_stroop_no_CC = i
    print(i, winner)

    # Incongruent Stroop trial with biasing_mult = 1
    biasing_mult = user_biasing_mult
    i, winner, CogCtrl, LingKnow, Stroop, activations = run_trial(incongruent_stroop,params)
    rt_incong_stroop_with_CC = i
    print(i, winner)

    # Figure
    df = pd.DataFrame({'CC':['Without','With','Without','With'],
                       'Trial':['Congruent','Congruent','Incongruent','Incongruent'],
                       'RT':[rt_cong_stroop_no_CC,rt_cong_stroop_with_CC,rt_incong_stroop_no_CC,rt_incong_stroop_with_CC]})

    fig = px.bar(df,x='Trial',y='RT',color='CC',barmode="group",
                 title = "Stroop trials with vs. without cognitive control",
                 labels = {"Trial" : "Trial Type", "RT": "Reaction time (iterations)", "CC" : "Cognitive Control"})
    fig.update_layout(title_x = 0.5)
    return fig


def create_fig_with_withoutCC_lang(params):
    represetations_decay_rate, cognitive_control_decay_rate, activation_rate, monitor_bias_activation_rate, inhibition_rate, biasing_mult, max_iter, activation_threshold, between_trials_interval, congruent_stroop, incongruent_stroop, congruent_sentence, anomalous_sentence = params
    params_no_CC = represetations_decay_rate, cognitive_control_decay_rate, activation_rate, monitor_bias_activation_rate, inhibition_rate, 0, max_iter, activation_threshold, between_trials_interval, congruent_stroop, incongruent_stroop, congruent_sentence, anomalous_sentence

    user_biasing_mult = biasing_mult
    # Control sentence trial with biasing_mult = 0
    biasing_mult = 0
    i, winner, CogCtrl, LingKnow, Stroop, activations = run_trial(congruent_sentence,params_no_CC)
    rt_cong_lang_no_CC = i
    print(i, winner)

    # control sentence trial with biasing_mult = 1
    biasing_mult = user_biasing_mult
    i, winner, CogCtrl, LingKnow, Stroop, activations = run_trial(congruent_sentence,params)
    rt_cong_lang_With_CC = i
    activations_cong = activations
    print(i, winner)

    # Anomalous sentence trial with biasing_mult = 0
    biasing_mult = 0
    i, winner, CogCtrl, LingKnow, Stroop, activations = run_trial(anomalous_sentence,params_no_CC)
    rt_anom_lang_no_CC = i
    print(i, winner)

    # Anomalous sentence trial with biasing_mult = 1
    biasing_mult = user_biasing_mult
    i, winner, CogCtrl, LingKnow, Stroop, activations = run_trial(anomalous_sentence,params)
    rt_anom_lang_with_CC = i
    activations_incong = activations
    print(i, winner)

    # Figure
    df = pd.DataFrame({'CC':['Without','With','Without','With'],
                       'Trial':['Congruent','Congruent','Anomaly','Anomaly'],
                       'RT':[rt_cong_lang_no_CC,rt_cong_lang_With_CC,rt_anom_lang_no_CC,rt_anom_lang_with_CC]})

    fig_with_without = px.bar(df,x='Trial',y='RT',color='CC',barmode="group",
                             title = "Linguistic trials with vs. without cognitive control",
                             labels = {"Trial" : "Sentence Type", "RT": "Reaction time (iterations)", "CC" : "Cognitive Control"})
    fig_with_without.update_layout(title_x = 0.5)
    
    fig_cong_bias = px.scatter(x=range(1,len(activations_cong['Biasing'])+1),y=activations_cong['Biasing'],
                               title = "Cognitive control - Biasing unit")
    fig_cong_bias.update_layout(xaxis_title = "Time (iterations)",yaxis_title = "Activation level",title_x = 0.5)
    fig_cong_agent = px.scatter(x=range(1,len(activations_cong['SubjIsAgent'])+1),y=activations_cong['SubjIsAgent'],
                               title = "Linguistic - Subject is Agent")
    fig_cong_agent.update_layout(xaxis_title = "Time (iterations)",yaxis_title = "Activation level",title_x = 0.5)
    fig_cong_theme = px.scatter(x=range(1,len(activations_cong['SubjIsTheme'])+1),y=activations_cong['SubjIsTheme'],
                               title = "Linguistic - Subject is Theme")
    fig_cong_theme.update_layout(xaxis_title = "Time (iterations)",yaxis_title = "Activation level",title_x = 0.5)
    fig_incong_bias = px.scatter(x=range(1,len(activations_incong['Biasing'])+1),y=activations_incong['Biasing'],
                               title = "Cognitive control - Biasing unit")
    fig_incong_bias.update_layout(xaxis_title = "Time (iterations)",yaxis_title = "Activation level",title_x = 0.5)
    fig_incong_agent = px.scatter(x=range(1,len(activations_incong['SubjIsAgent'])+1),y=activations_incong['SubjIsAgent'],
                               title = "Linguistic - Subject is Agent")
    fig_incong_agent.update_layout(xaxis_title = "Time (iterations)",yaxis_title = "Activation level",title_x = 0.5)
    fig_incong_theme = px.scatter(x=range(1,len(activations_incong['SubjIsTheme'])+1),y=activations_incong['SubjIsTheme'],
                               title = "Linguistic - Subject is Theme")
    fig_incong_theme.update_layout(xaxis_title = "Time (iterations)",yaxis_title = "Activation level",title_x = 0.5)

    return fig_with_without, fig_cong_bias,fig_cong_agent,fig_cong_theme, fig_incong_bias,fig_incong_agent,fig_incong_theme

def create_fig_crosstask_adapt(params):
    represetations_decay_rate, cognitive_control_decay_rate, activation_rate, monitor_bias_activation_rate, inhibition_rate, biasing_mult, max_iter, activation_threshold, between_trials_interval, congruent_stroop, incongruent_stroop, congruent_sentence, anomalous_sentence = params

    # A Congruent Stroop -> Control Sentence sequence
    results = run_trial_sequence((congruent_stroop,congruent_sentence),params)
    rt_cong_cong = results[1][0]
    print([(r[0],r[1]) for r in results])

    # An Inongruent Stroop -> Control Sentence sequence
    results = run_trial_sequence((incongruent_stroop,congruent_sentence),params)
    rt_incong_cong = results[1][0]
    print([(r[0],r[1]) for r in results])

    # A Congruent Stroop -> Anomalous Sentence sequence
    results = run_trial_sequence((congruent_stroop,anomalous_sentence),params)
    rt_cong_Anom = results[1][0]
    print([(r[0],r[1]) for r in results])

    # An Incongruent Stroop -> Anomalous Sentence sequence
    results = run_trial_sequence((incongruent_stroop,anomalous_sentence),params)
    rt_incong_Anom = results[1][0]
    print([(r[0],r[1]) for r in results])

    # Figure
    df = pd.DataFrame({'PrevStroop':['Congruent','Incongruent','Congruent','Incongruent'],
                       'Trial':['Congruent','Congruent','Anomaly','Anomaly'],
                       'RT':[rt_cong_cong,rt_incong_cong,rt_cong_Anom,rt_incong_Anom]})

    fig = px.bar(df,x='PrevStroop',y='RT',color='Trial',barmode="group",
                 title = "Linguistic trials when preceding stroop trial is congruent vs. incongruent",
                 labels = {"PrevStroop":"Preceding Stroop", "RT": "Reaction time (iterations)","Trial" : "Sentence Type"})
    fig.update_layout(title_x = 0.5)
    
    return fig

