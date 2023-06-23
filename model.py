import copy


# Classes
class CognitiveControl(object):
    def __init__(self):
        self.conflict_monitoring = 0
        self.biasing = 0
 
    def update(self,LingKnow,Stoop):
        # increase CM activity based on the ratio between SubjIsAgent & SubjIsTheme
        if LingKnow.subj_is_agent+LingKnow.subj_is_theme > 0:
            self.conflict_monitoring += activation_rate*1/(abs(LingKnow.subj_is_agent-LingKnow.subj_is_theme))
        # increase CM activity based on the ratio between Blue & Red
        if Stroop.blue+Stroop.red > 0:
            self.conflict_monitoring += activation_rate*1/(abs(Stroop.blue-Stroop.red))
        # increase B activation based on CM activation
        self.biasing += monitor_bias_activation_rate*self.conflict_monitoring       
        # decay
        self.conflict_monitoring -= cognitive_control_decay_rate
        self.biasing -= cognitive_control_decay_rate      
        # prevent negative activation values
        if self.conflict_monitoring < 0:
            self.conflict_monitoring = 0
        if self.biasing < 0:
            self.biasing = 0
    
    def between_trials_decay(self):
        self.conflict_monitoring -= cognitive_control_decay_rate*between_trials_interval*1000
        self.biasing -= cognitive_control_decay_rate*between_trials_interval*1000
        if self.conflict_monitoring < 0:
            self.conflict_monitoring = 0
        if self.biasing < 0:
            self.biasing = 0


            
class LinguisticKnowledge(object):
    def __init__(self, world_act=0, morphsynt_Ing_act=0, morphsynt_ed_act=0):
        # Higher-Level
        self.world_knowledge = world_act
        self.morphosyntactic_knowledge_ing = morphsynt_Ing_act
        self.morphosyntactic_knowledge_ed = morphsynt_ed_act
        # Lower-Level
        self.subj_is_agent = 0
        self.subj_is_theme = 0
        
    def input_act(self,WK,MSK_ing,MSK_ed):
        self.world_knowledge += WK
        self.morphosyntactic_knowledge_ing += MSK_ing
        self.morphosyntactic_knowledge_ed += MSK_ed

    def update(self,CogCtrl,Stroop):        
        # save initial activations for calculations (to make sure all updating happens "at once")
        world_knowledge_initial_act = self.world_knowledge
        morphosyntactic_knowledge_ing_initial_act = self.morphosyntactic_knowledge_ing
        morphosyntactic_knowledge_ed_initial_act = self.morphosyntactic_knowledge_ed
        SubjIsAgentInitialAct = self.subj_is_agent
        SubjIsThemeInitialAct = self.subj_is_theme

        # increase WK & MSK activity based on CogCtrl.biasing and their activation levels
        self.world_knowledge += (world_knowledge_initial_act*CogCtrl.biasing*biasing_mult)
        self.morphosyntactic_knowledge_ing += (morphosyntactic_knowledge_ing_initial_act*CogCtrl.biasing*biasing_mult)        
        self.morphosyntactic_knowledge_ed += (morphosyntactic_knowledge_ed_initial_act*CogCtrl.biasing*biasing_mult)        
        
        # increase SubjIsAgent & SubjIsTheme based on WK & MSK
        self.subj_is_agent += (morphosyntactic_knowledge_ing_initial_act*activation_rate) # SubjIsAgent is supported by MSK_ing
        self.subj_is_theme += (morphosyntactic_knowledge_ed_initial_act*activation_rate) # SubjIsTheme is supported by MSK_ed
        self.subj_is_theme += (world_knowledge_initial_act*activation_rate) # SubjIsTheme is supported by WN
        
        # lateral inhibition between SubjIsAgent & SubjIsTheme
        self.subj_is_agent -= (SubjIsThemeInitialAct*inhibition_rate)
        self.subj_is_theme -= (SubjIsAgentInitialAct*inhibition_rate)
        
        # decay
        self.world_knowledge -= represetations_decay_rate
        self.morphosyntactic_knowledge_ing -= represetations_decay_rate
        self.morphosyntactic_knowledge_ed -= represetations_decay_rate
        self.subj_is_agent -= represetations_decay_rate
        self.subj_is_theme -= represetations_decay_rate
        # prevent negative activation values
        if self.world_knowledge < 0 :
            self.world_knowledge = 0
        if self.morphosyntactic_knowledge_ing <0:
            self.morphosyntactic_knowledge_ing = 0
        if self.morphosyntactic_knowledge_ed <0:
            self.morphosyntactic_knowledge_ed = 0
        if self.subj_is_agent < 0:
            self.subj_is_agent = 0
        if self.subj_is_theme < 0:
            self.subj_is_theme = 0
    
    def between_trials_decay(self):
        self.world_knowledge -= represetations_decay_rate*between_trials_interval*1000
        self.morphosyntactic_knowledge_ing -= represetations_decay_rate*between_trials_interval*1000
        self.morphosyntactic_knowledge_ed -= represetations_decay_rate*between_trials_interval*1000
        self.subj_is_agent -= represetations_decay_rate*between_trials_interval*1000
        self.subj_is_theme -= represetations_decay_rate*between_trials_interval*1000
        if self.world_knowledge < 0 :
            self.world_knowledge = 0
        if self.morphosyntactic_knowledge_ing <0:
            self.morphosyntactic_knowledge_ing = 0
        if self.morphosyntactic_knowledge_ed <0:
            self.morphosyntactic_knowledge_ed = 0
        if self.subj_is_agent < 0:
            self.subj_is_agent = 0
        if self.subj_is_theme < 0:
            self.subj_is_theme = 0


        
class StroopTaskRepresentation(object):
    def __init__(self, text_blue_act=0, text_red_act=0, font_col_act=0):
        # Higher-Level
        self.text_blue = text_blue_act
        self.text_red = text_red_act
        self.font_color = font_col_act
        # Lower-Level
        self.blue = 0
        self.red = 0
            
    def input_act(self,text_blue,text_red,font_color):
        self.text_blue += text_blue
        self.text_red += text_red
        self.font_color += font_color
        

    def update(self,CogCtrl,LingKnow):
        # save initial activations for calculations (to make sure all updating happens "at once")
        text_blue_initial_act = self.text_blue
        text_red_initial_act = self.text_red
        font_color_initial_act = self.font_color
        blue_initial_act = self.blue
        red_initial_act = self.red
        
        # increase Text & FontColor activity based on CogCtrl.biasing and their activation levels
        self.text_blue += (text_blue_initial_act*CogCtrl.biasing*biasing_mult)
        self.text_red += (text_red_initial_act*CogCtrl.biasing*biasing_mult)
        self.font_color += (font_color_initial_act*CogCtrl.biasing*biasing_mult)

        # increase Blue & Red based activity based on Text & FontColor
        self.blue += (font_color_initial_act*activation_rate) # Blue is supported by font_color
        self.blue += (text_blue_initial_act*activation_rate) # Blue is supported by text_blue
        self.red += (text_red_initial_act*activation_rate) # Red is supported by text_red

        # lateral inhibition between Blue & Red
        self.blue -= (red_initial_act*inhibition_rate)
        self.red -= (blue_initial_act*inhibition_rate)
        
        # decay
        self.text_blue -= represetations_decay_rate
        self.text_red -= represetations_decay_rate
        self.font_color -= represetations_decay_rate
        self.blue -= represetations_decay_rate
        self.red -= represetations_decay_rate
        # prevent negative activation values
        if self.text_blue < 0:
            self.text_blue = 0
        if self.text_red < 0:
            self.text_red = 0
        if self.font_color <0:
            self.font_color = 0
        if self.blue < 0:
            self.blue = 0
        if self.red < 0:
            self.red = 0

    def between_trials_decay(self):
        self.text_blue -= represetations_decay_rate*between_trials_interval*1000
        self.text_red -= represetations_decay_rate*between_trials_interval*1000
        self.font_color -= represetations_decay_rate*between_trials_interval*1000
        self.blue -= represetations_decay_rate*between_trials_interval*1000
        self.red -= represetations_decay_rate*between_trials_interval*1000
        if self.text_blue < 0:
            self.text_blue = 0
        if self.text_red < 0:
            self.text_red = 0
        if self.font_color <0:
            self.font_color = 0
        if self.blue < 0:
            self.blue = 0
        if self.red < 0:
            self.red = 0

        
# Functions
def update_all(CogCtrl,LingKnow,Stroop):    
    CogCtrl_prev = copy.copy(CogCtrl)
    LingKnow_prev = copy.copy(LingKnow)
    Stroop_prev = copy.copy(Stroop)
    
    CogCtrl.update(LingKnow_prev,Stroop_prev)
    LingKnow.update(CogCtrl_prev,Stroop_prev)
    Stroop.update(CogCtrl_prev,LingKnow_prev)
    return [CogCtrl, LingKnow, Stroop]


def run_trial(input_act:tuple,params,CC=None,LK=None,S=None):
    '''
    input_act specifies the input activations for WK,MSK_ing,MSK_ed,text_blue,text_red,font_color (in that order).
    '''
    global represetations_decay_rate, cognitive_control_decay_rate, activation_rate, monitor_bias_activation_rate, inhibition_rate, biasing_mult, max_iter, activation_threshold, between_trials_interval, congruent_stroop, incongruent_stroop, congruent_sentence, anomalous_sentence
    represetations_decay_rate, cognitive_control_decay_rate, activation_rate, monitor_bias_activation_rate, inhibition_rate, biasing_mult, max_iter, activation_threshold, between_trials_interval, congruent_stroop, incongruent_stroop, congruent_sentence, anomalous_sentence = params

    i = 1
    activation_levels = {'ConfMon':[],'Biasing':[],
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
        

    LingKnow.input_act(input_act[0],input_act[1],input_act[2])
    Stroop.input_act(input_act[3],input_act[4],input_act[5])

    nodes_dict = {'ConfMon':CogCtrl.conflict_monitoring,'Biasing':CogCtrl.biasing,
                        'WK':LingKnow.world_knowledge,'MSK_ing':LingKnow.morphosyntactic_knowledge_ing,'MSK_ed':LingKnow.morphosyntactic_knowledge_ed,'SubjIsAgent':LingKnow.subj_is_agent,'SubjIsTheme':LingKnow.subj_is_theme,
                       'FontColor':Stroop.font_color,'Text_blue':Stroop.text_blue,'Text_red':Stroop.text_red,'Blue':Stroop.blue,'Red':Stroop.red}
    for key in activation_levels.keys():
        activation_levels[key].append(nodes_dict[key])
    
    # print('biasing_mult = %2f' % biasing_mult)

    while i <= max_iter:
        update_all(CogCtrl,LingKnow,Stroop)
        nodes_dict = {'ConfMon':CogCtrl.conflict_monitoring,'Biasing':CogCtrl.biasing,
                        'WK':LingKnow.world_knowledge,'MSK_ing':LingKnow.morphosyntactic_knowledge_ing,'MSK_ed':LingKnow.morphosyntactic_knowledge_ed,'SubjIsAgent':LingKnow.subj_is_agent,'SubjIsTheme':LingKnow.subj_is_theme,
                       'FontColor':Stroop.font_color,'Text_blue':Stroop.text_blue,'Text_red':Stroop.text_red,'Blue':Stroop.blue,'Red':Stroop.red}
        for key in activation_levels.keys():
            activation_levels[key].append(nodes_dict[key])
        i += 1
        max_act = max(LingKnow.subj_is_agent,LingKnow.subj_is_theme,Stroop.blue,Stroop.red)
        if max_act > activation_threshold:
            break
    
    # determine the final interpratation
    if [LingKnow.subj_is_agent,LingKnow.subj_is_theme,Stroop.blue,Stroop.red].count(max_act) > 1:
        winner = None
    if LingKnow.subj_is_agent == max_act:
        winner = 'SubjIsAgent'
    if LingKnow.subj_is_theme == max_act:
        winner = 'SubjIsTheme'
    if Stroop.blue == max_act:
        winner = 'Blue'
    if Stroop.red == max_act:
        winner = 'Red'
    
    return i, winner, CogCtrl, LingKnow, Stroop, activation_levels


def run_trial_sequence(trials:"list of tuples",params):
    '''
    trials is a list of tuples.
    Each tuple consists of 4 values that are the input activations of WK,MSK_ing,MSK_ed,text_blue,text_red,font_color (in that order) in a specific trial.
    '''
    global represetations_decay_rate, cognitive_control_decay_rate, activation_rate, monitor_bias_activation_rate, inhibition_rate, biasing_mult, max_iter, activation_threshold, between_trials_interval, congruent_stroop, incongruent_stroop, congruent_sentence, anomalous_sentence
    represetations_decay_rate, cognitive_control_decay_rate, activation_rate, monitor_bias_activation_rate, inhibition_rate, biasing_mult, max_iter, activation_threshold, between_trials_interval, congruent_stroop, incongruent_stroop, congruent_sentence, anomalous_sentence = params

    global CogCtrl, LingKnow, Stroop

    CogCtrl = CognitiveControl()
    LingKnow = LinguisticKnowledge()
    Stroop = StroopTaskRepresentation()
    
    Results = []
    for trial in trials:
        CogCtrl.between_trials_decay()
        LingKnow.between_trials_decay()
        Stroop.between_trials_decay()
        i, winner, CogCtrl, LingKnow, Stroop, activations = run_trial(trial,params,CogCtrl,LingKnow,Stroop)
        Results.append([i,winner,activations])
        
    return Results
