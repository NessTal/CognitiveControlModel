import copy


# Classes
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

        
# Functions
def UpdateAll(CogCtrl,LingKnow,Stroop):    
    CogCtrl_prev = copy.copy(CogCtrl)
    LingKnow_prev = copy.copy(LingKnow)
    Stroop_prev = copy.copy(Stroop)
    
    CogCtrl.Update(LingKnow_prev,Stroop_prev)
    LingKnow.Update(CogCtrl_prev,Stroop_prev)
    Stroop.Update(CogCtrl_prev,LingKnow_prev)
    return [CogCtrl, LingKnow, Stroop]


def RunTrial(InputAct:tuple,params,CC=None,LK=None,S=None):
    '''
    InputAct specifies the input activations for WK,MSK_ing,MSK_ed,Text_blue,Text_red,FontColor (in that order).
    '''
    global RepresetationsDecayRate, CognitiveControlDecayRate, ActivationRate, MonitorBiasActivationRate, InhibitionRate, BiasingMult, MaxIter, ActivationThreshold, BetweenTrialsInterval, CongruentStroop, IncongruentStroop, CongruentSentence, AnomalousSentence
    RepresetationsDecayRate, CognitiveControlDecayRate, ActivationRate, MonitorBiasActivationRate, InhibitionRate, BiasingMult, MaxIter, ActivationThreshold, BetweenTrialsInterval, CongruentStroop, IncongruentStroop, CongruentSentence, AnomalousSentence = params

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


def RunTrialSequence(Trials:"list of tuples",params):
    '''
    Trials is a list of tuples.
    Each tuple consists of 4 values that are the input activations of WK,MSK_ing,MSK_ed,Text_blue,Text_red,FontColor (in that order) in a specific trial.
    '''
    global RepresetationsDecayRate, CognitiveControlDecayRate, ActivationRate, MonitorBiasActivationRate, InhibitionRate, BiasingMult, MaxIter, ActivationThreshold, BetweenTrialsInterval, CongruentStroop, IncongruentStroop, CongruentSentence, AnomalousSentence
    RepresetationsDecayRate, CognitiveControlDecayRate, ActivationRate, MonitorBiasActivationRate, InhibitionRate, BiasingMult, MaxIter, ActivationThreshold, BetweenTrialsInterval, CongruentStroop, IncongruentStroop, CongruentSentence, AnomalousSentence = params

    global CogCtrl, LingKnow, Stroop

    CogCtrl = CognitiveControl()
    LingKnow = LinguisticKnowledge()
    Stroop = StroopTaskRepresentation()
    
    Results = []
    for Trial in Trials:
        CogCtrl.BetweenTrialsDecay()
        LingKnow.BetweenTrialsDecay()
        Stroop.BetweenTrialsDecay()
        i, Winner, CogCtrl, LingKnow, Stroop, Activations = RunTrial(Trial,params,CogCtrl,LingKnow,Stroop)
        Results.append([i,Winner,Activations])
        
    return Results
