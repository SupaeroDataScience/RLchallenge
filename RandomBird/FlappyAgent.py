import numpy as np

file_array="C:/Users/asus/Desktop/kierszbaum/RLchallenge/RandomBird/Q.npy"
count_map=np.load(file_array)

step=0
lmemory=[None]*6
memory=np.array([None]*6)
action_seq=[119,119,None]
prev_state=0
two_steps_previous_action=[None,None]
one_step_previous_column=12

# idea, si la prochaine pipe est au dessus de lactuelle, le zero est place plus bas
# sinon, il est place plus haut
def FlappyPolicy(state, screen):
    
   
    global count_map
    global step

    index_state=0
    step+=1
    # verifier que letat na pas deja ete rencontre dans count_map:
    bool1=(state not in count_map[:,0])
    if bool1:
        action=default_policy(state)
    if not bool1:
        index_state=count_map[:,0].tolist().index(state)
        
        if count_map[index_state,1]==-1:
            action=119
            
        if count_map[index_state,2]==-1:
            
            action=None
            
            
        if (count_map[index_state,1]==-1) and (count_map[index_state,2]==-1):
            print('pas ok')
    return action
    


def default_policy(state):
    global memory
    global action_seq
    global prev_state
    global step
    r=True
    rr=False
    global two_steps_previous_action
    global one_step_previous_column

    state_simplif,statelist=statedefine(state)
    
    
    #if one_step_previous_column!=statelist[1]:
        #print("tuyzuuu passseey")
    
    rr=(one_step_previous_column!=statelist[1])
    one_step_previous_column=statelist[1]
    single_action=None
    if prev_state!=state_simplif:
        memory=np.array([None,None,None,None,None,None])
        step=0
    #define the state
    if step%2==0:
        # sequence action pour remonter
        if state_simplif==-1:
            action_seq=[None, 119]
    # sequence daction pour descendre    
        if state_simplif==1:
            action_seq=[None, None]
         # define the action:   
        single_action=action_seq[0]
    if step%2==1:
        single_action=action_seq[1]
        
     # sequence daction pour aller tout droit
    if (state_simplif==0) and (memory==np.array([None,119,119,None,None,None])).all():
        single_action=None
        if r==True:
            memory=np.array([119,119,None,None,None,None])
            r=False
        else :
            r=True
            memory==np.array([None,None,None,None,None,None])
            #single action = 119
    
    if (prev_state==0 and state_simplif==1 and two_steps_previous_action==[None,119]):
        state_simplif=0
        memory=np.array([None,None,None,None,None,119])
        r=True
            
    if (state_simplif==0) and (memory==np.array([None,None,119,119,None,None])).all():
        memory=memory=np.array([None,119,119,None,None,None])
        single_action=None 
    if (state_simplif==0) and (memory==np.array([None,None,None,119,119,None])).all():
        memory=np.array([None,None,119,119,None,None])
        single_action=None
    if (state_simplif==0) and (memory==np.array([None,None,None,None,119,119])).all():
        memory=np.array([None,None,None,119,119,None])
        single_action=None
    if (state_simplif==0) and (memory==np.array([None,None,None,None,None,119])).all():
        memory=np.array([None,None,None,None,119,119])
        single_action=119
    if (state_simplif==0) and (memory==np.array([None,None,None,None,None,None])).all():
        memory=np.array([None,None,None,None,None,119])
        single_action=119 
        
    if (memory==np.array([119,119,None,None,None,None])).all():
        memory=np.array([None,None,None,None,None,None])
        
     
    if (two_steps_previous_action==[119,119]):
        single_action=None
        memory=np.array([None,None,None,119,119,None])
    if prev_state-state_simplif==-1 and (two_steps_previous_action[1]==119) :
        single_action==119
        memory==np.array([None,None,None,None,119,119])
        r=True
    two_steps_previous_action=[two_steps_previous_action[1]]+[single_action]
    
    prev_state=state_simplif
    # probleme : se heurte au bord des tuyaux
    # capturer le changement detat: et faire comme s'il n'existait pas pendant 2 tours;
        
    if rr:
        single_action=state0(step)
    #print(single_action)
    return single_action



# fonction qui pendant 5 steps apres le passage dun tuyau force le state simplif a 0 dans la continuite
# prend step en entree
def state0(memory):
    if (memory==np.array([None,None,119,119,None,None])).all():
        memory=memory=np.array([None,119,119,None,None,None])
        single_action=None 
    if (memory==np.array([None,None,None,119,119,None])).all():
        memory=np.array([None,None,119,119,None,None])
        single_action=None
    if (memory==np.array([None,None,None,None,119,119])).all():
        memory=np.array([None,None,None,119,119,None])
        single_action=None
    if (memory==np.array([None,None,None,None,None,119])).all():
        memory=np.array([None,None,None,None,119,119])
        single_action=119
    if (memory==np.array([None,None,None,None,None,None])).all():
        memory=np.array([None,None,None,None,None,119])
        single_action=119 
    




def statedefine(state):
    statelist=[]
    statereduced=[]
    for j in state.items():
        statelist.append(j[1])
    statereduced=[statelist[0]]+statelist[3:5]
    state_simplif=0
    # axe des y vers le bas
    if statereduced[0]>(statereduced[2]-40):
        state_simplif=-1# au dessous
    if statereduced[0]<(statereduced[1]+40):
        state_simplif=1#au dessus
     # prochains tuyaux plus hauts-> se mettre le plus en bas possible
    if statelist[-1]<statelist[4]:
        if statereduced[0]>(statereduced[2]-25):
            state_simplif=-1# au dessous
        if statereduced[0]<(statereduced[1]+55):
            state_simplif=1#au dessus
    # prochains tuyaux plus bas-> se mettre le plus en haut possible
    if statelist[-1]>statelist[4]:
        if statereduced[0]>(statereduced[2]-55):
            state_simplif=-1# au dessous
        if statereduced[0]<(statereduced[1]+25):
            state_simplif=1#au dessus        
    return (state_simplif,statereduced)