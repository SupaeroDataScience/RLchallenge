#import matplotlib.pyplot as plt
from skimage import transform,color

buffersize = 2
bufferTab = [None]*2
i = 0

def FlappyPolicy(state, screen):
    global bufferTab
    global i
    global buffersize

    #downsize = 80
    #screen = transform.resize(color.rgb2gray(screen[:,:404,:]),(downsize,downsize)) # Crop at 404 px 


    if(state['player_y']+60 > state['next_pipe_bottom_y']):
        bufferTab[i] = 119
    else:
        bufferTab[i] = None

    i=(i+1)%buffersize
    return bufferTab[i]
    
    #return None # Should return an action
