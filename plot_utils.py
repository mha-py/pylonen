import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display



class Display:
    'Für interaktive Plots, diese werden in einen ´with Display():´ Block geplottet'
    def __init__(self, doit=True):
        self.doit = doit
        
    def __enter__(self):
        if self.doit:
            clear_output(wait=True)
        
    def __exit__(self, typ, wert, traceback):
        if self.doit:
            display(plt.gcf())
            
            
            
            
################  Gewrappte plt Funktionen  ################

# ANMERKUNG: arrange_pts ist in kleinkram upgedated worden!
def _arrange_pts(pts):
    '''Puts pts of whatever format into an array of n x 2
    More intelligent version of np.array(listofitems).'''
    if type(pts) is np.ndarray:
        return pts.reshape((-1, 2))
    if type(pts) in (tuple, list):
        if type(pts[0]) is np.ndarray and len(pts[0].shape) == 2:
            return np.stack([ _arrange_pts(ps) for ps in pts ])
        else:
            return np.array(pts).reshape((-1, 2))
            
                   
def plot(pts, *args, **kwargs):
    #pts = np.array(pts)
    pts = _arrange_pts(pts)
    plt.plot(pts[:,0], pts[:,1], *args, **kwargs)
            
            
def scatter(pts, *args, **kwargs):
    pts = _arrange_pts(pts)
    plt.scatter(pts[:,0], pts[:,1], *args, **kwargs)
    

################  Verschiedenes  ################
    
def prepare_plt(w, h, dpi=72):
    '''Setzt Voreinstellungen für plt, sodass ein Bild mit Höhe und Breite gespeichert werden kann.
    Kann durch den normalen Befehl plt.savefig(fname) gespeichert werden'''
    plt.figure(figsize=(w/dpi, h/dpi))
    plt.axis('off')
    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(left = 0)
    fig.subplots_adjust(right = 1)