## Condensed Nearest Neighbors, or CNN for short, is an undersampling technique that seeks a subset of 
## a collection of samples that results in no loss in model performance, referred to as a minimal consistent set.

import pandas as pd 
import pandas
import seaborn
import matplotlib.pyplot as plt
import ImbalancedLearningRegression as iblr
from river import datasets
housing = pandas.read_csv(r"C:\Users\MAMANROY CHOUDHURY\OneDrive\Desktop\phi6.csv")
y=housing['y']

ph=iblr.phi_ctrl_pts(
    
    ## arguments / inputs
    y,                    ## response variable y
    method = "auto",      ## relevance method ("auto" or "manual")
    xtrm_type = "high",   ## distribution focus ("high", "low", "both")
    coef = 1.5,           ## coefficient for box plot
    ctrl_pts = None       ## input for "manual" rel method
    )

values=iblr.phi(
    
    ## arguments / inputs
    y,        ## reponse variable y
    ph  ## params from the 'ctrl_pts()' function
    
    )
for i in range(len(values)):
    print(values[i])
    
