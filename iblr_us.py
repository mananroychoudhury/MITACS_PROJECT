## Random undersampling involves randomly selecting examples from the majority class and deleting them from
## the training dataset. In the random under-sampling, the majority class instances are discarded at random 
## until a more balanced distribution is reached.
import pandas as pd 
import pandas
import seaborn
import matplotlib.pyplot as plt
import ImbalancedLearningRegression as iblr
housing = pandas.read_csv(
    "C:\MITACS_PROJECT\Bike Sharing Data.csv"
)
## conduct random undersampling
housing_ro = iblr.random_under(
    data = housing, 
    y = "cnt"
)
housing.shape
housing_ro.shape
iblr.box_plot_stats(housing['cnt'])['stats']
iblr.box_plot_stats(housing_ro['cnt'])['stats']
seaborn.kdeplot(housing['cnt'], label = "Original")
seaborn.kdeplot(housing_ro['cnt'], label = "Modified")
plt.legend(labels=["Original","Modified"])
plt.show()