## Tomek links are pairs of instances of opposite classes who are their own nearest neighbors. In other words,
## they are pairs of opposing instances that are very close together. Tomek's algorithm looks for such pairs 
## and removes the majority instance of the pair.
import pandas as pd 
import pandas
import seaborn
import matplotlib.pyplot as plt
import ImbalancedLearningRegression as iblr
housing = pandas.read_csv(
    "C:\MITACS_PROJECT\Bike Sharing Data.csv"
)
## conduct tomeklinks
housing_ro = iblr.tomeklinks(
    data = housing, 
    y = "cnt"
)
housing.shape
housing_ro.shape
housing_ro.reset_index(inplace=True)
iblr.box_plot_stats(housing['cnt'])['stats']
iblr.box_plot_stats(housing_ro['cnt'])['stats']
seaborn.kdeplot(housing['cnt'], label = "Original")
seaborn.kdeplot(housing_ro['cnt'], label = "Modified")
plt.legend(labels=["Original","Modified"])
plt.show()