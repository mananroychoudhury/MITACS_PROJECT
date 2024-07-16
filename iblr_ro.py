## Random oversampling involves randomly duplicating examples from the minority class and adding them to the training dataset.
import pandas as pd 
import pandas
import seaborn
import matplotlib.pyplot as plt
import ImbalancedLearningRegression as iblr
housing = pandas.read_csv(
    "C:\MITACS_PROJECT\Bike Sharing Data.csv"
)
## conduct Random Over-sampling
housing_ro = iblr.ro(
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