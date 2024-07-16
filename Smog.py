# importing the libraries
import seaborn as sns
import smogn
import pandas as pd 
import pandas
import matplotlib.pyplot as plt
housing = pandas.read_csv(
    "C:\MITACS_PROJECT\Housing.csv"
)
# plotting the points 
plt.plot(housing.area,housing.price) 
# naming the x axis
plt.xlabel('area')
# naming the y axis
plt.ylabel('price')
plt.show()
cleaned = smogn.smoter(data=housing,y="price")
print(cleaned)
# plotting the cleaned points 
plt.plot(cleaned.area,cleaned.price) 
# naming the x axis
plt.xlabel('cleaned area')
# naming the y axis
plt.ylabel('cleaned price')
plt.show()