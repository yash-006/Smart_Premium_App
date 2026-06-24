import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../data/raw/insurance.csv")

# Basic info
print(data.info())
print(data.describe())

# Missing values
print(data.isnull().sum())

# Distribution
sns.histplot(data["Premium Amount"])
plt.show()

# Correlation
sns.heatmap(data.corr(), annot=True)
plt.show()