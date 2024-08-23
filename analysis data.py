import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
name_fig = 'Analysis_dataset_rainseason.png'
file_train = './Data/dataset_rainseason_train_80.csv'
df = pd.read_csv(file_train)
plt.figure(figsize=(18,5))
hm = sns.heatmap(df.corr(), annot=True, lw = 1, linecolor="r",cmap="coolwarm")
plt.savefig(name_fig)
plt.show()
