import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import dataprocessor as dp

X, y = dp.label('obesity.csv')

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# t-SNE to reduce dimensions to 2
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Convert to DataFrame for easier plotting
tsne_df = pd.DataFrame(data = X_tsne, columns = ['TSNE Component 1', 'TSNE Component 2'])
tsne_df['Target'] = y

# Plotting by classes
colors = ['r', 'g', 'b', 'c']  # Define more colors if more than 4 classes
labels = np.unique(y)

fig, ax = plt.subplots()
for color, label in zip(colors, labels):
    indicesToKeep = tsne_df['Target'] == label
    ax.scatter(tsne_df.loc[indicesToKeep, 'TSNE Component 1'],
               tsne_df.loc[indicesToKeep, 'TSNE Component 2'],
               c = color, s = 20, alpha = 0.5)  # Smaller dots and 50% opacity
ax.legend(['Underweight', 'Normal', 'Overweight', 'Obese'])
ax.grid()
ax.set_xlabel('TSNE Component 1')
ax.set_ylabel('TSNE Component 2')
ax.set_title('t-SNE projection of the dataset')

plt.show()
plt.savefig('figures/tsne.png', format='png', dpi=300)
