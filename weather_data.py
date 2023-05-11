import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from scipy import stats

# Set the background color of all plots to black
plt.style.use('dark_background')

# Read the CSV file
df = pd.read_csv('Weather Data.csv')


# Normalize the data
scaler = StandardScaler()
df_norm = scaler.fit_transform(df[['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']])

# Perform k-means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_norm)
labels = kmeans.labels_

# Add cluster labels to the dataframe
df['Cluster'] = labels

# Print mean and describe for each cluster
for i in range(2):
    cluster_df = df[df['Cluster']==i]
    print("Cluster", i)
    print(cluster_df.mean())
    print(cluster_df.describe())

# Create a scatter plot matrix of the data with cluster labels
pd.plotting.scatter_matrix(df.drop('Cluster', axis=1), figsize=(10,10), c=labels, cmap='viridis')
plt.show()

# Create a bar chart of the cluster sizes
fig, ax = plt.subplots(figsize=(10,5))
plt.bar([0,1,2], df['Cluster'].value_counts().sort_index())
plt.xticks([0,1,2], ['Cluster 0', 'Cluster 1', 'Cluster 2'])
plt.ylabel('Count')
plt.title('Cluster Sizes')
plt.show()

# Create a scatter plot of temperature vs. pressure with cluster labels
fig, ax = plt.subplots(figsize=(10,5))
for i in range(2):
    cluster_df = df[df['Cluster']==i]
    plt.scatter(cluster_df['Press_kPa'], cluster_df['Temp_C'], label='Cluster '+str(i))
plt.xlabel('Pressure (kPa)')
plt.ylabel('Temperature (C)')
plt.title('Temperature vs. Pressure')
plt.legend()
plt.show()

# Create a scatter plot of temperature vs. wind speed with cluster labels
fig, ax = plt.subplots(figsize=(10,5))
for i in range(2):
    cluster_df = df[df['Cluster']==i]
    plt.scatter(cluster_df['Wind Speed_km/h'], cluster_df['Temp_C'], label='Cluster '+str(i))
plt.xlabel('Wind Speed (km/h)')
plt.ylabel('Temperature (C)')
plt.title('Temperature vs. Wind Speed')
plt.legend()
plt.show()

# Create a histogram of the temperature data for each cluster
for i in range(2):
    cluster_df = df[df['Cluster']==i]
    fig, ax = plt.subplots(figsize=(10,5))
    plt.hist(cluster_df['Temp_C'], bins=20, color='blue', edgecolor='black')
    plt.xlabel('Temperature (C)')
    plt.ylabel('Frequency')
    plt.title('Temperature Distribution for Cluster ' + str(i))
    plt.show()


# Define the exponential growth model
def exp_model(x, a, b, c):
    return a * np.exp(b * x) + c

# Extract the temperature data
xdata = np.arange(len(df))
ydata = df['Temp_C']

# Fit the model to the data
popt, pcov = curve_fit(exp_model, xdata, ydata, maxfev=5000)

# Generate predictions for the next 20 years
xpred = np.arange(len(df), len(df)+20)
ypred = exp_model(xpred, *popt)

# Estimate confidence ranges using the err_ranges function
def err_ranges(popt, pcov, xdata, conf=0.95):
    perr = np.sqrt(np.diag(pcov))
    tval = stats.t.ppf(1-conf/2, len(xdata)-len(popt))
    err = tval * perr
    return err

conf_range = err_ranges(popt, pcov, xdata)

# Plot the data, the best fit function, and the confidence range
fig, ax = plt.subplots(figsize=(10,5))
plt.scatter(xdata, ydata, c='b')
plt.plot(xpred, ypred, 'r')
plt.fill_between(xpred, ypred-conf_range, ypred+conf_range, alpha=0.2, color='r')
plt.xlabel('Time')
plt.ylabel('Temperature (C)')
plt.title('Exponential Growth Model of Temperature Data')
plt.show()
