import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import itertools
from scipy import linalg
from sklearn.mixture import GMM
from scipy import stats


color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange','yellow', 'grey'])




####input data
df_data = pd.read_csv("pulsar.csv" ,delim_whitespace=True, header=0, skiprows = None, usecols = ["P0", "P1"])

####dropping the rowswhich have no values or nan as values
df_data =  df_data.dropna(axis = 0)
df_data_no_null = df_data.drop(df_data[(df_data.P0 == 0) | (df_data.P1 <= 0)].index)
df_data1 = df_data.drop(df_data[(df_data.P0 == 0) | (df_data.P1 <= 0)].index)


#####Finding the interquartile range for the input dataset and tried to remove 
#####outliers by removing the ones outside the IQR or raw dataset which is taken
#####as input. DIdn't work as the range is still large.
Q1 = df_data1.quantile(0.25)
m = df_data1.median()
Q3 = df_data1.quantile(0.75)
IQR = Q3 - Q1
#df_data1 = df_data1[~((df_data1 < (Q1 - 1.5 * IQR)) |(df_data1 > (Q3 + 1.5 * IQR))).any(axis=1)]


####Hence to decrease the range, took log of each colums of input data 
####And removed all the rows in which either of the columns crossed the IQR +- 1.5 range.Since 
####rest of the calculations is done in log P and log p_dot, this is the best way to remove outliers, 
####i.e. we have to remove outliers in log p not in p, as that would be effecient
#### in removing outliers from the data we ues further
df_data_log = df_data1.apply(np.log10)
df_data1 = df_data1.apply(np.log10)
Q1 = df_data1.quantile(0.25)
m = df_data1.median()
Q3 = df_data1.quantile(0.75)
IQR = Q3 - Q1

df_data1 = df_data1[~((df_data1 < (Q1 - 1.5 * IQR)) |(df_data1 > (Q3 + 1.5 * IQR))).any(axis=1)]



####tried removing outliers with z-scores, didnt work for p, might work for logp
#df_data1 = df_data1[(np.abs(stats.zscore(df_data1)) < 2.5).all(axis=1)]


####converting log data into array
data_log_arr = df_data1.as_matrix()

####this would be useful for single variable analysis
P = data_log_arr[:,0]
P = P.reshape(-1,1)


####Plotting AIC/BIC for log P and log p_dot dataset
N = np.arange(1, 11)
models = [None for i in range(len(N))]

for i in range(len(N)):
    models[i] = GMM(N[i]).fit(data_log_arr)

AIC = [m.aic(data_log_arr) for m in models]
BIC = [m.bic(data_log_arr) for m in models]

print(AIC)
print(BIC)

plt.plot(N, AIC, '-b', label='AIC',linewidth = 1.5)
plt.plot(N, BIC, '--g', label='BIC',linewidth = 1.5)
plt.xlabel('Number of components', fontsize=16)
plt.ylabel('Information criterion', fontsize=16)
plt.title("AIC/BIC v/s No. of Components for 2 features")
plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(10,7)
plt.savefig("AIC_P2d.png", bbox_inches='tight')

plt.show()












def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = matplotlib.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

   

    plt.title(title)
    fig = plt.gcf()
    fig.set_size_inches(10,7)

'''
plt.scatter(data_log_arr[:,0], data_log_arr[:,1], s = 0.2)

fig = plt.gcf()
fig.set_size_inches(10,7)
plt.savefig("2d_data.png")

plt.show()
'''

# Fit a Gaussian mixture with EM using 3 components
gmm = sklearn.mixture.GaussianMixture(n_components=3, covariance_type='full').fit(data_log_arr)
plot_results(data_log_arr, gmm.predict(data_log_arr), gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture')
#fig = plt.gcf()
#fig.set_size_inches(10,7)
plt.savefig("P2d.png")

plt.show()





######SINGLE VARIABLE AIC/BIC PLOT STARTS HERE
N = np.arange(1, 11)
models = [None for i in range(len(N))]

for i in range(len(N)):
    models[i] = GMM(N[i]).fit(data_log_arr[:,0].reshape(-1,1))

AIC = [m.aic(data_log_arr[:,0].reshape(-1,1)) for m in models]
BIC = [m.bic(data_log_arr[:,0].reshape(-1,1)) for m in models]

print(AIC)
print(BIC)

plt.plot(N, AIC, '-b', label='AIC',linewidth = 1.5)
plt.plot(N, BIC, '--g', label='BIC',linewidth = 1.5)
plt.xlabel('Number of components', fontsize=16)
plt.ylabel('Information criterion', fontsize=16)
plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.title("AIC/BIC v/s No. of Components for log P ")
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(10,7)
plt.savefig("AIC_P2d.png", bbox_inches='tight')

plt.show()


gmm_single = sklearn.mixture.GaussianMixture(n_components=3, covariance_type='full').fit(data_log_arr[:,0].reshape(-1,1))

gmm_x = np.linspace(-1.5, 1.0, 5000)
gmm_y = np.exp(gmm_single.score_samples(gmm_x.reshape(-1, 1)))

plt.hist(data_log_arr[:,0], 32, facecolor='g', alpha=0.9,normed = True)
plt.plot(gmm_x, gmm_y, color="black", lw=4, label="GMM")
fig = plt.gcf()
fig.set_size_inches(10,7)
plt.savefig("1d_data_hist.png")

plt.show()


plt.scatter(data_log_arr[:,0], data_log_arr[:,1], s = 0.2)

fig = plt.gcf()
fig.set_size_inches(10,7)
plt.savefig("outlier_rem_data.png")

plt.show()

plt.scatter(df_data_log['P0'], df_data_log['P1'], s = 0.2)

fig = plt.gcf()
fig.set_size_inches(10,7)
plt.savefig("raw_data_log.png")

plt.show()

plt.scatter(df_data_no_null['P0'], df_data_no_null['P1'], s = 0.2)

fig = plt.gcf()
fig.set_size_inches(10,7)
plt.savefig("raw_data.png")

plt.show()