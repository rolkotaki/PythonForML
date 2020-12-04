from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import pandas as pd


boston = load_boston()
print(type(boston))
print(boston.keys())
print(type(boston['data']))

clf = RandomForestRegressor()
clf.fit(boston['data'], boston['target'])  # training the model

print(clf.score(boston['data'], boston['target']))

print(clf.n_features_)
print(boston['data'].shape)

row = boston['data'][17]

print(clf.predict(row.reshape(-1, 13)))  # 18.0001
print(boston['target'][17])              # 17.5

########################################################################################################################
# splitting data to training and testing

x_train, x_test, y_train, y_test = train_test_split(boston['data'], boston['target'], test_size=0.3)
clf = RandomForestRegressor()
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))

########################################################################################################################
# scaling

df = pd.DataFrame(boston['data'], columns=boston['feature_names'])
print(df.max(axis=0))

clf = SVR()
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))

Xs = preprocessing.scale(boston['data'])

df = pd.DataFrame(Xs, columns=boston['feature_names'])
print(df.max(axis=0))

Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, boston['target'], test_size=0.3)
clf = SVR()
clf.fit(Xs_train, ys_train)
print(clf.score(Xs_test, ys_test))

########################################################################################################################
# PCA to reduce number of features

pca = PCA(n_components=5)  # we would like 5 features, components
pca.fit(boston['data'])

Xp = pca.transform(boston['data'])
print(Xp.shape)  # we have 5 features

clf = RandomForestRegressor()
Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, boston['target'], test_size=0.3)
clf = SVR()
clf.fit(Xp_train, yp_train)
print(clf.score(Xp_test, yp_test))

########################################################################################################################
# using pipeline to preprocess

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=5)),
    ('svr', SVR())
])

pipe.fit(x_train, y_train)
print(pipe.score(x_test, y_test))

pipe.get_params()
pipe.set_params(svr__C=0.9)  # updating a step in the pipe

########################################################################################################################
# saving and loading model

with open('data/model.pickle', 'wb') as out:
    pickle.dump(pipe, out)

with  open('data/model.pickle', 'rb') as fp:
    pipe1 = pickle.load(fp)

print(pipe1.steps)
print(pipe.score(x_test, y_test))

########################################################################################################################
