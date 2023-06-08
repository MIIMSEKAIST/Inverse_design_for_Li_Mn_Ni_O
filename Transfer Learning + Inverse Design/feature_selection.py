import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_csv("impute_Li_NCM_MICE.csv")
X = data.loc[:,'Li':'c_rate']  #independent columns
y = data['Dis_cap']    #target column i.e price range
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y.ravel())
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(len(X)).plot(kind='barh')
plt.show()
print(feat_importances)
#feat_importances.to_excel("feature selection.xlsx", sheet_name='Sheet_name_1')