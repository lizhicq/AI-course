import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts

train = pd.read_csv('./data/train_1.csv')
y = np.asarray(train['SalePrice'])
train1 = train.drop(['Id','SalePrice'],axis=1)
X = np.asarray(pd.get_dummies(train1).reset_index(drop=True))
X_train, X_test, y_train, y_test = tts(X,y,test_size=0.2)


def benchmark(model):
    pred=model.predict(X_test)
    logrmse=np.sqrt(mean_squared_error(np.log(y_test),np.log(pred)))
    return logrmse