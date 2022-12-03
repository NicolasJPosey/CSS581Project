import pandas
import numpy
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def runModels(X_train, y_train, X_test, y_test, mode):
    if (mode == 'logistic_regression'):
        model = LogisticRegression(max_iter=1000)
    elif (mode == 'naive_bayes'):
        model = GaussianNB()
    elif (mode == 'knn'):
        model = 1
    elif (mode == 'decision_tree'):
        model = 1
    elif (mode == 'random_forest'):
        model = RandomForestClassifier(max_depth=2,random_state=1)
    elif (mode == 'svm'):
        model = 1
    
    scaler=preprocessing.StandardScaler().fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    
    print("Model:",mode)
    print("Accuracy:",round(accuracy_score(y_test, y_pred),3))
    print("Precision:",round(precision_score(y_test, y_pred),3))
    print("Recall:",round(recall_score(y_test, y_pred),3))
    print("F-score:",round(f1_score(y_test, y_pred),3))
    

if __name__ == "__main__":
    #For calculating missing OOBP and OSLG
    #https://imaginesports.com/bball/reference/stats101/popup#:~:text=OOBP%20%E2%80%93%20Opponents%20On%20Base%20Percentage,OAVG%2B%20%E2%80%93%20Normalized%20Opponents%20Batting%20Average.
    dataframe = pandas.read_csv("Final_2022_1962_Dataset.csv",delimiter=',',header=0)
    dataframe.drop("RankSeason",axis=1,inplace=True)
    dataframe.drop("RankPlayoffs",axis=1,inplace=True)
    dataframe['Win_Percentage'] = dataframe.apply(lambda x: round(x['W']/x['G'],3),axis=1)
    dataframe.drop("W",axis=1,inplace=True)
    dataframe.drop("G",axis=1,inplace=True)
    dataframe.drop("Team",axis=1,inplace=True)
    dataframe.drop("League",axis=1,inplace=True)
    dataframe.drop("Year",axis=1,inplace=True)
    
    #Get 1962 to 2021 teams for training
    train = dataframe.iloc[30:]
    #Get 2022 teams for testing
    test = dataframe.iloc[:30]

    X_train=train.drop("Playoffs",axis=1)
    y_train=numpy.ravel(train['Playoffs'])
    X_test=test.drop("Playoffs",axis=1)
    y_test=numpy.ravel(test['Playoffs'])
    
    #Define the classification models that we want to try to see which has the best results
    #,'naive_bayes','knn','decision_tree','random_forest','svm'
    classificationModels = ['logistic_regression']
    
    for model in classificationModels:
        runModels(X_train, y_train, X_test, y_test, model)
    