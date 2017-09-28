'''
Description:
    This dataset contains transactions made by credit cards in September 2013 by european cardholders.
    This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.
    The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
    Using that dataset applied the several approaches to handle the imbalanced classes using resampling modules.
'''


# Importing the Required libraries

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# Reading the dataset by using pandas
df = pd.read_csv('creditcard.csv')
# Displaying the example observations from the dataset 
df.head()

# Finding the Imbalance based on Dependent variable
df['Class'].value_counts()

############################################# SAMPLING TECHNIQUES #############################################

#SAMPLING --->>>> 1. The sampling methods aim to modify an imbalanced data into balanced distribution using some mechanism.
#                 2. The modification occurs by altering the size of original data set and provide the same proportion of balance.

#The methods  are used to treat imbalanced datasets:
#1. Oversampling
#2. Undersampling
#3. Synthetic Data Generation

# Executor Function 
def sampling_exector():
    print("########### Sampling Process Started ###########")
    print("***** up-sampling techniques *****")
    up_sampling_process()
    print("*****down-sampling techniques *****")
    down_sampling_process()
    print("*****SMOTE-sampling techniques *****")
    smote_sampling_process()
    print("########### Process finished ###########")

# Up-sample Minority Class(Oversampling)
def up_sampling_process():

    #   Up-sampling ->> 1.The "upsampling" method works with minority class.
    #                   2.It replicates the observations from minority class to balance the data.
    #                   3.This method is also known as Oversampling technique.


    # Separating the  majority and minority classes
    df_majority = df[df.Class==0]
    df_minority = df[df.Class==1]

    # Upsample minority class 
    df_minority_upsampled = resample(df_minority,
                                     replace=True,     # sample with replacement
                                     n_samples=len(df_majority), # to match majority class
                                     random_state=123) # reproducible results

    # Combine majority class with upsampled minority class 
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    # Display new class counts
    df_upsampled.Class.value_counts()

    # Separating the  input features (X) and target variable (y) 
    y = df_upsampled.Class
    X = df_upsampled.drop('Class', axis=1)

    # Applying feature scaling 
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    # Training the  model 
    clf_1 = LogisticRegression().fit(X, y)

    # Predict on training set 
    pred_y_1 = clf_1.predict(X)

    # checking our model still predicting based on one class 
    print( np.unique( pred_y_1 ) )

    # finding the accuracy value of this model 
    print( accuracy_score(y, pred_y_1) )


# Down-sample Majority Class(Undersampling) 
def down_sampling_process():

    #  Down-sampling -->> 1.The "Down-sampling" method works with majority class.
    #                     2.It reduces the number of observations from majority class to make the data set balanced.
    #                     3.This method is best to use when the data set is huge and reducing the number of training samples helps to improve run time and storage troubles.
    #                     4.It is also known as Undersampling technique.

    # Separating the majority and minority classes 
    df_majority = df[df.Class==0]
    df_minority = df[df.Class==1]

    # Downsample majority class 
    df_majority_downsampled = resample(df_majority,
                                     replace=False,    # sample without replacement
                                     n_samples=len(df_minority),    # to match minority class
                                     random_state=123) # reproducible results

    # Combine minority class with downsampled majority class 
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    # Display new class counts 
    df_downsampled.Class.value_counts()

    # Separating the  input features (X) and target variable (y) 
    y = df_downsampled.Class
    X = df_downsampled.drop('Class', axis=1)

    # Applying feature scaling
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    # Training the  model 
    clf_2 = LogisticRegression().fit(X, y)

    # Predict on training set 
    pred_y_2 = clf_2.predict(X)

    # checking our model still predicting based on one class 
    print( np.unique( pred_y_2 ) )

    # finding the accuracy value of this model
    print( accuracy_score(y, pred_y_2) )


#  SMOTE(Synthetic Minority Over-sampling Technique) Sampling 
def smote_sampling_process():

    # SMOTE-sampling -->> 1.There are systematic algorithms that you can use to generate synthetic samples.
    #                     2.The most popular of such algorithms is called SMOTE or the Synthetic Minority Over-sampling Technique was introduced that try to address the class #                         imbalance problem.
    #                     3.It is one of the most adopted approaches due to its simplicity and effectiveness.
    #                     4.It is a combination of Oversampling and Undersampling, but the oversampling approach is not by replicating minority class but constructing new                               minority class data instance via an algorithm.



    # Separating the  input features (X) and target variable (y) 
    y_target = df.Class
    X_features = df.drop('Class', axis=1)
    # feature Scaling 
    sc_X = StandardScaler()
    X_features = sc_X.fit_transform(X_features)
    # splitting the data into training set,test set and validation sets using train_test_split 
    training_features, test_features,training_target, test_target, = train_test_split(X_features,
                                                   y_target,
                                                   test_size = .2,
                                                   random_state=123)
    x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
                                                      test_size = .2,
                                                      random_state=123)

    # Using SMOTE, oversampling the training data 
    sm = SMOTE(random_state=123, ratio = .2)
    x_train_res, y_train_res = sm.fit_sample(x_train, y_train)

    # Training the LogisticRegression model for SMOTE sampling
    clf_rf =LogisticRegression()
    clf_rf.fit(x_train_res, y_train_res)
    print('Validation Results')
    print(clf_rf.score(x_val, y_val))
    print(recall_score(y_val, clf_rf.predict(x_val)))
    print('\nTest Results')
    print(clf_rf.score(test_features, test_target))
    print(recall_score(test_target, clf_rf.predict(test_features)))

    # Evaluating the performance of this model using AUROC metrics 
    clf_rf = clf_rf.predict_proba(test_features)
    clf_rf = [p[1] for p in clf_rf]
    print("\nAccuracy value")
    print(roc_auc_score(test_target, clf_rf) )




# Calling the sampling executor function 
sampling_exector()


##Conclusion:
    #From the above sampling techniques the SMOTE sampling method gives the best result for our dataset and
    #It gives the 98% of accuracy value ,so which is the best sampling model for our dataset.
