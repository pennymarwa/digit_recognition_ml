#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:11:59 2017

@author: marwa
"""

import matplotlib.pyplot as plt
from sklearn import datasets
import random
from sklearn import ensemble
import numpy as np
import warnings
warnings.filterwarnings("ignore")



def pred_ints(model, X, percentile=95):
    err_down = []
    err_up = []
    for x in range(len(X)):
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict(X[x])[0])
        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
    return err_down, err_up



if __name__ == "__main__":

    digits = datasets.load_digits()
    
    #Declaring Classifier Object to make
    classifier = ensemble.RandomForestClassifier()
    
    i=0
    
    
    #print(len(digits.data))
    
    a,b = digits.data[:-2], digits.target[:-2]
    
    classifier.fit(a,b)
    
    
    plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation = "nearest")
    plt.show()
    
    print('Prediction-',classifier.predict(digits.data[i]))
    
    print('Digit Matrix', digits.images[i])
    
    print ('___________________________________________________________ \n')
    
    
    images_and_labels = list(zip(digits.images, digits.target))
    plt.figure(figsize=(5,5))
    for index, (image, label) in enumerate(images_and_labels[:15]):
        plt.subplot(5, 5, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('%i' % label)
        
        
    print ('___________________________________________________________ \n')
    
    
    
    #Define length variable for number of entries in dataset
    no_of_samples = len(digits.images)
    print('Total digits', no_of_samples)
    #x - Sample Data
    x = digits.images.reshape((no_of_samples, -1))
    
    #y - Class labels
    y = digits.target
    
    
    #Training
    #Fetching random images for training 
    training_index=random.sample(range(len(x)),int(len(x)/5)) #20 for training-80 for validation
    
    #Assigning indices to training image set
    training_images=[x[i] for i in training_index]
    
    #Deciding target class[0,1,2,3,4,5,6,7,8,9] for training images
    training_target=[y[i] for i in training_index]
    
    
    
    #training data with Random Forest Classifier
    classifier.fit(training_images, training_target)
    
    #Validating
    #Fetching random images for validation 
    validation_index=[i for i in range(len(x)) if i not in training_index]
    
    #Assigning indices to validation image set
    validation_images=[x[i] for i in validation_index]
    
    #Deciding target class[0,1,2,3,4,5,6,7,8,9] for Validation images
    validation_target=[y[i] for i in validation_index]
    
    #Predicting score for validation data - Returns the mean accuracy on the given test data and labels
    score=classifier.score(validation_images, validation_target)
    print ('Prediction Score by Random Tree Classifier:\t'+str(round(score*100,2))+'%')
    
    from sklearn import tree
    i_tree = 0
    for tree_in_forest in classifier.estimators_:
        with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
            my_file = tree.export_graphviz(tree_in_forest, out_file = my_file)
        i_tree = i_tree + 1
        
        
        
importance = classifier.feature_importances_
importance = pd.DataFrame(importance, index=training_images.columns, 
                          columns=["Importance"])

importance["Std"] = np.std([tree.feature_importances_
                            for tree in clf.estimators_], axis=0)

x = range(importance.shape[0])
y = importance.ix[:, 0]
yerr = importance.ix[:, 1]

plt.bar(x, y, yerr=yerr, align="center")

plt.show()       
        
#    err_down, err_up = pred_ints(classifier, training_images, percentile=90)
# 
#    truth = training_target
#    correct = 0.
#    for i, val in enumerate(truth):
#        if err_down[i] <= val <= err_up[i]:
#            correct += 1
#    print (correct/len(truth))
#    
    


#getTree(classifier, k=1, labelVar=FALSE)


#y_pred=classifier.predict(digits.data[i])
#confusion_mat=confusion_matrix(training_index,y_pred)
#print(confusion_mat)



#ensemble.export_graphviz(classifier,out_file='Dtree.dot')


#X, y = make_blobs(n_samples=10000, n_features=10, centers=100,
#    random_state=0)
#
#clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
#    random_state=0)
#scores = cross_val_score(clf, X, y)
#scores.mean()                             
#
#
#clf = RandomForestClassifier(n_estimators=10, max_depth=None,
#    min_samples_split=2, random_state=0)
#scores = cross_val_score(clf, X, y)
#scores.mean()                             
#
#
#clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
#    min_samples_split=2, random_state=0)
#scores = cross_val_score(clf, X, y)
#scores.mean() > 0.999
           
#print('Complete data', digits.data)
#print('Target data', digits.target)
#print('individual digit image', digits.images[0])