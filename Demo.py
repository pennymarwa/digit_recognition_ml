
# coding: utf-8

# <h1> Classification of Handwritten Digits - Demo </h1>

# <h3>1. Importing Datasets and Libraries.</h3>

# In[17]:

#Importing Matplotlib.pyplot(for plotting) and Handwritten Digits Database from Scikit-Learn
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

#Assigning the datsets to a local variable
handwritten_digits = load_digits()


plt.matshow(handwritten_digits.images[0])
plt.gray()
plt.show() #8x8 pixel Image - Intensity of each cell between 0-255

#Image understood by Computer
handwritten_digits.images[0]


# <h3>2. Training the <i>handwritten_digits</i> Datasets.</h3>

# In[5]:

import random
from sklearn import ensemble

#Define length variable for number of entries in dataset
no_of_samples = len(handwritten_digits.images)
print(no_of_samples)
#x - Sample Data
x = handwritten_digits.images.reshape((no_of_samples, -1))

#y - Class labels
y = handwritten_digits.target


#Training
#Fetching random images for training 
training_index=random.sample(range(len(x)),int(len(x)/5)) #20 for training-80 for validation

#Assigning indices to training image set
training_images=[x[i] for i in training_index]

#Deciding target class[0,1,2,3,4,5,6,7,8,9] for training images
training_target=[y[i] for i in training_index]

#Declaring Classifier Object to make
classifier = ensemble.RandomForestClassifier()

#training data with Random Forest Classifier
classifier.fit(training_images, training_target)


# <h3>3. Validating the training.</h3>

# In[18]:

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


# <h3>4. Testing the model

# In[20]:

i=345
#Ignoring Warnings generated
import warnings
warnings.filterwarnings('ignore')
plt.gray() 
plt.matshow(handwritten_digits.images[i])
plt.show() 
classifier.predict(x[i])


# <img src="Thankyou.png" height=600 width=600>
