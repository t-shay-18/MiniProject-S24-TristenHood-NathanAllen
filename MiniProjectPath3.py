#MiniProjectPath3
import numpy as np
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
#import models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import copy
from sklearn.decomposition import KernelPCA


rng = np.random.RandomState(1)
digits = datasets.load_digits()
images = digits.images
labels = digits.target

#Get our training data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.6, shuffle=False)

def dataset_searcher(number_list,images,labels):
  #insert code that when given a list of integers, will find the labels and images
  #and put them all in numpy arrary (at the same time, as training and testing data)
  
  length = len(number_list)

  images_nparray = np.zeros((length, 8, 8))
  labels_nparray = np.zeros((length))
  for i, j in enumerate(number_list):
  
    images_nparray[i] = images[j]
    labels_nparray[i] = labels[j]
    
    
  
  return images_nparray, labels_nparray



def print_numbers(images,labels):
  #insert code that when given images and labels (of numpy arrays)
  #the code will plot the images and their labels in the title. 
  fig, axes = plt.subplots(1, len(images), figsize=(8, 8))
  for ax, image, label in zip(axes, images, labels):
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('%i' % label)
    ax.axis('off')
  plt.show()

class_numbers = [2,0,8,7,5]
#Part 1
class_number_images , class_number_labels = dataset_searcher(class_numbers, images, labels)
#Part 2
print_numbers(class_number_images , class_number_labels )



model_1 = GaussianNB()

#however, before we fit the model we need to change the 8x8 image data into 1 dimension
# so instead of having the Xtrain data beign of shape 718 (718 images) by 8 by 8
# the new shape would be 718 by 64

X_train_reshaped = X_train.reshape(X_train.shape[0], -1)

#Now we can fit the model
model_1.fit(X_train_reshaped, y_train)
#Part 3 Calculate model1_results using model_1.predict()
X_test_new = X_test.reshape(X_test.shape[0], -1)
model1_results = model_1.predict(X_test_new)

def OverallAccuracy(results, actual_values):
  #Calculate the overall accuracy of the model (out of the predicted labels, how many were correct?)
  Accuracy = 0
  for i, j in enumerate(results):
    if j == actual_values[i]:
      Accuracy += 1

  return Accuracy


# Part 4
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print("The overall results of the Gaussian model is " + str(Model1_Overall_Accuracy))
## print("Tot: " + str(len(model1_results)))


#Part 5
allnumbers = [0,1,2,3,4,5,6,7,8,9]
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, images, labels)
print_numbers(allnumbers_images, allnumbers_labels)




#Part 6
#Repeat for K Nearest Neighbors
model_2 = KNeighborsClassifier(n_neighbors=10)
model_2.fit(X_train_reshaped, y_train)
model_2_results = model_2.predict(X_test_new)
Model2_Overall_Accuracy = OverallAccuracy(model_2_results, y_test)
print("The overall results of the KNN model is " + str(Model2_Overall_Accuracy))


#Repeat for the MLP Classifier
model_3 = MLPClassifier(random_state=0)
model_3.fit(X_train_reshaped, y_train)
model_3_results = model_3.predict(X_test_new)
Model3_Overall_Accuracy = OverallAccuracy(model_3_results, y_test)
print("The overall results of the MLP model is " + str(Model3_Overall_Accuracy)) 



#Part 8
#Poisoning
# Code for generating poison data. There is nothing to change here.
noise_scale = 10.0
#rng = np.random.default_rng(12345)
poison = rng.normal(scale=noise_scale, size=X_train.shape)

X_train_poison = X_train + poison

# print("1:" +str(X_train_poison[0:9]))
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, X_train_poison, labels)
print_numbers(allnumbers_images, allnumbers_labels)

#Part 9-11
#Determine the 3 models performance but with the poisoned training data X_train_poison and y_train instead of X_train and y_train
X_ttrain_poison_new = X_train_poison.reshape(X_train_poison.shape[0], -1)
model_1 = GaussianNB()
model_1.fit(X_ttrain_poison_new, y_train)
model_1_results = model_1.predict(X_test_new)
Model1_Overall_Accuracy = OverallAccuracy(model_1_results, y_test)
print("The poisoned results of the Gaussian model is " + str(Model1_Overall_Accuracy))

model_2 = KNeighborsClassifier(n_neighbors = 10)
model_2.fit(X_ttrain_poison_new, y_train)
model_2_results = model_2.predict(X_test_new)
Model2_Overall_Accuracy = OverallAccuracy(model_2_results, y_test)
print("The poisoned results of the KNN model is " + str(Model2_Overall_Accuracy))

model_3 = MLPClassifier(random_state = 0)
model_3.fit(X_ttrain_poison_new, y_train)
model_3_results = model_3.predict(X_test_new)
Model3_Overall_Accuracy = OverallAccuracy(model_3_results, y_test)
print("The poisoned results of the MLP model is " + str(Model3_Overall_Accuracy)) 

#Part 12-13
# Denoise the poisoned training data, X_train_poison. 
# hint --> Suggest using KernelPCA method from sklearn library, for denoising the data. 
# When fitting the KernelPCA method, the input image of size 8x8 should be reshaped into 1 dimension
# So instead of using the X_train_poison data of shape 718 (718 images) by 8 by 8, the new shape would be 718 by 64

X_train_recon = KernelPCA(n_components = 64, kernel = "poly", gamma = 3e-5, fit_inverse_transform = True, alpha = 0.9e-3, max_iter = 500) # fill in the code here
# X_train_recon = KernelPCA(n_components = 64, fit_inverse_transform = True)
X_train_recon.fit(X_ttrain_poison_new)
# X_train_denoised = X_train_recon.fit_inverse_transform(X_train_recon.transform(X_ttrain_poison_new))
X_train_denoised = X_train_recon.inverse_transform(X_train_recon.fit_transform(X_ttrain_poison_new))
# X_train_denoised = X_train_recon.fit_inverse_transform(X_train_recon.transform(X_ttrain_poison_new))
x_train_denoised_print = X_train_denoised.reshape(X_train_denoised.shape[0], 8, 8)

allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, x_train_denoised_print, labels)
print_numbers(allnumbers_images, allnumbers_labels)

#Part 14-15
#Determine the 3 models performance but with the denoised training data, X_train_denoised and y_train instead of X_train_poison and y_train
#Explain how the model performances changed after the denoising process.


# print("Tot: "+ str(len(model1_results)))

model_1 = GaussianNB()
model_1.fit(X_train_denoised, y_train)
model_1_results = model_1.predict(X_test_new)
Model1_Overall_Accuracy = OverallAccuracy(model_1_results, y_test)
print("The denoised results of the Gaussian model is " + str(Model1_Overall_Accuracy))

model_2 = KNeighborsClassifier(n_neighbors = 10)
model_2.fit(X_train_denoised, y_train)
model_2_results = model_2.predict(X_test_new)
Model2_Overall_Accuracy = OverallAccuracy(model_2_results, y_test)
print("The denoised results of the KNN model is " + str(Model2_Overall_Accuracy))

model_3 = MLPClassifier(random_state = 0, max_iter =  1000)
model_3.fit(X_train_denoised, y_train)
model_3_results = model_3.predict(X_test_new)
Model3_Overall_Accuracy = OverallAccuracy(model_3_results, y_test)
print("The denoised results of the MLP model is " + str(Model3_Overall_Accuracy)) 

# print_numbers(X_train_poison[0:20], allnumbers_labels)
