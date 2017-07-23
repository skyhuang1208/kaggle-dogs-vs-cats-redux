# Dog/Cat Classifier #
  
This repository contains the code and the results to classify images of cats and dogs.  
Using **Convolution Neural Network**.  
The CNN was developed using **keras** with **Tensorflow** as the backend.  
Running on **AWS** p2.xlarge (Ohio) **GPU** instance.

### Contributor ###
Chen-Hsi (Sky) Huang (github.com/skyhuang1208)

### Achievement ###
An validation accuracy of **91.96%** was achieved.  
An score of 0.31325 was obtained from kaggle.

### Neural Network Architecture ###
* Layer in: A batch (32) of 128x128 images  
* Layer 1:  Convolution2D(N of filters=32, Filter size= (3,3), activation='linear rectifier')
* Layer 1b: Maxpooling(pool size=(2,2)) 
* Layer 2:  Convolution2D(N of filters=32, Filter size= (3,3), activation='linear rectifier')
* Layer 2b: Maxpooling(pool size=(2,2)) 
* Layer 3:  Convolution2D(N of filters=64, Filter size= (3,3), activation='linear rectifier')
* Layer 3b: Maxpooling(pool size=(2,2))
* Layer 4: Dense(units=64, activation='linear rectifier')
* Layer 5: Dropout(rate=0.5)
* Layer 6: Dense(units=1, activation='sigmoid')
* Layer out: 1x1 matrix with probability of cat/dog.

