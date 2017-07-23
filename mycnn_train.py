# Import Keras packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

### PARS ---
isize=   128		# img size
nfilter= (32,32,64)	# number of filters
ndense=  64		# number of dense layer (nfiler * 2)
nepochs= 90		# number of epochs
Ntrain= 25000
Ntest=      0
ofname_model= 'classifier_DogCat_is128nep90_cv32cv32cv64ds64drp2.h5'
### PARS ---

# Initializing CNN
classifier= Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(nfilter[0], (3, 3), input_shape=(isize,isize,3), activation='relu'))
# input_shape order for Tensorflow
# input_shape can be as large of 128x128 if run in GPU
# Stride= 1

# Step 2 - Maxpooling
classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# Input of "pool_size" & "strides" are actually default values

# ADDING A LAYER TO IMPROVE ACCURACY
classifier.add(Convolution2D(nfilter[1], (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# ADDING THIRD LAYER
classifier.add(Convolution2D(nfilter[2], (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units=ndense, activation='relu'))
classifier.add(Dropout(0.5))
#classifier.add(Dense(output_dim=isize*2, activation='relu')) # try adding one layer (bad idea)
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory('dataset/training_set',
        target_size=(isize,isize), batch_size=32, class_mode='binary')

#test_datagen = ImageDataGenerator(rescale=1./255)
#test_set = test_datagen.flow_from_directory('dataset/test_set',
#        target_size=(isize,isize), batch_size=32, class_mode='binary')

print(training_set.class_indices)

classifier.fit_generator(training_set, steps_per_epoch= Ntrain/32.0, epochs=nepochs)
#        validation_data=test_set, validation_steps= Ntest/32.0)

classifier.save(ofname_model)
