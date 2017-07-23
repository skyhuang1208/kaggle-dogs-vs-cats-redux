import sys

if len(sys.argv) != 5:
    print("Load trained model and predict cat or dog")
    exit("Usage: %s <model_file> <img_file> _img_size_ ISsubmission" % sys.argv[0])

# Import Keras packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model

classifier= load_model(sys.argv[1])

import os
import numpy as np
from keras.preprocessing import image

print()
print("********** Prediction begins!!!! **********")
rescale= 1./255
ncats= 0
ndogs= 0
prob= {}

if os.path.isdir(sys.argv[2]):  files= os.listdir(sys.argv[2])
else:                           files= [sys.argv[2]]

if sys.argv[2][-1]=='/': sys.argv[2]= sys.argv[2][:-1]

for f in files:
    if len(files)==1:
        test_image= image.load_img(f, target_size= (int(sys.argv[3]), int(sys.argv[3])) )
#        folder, name= sys.argv[2].rsplit('/', 1)
#        test_image.save(folder+"/"+"MODIFIED_"+name)
    else:
        if f[0]=='.': continue # skip .DS
        test_image= image.load_img(sys.argv[2]+"/"+f, target_size= (int(sys.argv[3]), int(sys.argv[3])) )
#        test_image.save(sys.argv[2]+"/"+"MODIFIED_"+f)
    test_image= image.img_to_array(test_image)
    test_image= np.expand_dims(test_image, axis=0)
    test_image *= rescale
    
    predict= classifier.predict(test_image)
    
    if sys.argv[4]=='yes':
        num, jpg= f.split('.')
        prob[int(num)]= predict[0][0]

    if predict[0][0] < 0.5:
        ncats +=1
        print("**** Image <", f, "> is a cat!", predict)
    else:
        ndogs +=1
        print("**** Image <", f, "> is a dog!", predict)

print("@@@ Pridicted", ncats, "cats and", ndogs, "dogs. Ratio(cats, dogs):", ncats/(ncats+ndogs), ndogs/(ncats+ndogs))
print("********** Prediction end... **********")
print()

if sys.argv[4]=='yes':
    outdata= sorted(prob.items(), key= lambda x: x[0])
    OFILE= open("submission.csv", 'w')
    print("id,label", file=OFILE)
    for key, value in outdata:
        print("%d,%f" % (key,value), file=OFILE)

