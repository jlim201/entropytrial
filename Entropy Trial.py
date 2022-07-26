import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax
from tensorflow.keras import regularizers

#%%
np.random.seed(42)

#%%
img = cv2.imread('E:/Negin Project/entropy.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x_data = np.load('E:/Negin Project/data/X_small.npy')
y_data = np.load('E:/Negin Project/data/y_small.npy')
#%% 
#convert 2D image into single list (array)
df = pd.DataFrame()
count = 0
for x in x_data:
    reshaped = x_data[count].reshape(-1)
    label = 'frame' + str(count)
    df[label] = reshaped
    count+=1
    

#%%
newdf= df.copy()
#print(reshaped.shape)
plt.imshow(x_data[1], cmap='gray')
plt.show()
#%%
image_num = 196
image_label = 'frame' + str(image_num)
#print(y_data[0:100])
#img = x_data[1].values.reshape(256,256)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#print(img)
plt.imshow(x_data[1], cmap='gray')
plt.show()

#%%
from PIL import Image
import os 

#path1 = r'E:/Negin Project/images/0'
#path2 = r'E:/Negin Project/images/1'
for x in range(0, len(y_data)):
    im = Image.fromarray(x_data[x])
    image_label = 'frame' + str(x)
    if y_data[x] == 1.0:
        string = 'E:/Negin Project/images/1/' + str(x) + '.png'
        plt.imsave(string, x_data[x])
    elif y_data[x] == 0.0:
        string = 'E:/Negin Project/images/0/' + str(x) + '.png'
        plt.imsave(string, x_data[x])
        
    
#%%
os.listdir('E:/Negin Project')
DATADIR = r'E:/Negin Project/images'
CATEGORIES = ['0', '1']
CATEGORIES
#%%
def load_data():
    DATADIR = r'E:/Negin Project/images'
    data = []
    # loading training data
    for category in CATEGORIES:
        # create path to image of respective expression
        path = os.path.join(DATADIR, category)
        # get the classification  for each expression 
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
            img_array = cv2.imread(os.path.join(path, img), 0)
            data.append([img_array, class_num])
            
    return data
#%%
data = load_data()
#%%
L = 4
W = 4
fig, axes = plt.subplots(L, W, figsize = (15,15))
axes = axes.ravel()

for i in range(0, L * W):  
    sample = random.choice(data)
    axes[i].set_title("Expression = "+str(CATEGORIES[sample[1]]))
    axes[i].imshow(sample[0], cmap='gray')
    axes[i].axis('off')
plt.subplots_adjust(wspace=0.5)
#%%
X = np.array([x[0] for x in data])
y = np.array([Y[1] for Y in data])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = True)
y_train = np.reshape(y_train, (len(y_train),1))
y_test  = np.reshape(y_test , (len(y_test ),1))

print("After reshaping")
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

#%%
X_train_Gabor  = X_train
X_test_Gabor = X_test

#%%
X_train = X_train / 255.0
X_test = X_test / 255.0


#%%
print(y_train[0])


y_train = tf.keras.utils.to_categorical(y_train).reshape((-1,1))
y_test = tf.keras.utils.to_categorical(y_test).reshape((-1,1))
print(y_train[0])
y_train.shape, y_test.shape
#%%
def create_model(input_shape=None):
    if input_shape is None :
        input_shape=(246, 246, 1)

    model = Sequential()
    model.add(Conv2D(6, (5, 5), input_shape=input_shape, padding='same', activation = 'relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (5, 5), padding='same', activation = 'relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'softmax'))
    
    
    return model 
#%%
es = EarlyStopping(
    monitor='val_binary_accuracy', min_delta=0.0001, patience=10, verbose=2,
    mode='max', baseline=None, restore_best_weights=True
)
lr = ReduceLROnPlateau(
    monitor='val_binary_accuracy', factor=0.1, patience=5, verbose=2,
    mode='max', min_delta=1e-5, cooldown=0, min_lr=0
)

callbacks = [es, lr]


#%%
def GaborFeature(img, df2):
    ksize = 5 #kernelsize - feature size
    phi = 1
    num = 1
    dim = len(img.shape)
    features = []
    fmasks = []
    #print(dim)
    gabor_label = 'Gabor' + str(num)
    sigma = 3
    theta = 1.570796327
    lamda = 0.785398163
    gamma = 1
    kernel = cv2.getGaborKernel((ksize, ksize),sigma, theta, lamda*5, gamma, phi, ktype = cv2.CV_32F)
    #fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
    window = np.outer(kernel, kernel.transpose())
    fimg = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    filtered_img = fimg.reshape(-1)
    df2[gabor_label] = filtered_img
    gabor_img = df2[gabor_label].values.reshape(246,246)
    #plt.imshow(gabor_img, cmap='gray')
    #plt.title('ksize:'+str(ksize)+' sigma:'+str(sigma)+' theta:'+str(theta)+' lamda:'+str(lamda*5)+' gamma:'+str(gamma)+' phi:'+str(phi))
    #plt.show()
    num+=1
    if(dim == 2):
      ret,fmask = cv2.threshold(fimg,254,1,cv2.THRESH_BINARY_INV)
      fmask = fmask.reshape((gabor_img.shape[0]*gabor_img.shape[1],))
      fimg = fimg.reshape((gabor_img.shape[0]*gabor_img.shape[1],))
      features.append(fimg)
      features.append(fmask)
    elif(dim == 3):
      fimg = fimg.reshape((fimg.shape[0]*fimg.shape[1],3))
      features.append(fimg[:,0])
      features.append(fimg[:,1])
      features.append(fimg[:,2])
    else:
      print("Channels error")
      return 0

                      
    features = np.array(features)
    features = features.T
    return gabor_img
#%%
def create_Gabor_features(data, df2):
    Feature_data = np.zeros((len(data),246,246))

    for i in range(len(data)):
        img = data[i]
        out = GaborFeature(img, df2)
        Feature_data[i] = out/255.00

        
    return Feature_data

#%%
df2 = pd.DataFrame()
plt.imshow(X_train_Gabor[0]/255.0, cmap ='gray')
X_train_Gabor=create_Gabor_features(X_train_Gabor, df2)
X_test_Gabor=create_Gabor_features(X_test_Gabor, df2)
#%%
print(X_train_Gabor.max())
#%%
print(X_train_Gabor.shape , X_test_Gabor.shape)
sample = 3#random.randint(0,5)
plt.subplot(1,2,1)
plt.imshow(X_train[sample],cmap='gray')
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(X_train_Gabor[sample],cmap='gray')
plt.axis("off")
print(X_train_Gabor.shape)

#%%
Gabor_model = create_model()
#loss= activation='sigmoid'
Gabor_model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer='adam' )
#%%
Gabor_model.summary()
#%%
Gabor_history = Gabor_model.fit(X_train_Gabor, y_train[0:1796], batch_size= 8 , epochs=50, validation_data = (X_test_Gabor, y_test[0:449]) ,callbacks = [callbacks])

#%%
def plot_performance(history):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')

    plt.legend()
    plt.grid()
    plt.title('train and val loss evolution')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')

    plt.legend()
    plt.grid()
    plt.title('train and val accuracy')
#%%
plot_performance(Gabor_history)
acc = []
Gabor_acc = Gabor_model.evaluate(X_test_Gabor, y_test, verbose = 0)[1]
acc.append(Gabor_acc)
print("Gabor Accuracy :",Gabor_model.evaluate(X_test_Gabor, y_test, verbose = 0)[1])
Gabor_model.save('Gabor_model.h5')
