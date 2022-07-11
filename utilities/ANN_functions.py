import tensorflow as tf
from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras import optimizers
from keras.layers import GaussianNoise



def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def r2_keras_loss(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return SS_res/(SS_tot + K.epsilon())

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

    
def create_loso_model(n_features, final):
    model = Sequential()
    
    # Input layer
    model.add(Dense(n_features, input_shape=(n_features,), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    
    if(final):
        # Hidden layers
        model.add(Dense(60, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
    #        
        model.add(Dense(50, kernel_initializer='he_normal'))
        model.add(BatchNormalization())  
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(40, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(30, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(20, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(10, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        
    else:
           # Hidden layers
        model.add(Dense(40, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.2))
    #        
        model.add(Dense(20, kernel_initializer='he_normal'))
        model.add(BatchNormalization())  
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.1)) 
        
        model.add(Dense(10, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.05))
        
        model.add(Dense(5, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.01))
    
    # Output layer
    model.add(Dense(1))
    
    ##### Optimizers  #######
    optimizer = optimizers.RMSprop(lr=0.005)
#    optimizer = optimizers.rmsprop(lr=0.0005)
#    optimizer = optimizers.RMSprop(lr=0.001)
    
    # Compilation
    model.compile(optimizer = optimizer, loss=root_mean_squared_error, metrics=[r2_keras])
    
    return model

def create_loyo_model(n_features):
    model = Sequential()
    
    # Input layer
    model.add(Dense(n_features, input_shape=(n_features,), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    
    # Hidden layers
    model.add(Dense(40, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.3))
    
    model.add(Dense(20, kernel_initializer='he_normal'))
    model.add(BatchNormalization()) 
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.2))
        
    model.add(Dense(10, kernel_initializer='he_normal'))
    model.add(BatchNormalization()) 
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.1))
    
    model.add(Dense(5, kernel_initializer='he_normal'))
    model.add(BatchNormalization()) 
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.1))
    
    # Output layer
    model.add(Dense(1))
    
    ##### Optimizers  #######
#    optimizer = optimizers.rmsprop(lr=0.05)
    optimizer = optimizers.RMSprop(lr=0.02)
    
    # Compilation
    model.compile(optimizer = optimizer, loss=root_mean_squared_error, metrics=[root_mean_squared_error])
#    model.compile(optimizer = optimizer, loss=r2_keras_loss, metrics=[r2_keras])

    
    return model

def create_lsyso_model(n_features, final):
    model = Sequential()
        
    # Input layer
    model.add(Dense(n_features, input_shape=(n_features,), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    
    # Hidden layers
    if(final):
        # Hidden layers
        model.add(Dense(80, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
    #        
        model.add(Dense(60, kernel_initializer='he_normal'))
        model.add(BatchNormalization())  
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(50, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(30, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(20, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(10, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(10, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(5, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        
    else:
        model.add(GaussianNoise(0.1))
        model.add(Dense(40, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.30))
    #    model.add(Dropout(0.1))
        
        model.add(Dense(20, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.20))
    #    model.add(Dropout(0.1))
        
        model.add(Dense(10, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.1))
    #    model.add(Dropout(0.05))
    
        model.add(Dense(5, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.1))
    #    model.add(Dropout(0.05))
    
        model.add(Dense(5, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.1))
    #    model.add(Dropout(0.05))
    
    # Output layer
    model.add(Dense(1))
        
    ##### Optimizers  #######
    optimizer = optimizers.RMSprop(lr=0.01)
#    optimizer = optimizers.rmsprop(lr=0.002)
#    optimizer = optimizers.rmsprop(lr=0.001)
#    optimizer = optimizers.rmsprop(lr=0.0005)
#    optimizer = optimizers.adam(lr=0.0005)
#    optimizer = optimizers.adam(lr=0.0003)
#    optimizer = optimizers.rmsprop(lr=0.0007)
#    optimizer = optimizers.rmsprop(lr=0.0008)
    
    # Compilation
    model.compile(optimizer = optimizer, loss=root_mean_squared_error, metrics=[root_mean_squared_error])
#    model.compile(optimizer = optimizer, loss=r2_keras_loss, metrics=[r2_keras])

    
    return model
