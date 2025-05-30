import tensorflow as tf
from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras import optimizers
from keras.layers import GaussianNoise
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


from sklearn import ensemble
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor

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


def create_RF_model(custom_cv):
    param_grid = {'n_estimators': [300, 400, 500, 600],
                  'max_features': [2, 3, 4, 8, 12],
                  'max_depth': [4, 8,], 
                  'min_samples_leaf': [5, 10, 20, 40],
                   },

    forest_reg = RandomForestRegressor()

    grid_search = RandomizedSearchCV(forest_reg, param_grid, cv=custom_cv, 
                               scoring='neg_mean_squared_error',
                               return_train_score=True, n_iter=10)
    return grid_search

def create_XGB_model(custom_cv):

    param_grid = {'n_estimators': [300, 400, 500, 600],
                  'max_features': [2, 3, 4, 8, 12],
                  'max_depth': [4, 8,], 
                  'min_samples_leaf': [5, 10, 20, 40],
                   },

    forest_reg = GradientBoostingRegressor()

    grid_search = RandomizedSearchCV(forest_reg, param_grid, cv=custom_cv, 
                               scoring='neg_mean_squared_error',
                               return_train_score=True, n_iter=10)

    return grid_search

def create_ANN_model(custom_cv):

    param_list = {'hidden_layer_sizes' : [50,30,20,10],
                  'max_iter' : [200, 250], 
                  'activation': ['identity', 'logistic', 'tanh', 'relu'],
                  'solver': ['lbfgs', 'sgd', 'adam'], 
                  'batch_size': [50, 100, 150],
                  'learning_rate':['constant', 'invscaling', 'adaptive'],
                  'alpha': [0.00005,0.0005], 'shuffle':[True, False], 
                  'validation_fraction':[0.1, 0.001]}

    forest_reg = MLPRegressor()

    MLP_gridCV = RandomizedSearchCV(estimator=forest_reg,  param_distributions=param_list, cv=custom_cv)  
    
    return MLP_gridCV

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
    model.compile(optimizer = optimizer, loss=root_mean_squared_error, metrics=[r2_keras])
#    model.compile(optimizer = optimizer, loss=r2_keras_loss, metrics=[r2_keras])
#    model.compile(optimizer = optimizer, loss=root_mean_squared_error, metrics=[r2_keras])

    
    return model

def create_loyso_model(n_features, final):
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
    model.compile(optimizer = optimizer, loss=root_mean_squared_error, metrics=[r2_keras])
#    model.compile(optimizer = optimizer, loss=root_mean_squared_error, metrics=[root_mean_squared_error])
#    model.compile(optimizer = optimizer, loss=r2_keras_loss, metrics=[r2_keras])

    
    return model
