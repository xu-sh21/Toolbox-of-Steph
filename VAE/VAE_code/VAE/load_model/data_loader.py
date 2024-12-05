'''Define the function for loading data from diffrent datasets!!!'''
#######################################################################
from sklearn.model_selection import train_test_split
from others.load_data_func.load_mnist import load_data_mnist
#######################################################################
def load_data(config):
    if config['settings']['dataset'] is None:
        raise ValueError('[Error]: No Dataset Specified!!! Please Check!!!')
    elif config['settings']['dataset'] == 'mnist':
        X_tr, Y_tr, X_test, Y_test = load_data_mnist('../../dataset/MNIST')
        X_train, X_val, Y_train, Y_val = train_data_split(X_tr, Y_tr)
    elif config['settings']['dataset'] == 'cifar10':
        raise ValueError('[Error]: Not Done!!! Sorry!!!')
        # TODO
    else:
        raise ValueError('[Error]: Not Done!!! Sorry!!!')
        # TODO
    
    return X_train, X_val, X_test


# Split training data into training and validation parts.
def train_data_split(X_tr, Y_tr):

    # Split size means the proportion of the validation set to the total training set.
    split_size = 1 / 6

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_tr, Y_tr, test_size=split_size, random_state=42, stratify=Y_tr
    )
    
    return X_train, X_val, Y_train, Y_val