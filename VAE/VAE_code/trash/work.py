'''Define the training, evaluation and test process !!!'''
#######################################################################
import os
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from network import myVAE
from load_data.load_mnist import load_data_mnist
from utils import shuffle, loss_vae, randomly_choose_from_test
#######################################################################
def train_single_epoch(device, model, loss_func, optimizer, args_config, train_data):
    model.train()
    loss = 0.0
    st, ed, times = 0, args_config.batch_size, 0
    while st < len(train_data) and ed <= len(train_data):
        optimizer.zero_grad()
        X_batch = torch.from_numpy(train_data[st:ed]).to(device)
        X_new, mu, sigma = model(X_batch)

        loss_ = loss_func(X_new, X_batch, mu, sigma)
        loss_.backward()
        optimizer.step()
        loss += loss_.cpu().data.numpy()
        st, ed = ed, ed + args_config.batch_size
        times += 1
    loss /= times

    return loss


def valid_single_epoch(device, model, loss_func, args_config, valuation_data):
    model.eval()
    loss = 0.0
    st, ed, times = 0, args_config.batch_size, 0
    while st < len(valuation_data) and ed <= len(valuation_data):
        X_batch = torch.from_numpy(valuation_data[st:ed]).to(device)
        X_new, mu, sigma = model(X_batch)

        loss_ = loss_func(X_new, X_batch, mu, sigma)
        loss += loss_.cpu().data.numpy()
        st, ed = ed, ed + args_config.batch_size
        times += 1
    loss /= times

    return loss


def inference_single_epoch(model, loss_func, X_test):
    model.eval()
    with torch.no_grad():
        X_new, mu , sigma = model(X_test)
        
        loss_ = loss_func(X_new, X_test, mu, sigma)
        loss += loss_.cpu().data.numpy()

        return X_new.cpu().data.numpy(), loss


def train_process(args_config,):
    # Check the work state.
    if args_config.is_train is False:
        raise ValueError('[Error]: Not Training Process!!! Please Check!!!')
    

    # Begin
    print("-------------------------------------")
    print("[INFO] Begin Training!!!")
    

    # Make training directory and result directory if not exists.
    train_dir = args_config.train_dir
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    
    result_dir = args_config.result_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


    # Load data for training process.
    if args_config.dataset is None:
        raise ValueError('[Error]: No Dataset Specified!!! Please Check!!!')
    elif args_config.dataset == 'mnist':
        X_train, X_val, X_test = load_data_mnist(args_config.data_dir)
    elif args_config.dataset == 'cifar10':
        raise ValueError('[Error]: Not Done!!! Sorry!!!')
        # TODO
    else:
        raise ValueError('[Error]: Not Done!!! Sorry!!!')
        # TODO


    # Define the model, loss function and optimizer.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_model = myVAE()
    vae_model.to(device)
    loss_func = loss_vae
    optimizer = optim.Adam(vae_model.parameters(), lr=args_config.lr)


    # Load the model if exists.
    model_path = train_dir + f"checkpoint_{args_config.inference_version}.pth.tar"
    if os.path.exists(model_path):
        vae_model = torch.load(model_path)


    # Train and valid circles.
    pre_losses = [1e18] * 3
    best_val_loss = 1e18
    epoch_nums = args_config.epoch_nums

    for epoch in range(1, epoch_nums):
        start_time = time.time()
        train_loss = train_single_epoch(device, vae_model, loss_func, optimizer, args_config, X_train)
        X_train = shuffle(X_train, 1)

        # Valid.
        val_loss = valid_single_epoch(device, vae_model, args_config, X_val)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            test_loss = valid_single_epoch(device, vae_model, loss_func, args_config, X_test)

            # Save the model if it is the best model till now.
            model_path = train_dir + f"checkpoint_{args_config.inference_version}.pth.tar"
            with open(model_path, 'wb') as fout:
                torch.save(vae_model, fout)

        
        # Print the results.
        epoch_time = time.time() - start_time
        print("-------------------------------------")
        print(f"[INFO] Training Epoch {epoch}/{epoch_nums}:")
        print(f"[INFO] Time {epoch_time}.")
        print(f"[INFO] Learning Rate {args_config.lr}")
        print(f"[INFO] Training Loss {train_loss}")
        print(f"[INFO] Validation Loss {val_loss}")
        print(f"[INFO] Best Validation Accuracy {best_val_loss} at Epoch {best_epoch}")
        print(f"[INFO] Testing Loss {test_loss}")

        # Save the training data.
        with open(result_dir + f'/{args_config.model_name}_train.txt', "a") as f:
            f.write(f'Epoch {epoch}/{args_config.epoch_nums}:\n')
            f.write(f'Training Loss: {train_loss:.4f}\n')
            f.write('-' * 50 + '\n')


        # Ajust learning rate.
        if train_loss > max(pre_losses):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9995
        pre_losses = pre_losses[1:] + [train_loss]


    # End Training.
    print("-------------------------------------")
    print("[INFO] Training Finished!!!")
    print(f"[INFO] Best Validation Accuracy {best_val_loss} at Epoch {best_epoch}")
    print("-------------------------------------")


def test_process(args_config,):
    # Check the work state.
    if args_config.is_train is True:
        raise ValueError('[Error]: Not Test Process!!! Please Check!!!')
    

    # Begin
    print("-------------------------------------")
    print("[INFO] Begin Test!!!")
    

    # Make training directory and result directory if not exists.
    train_dir = args_config.train_dir
    if not os.path.exists(train_dir):
        raise ValueError('[Error]: Training Directory Does Not Exist!!! Please Check!!!')
    result_dir = args_config.result_dir
    if not os.path.exists(result_dir):
        raise ValueError('[Error]: Training Directory Does Not Exist!!! Please Check!!!')
    

    # Load the model.
    model_path = os.path.join(args_config.train_dir, 'checkpoint_%d.pth.tar' % args_config.inference_version)
    if os.path.exists(model_path):
        vae_model = torch.load(model_path)
    
    
    # Load the test data.
    if args_config.dataset is None:
        raise ValueError('[Error]: No Dataset Specified!!! Please Check!!!')
    elif args_config.dataset == 'mnist':
        X_train, X_val, X_test = load_data_mnist(args_config.data_dir)
    elif args_config.dataset == 'cifar10':
        raise ValueError('[Error]: Not Done!!! Sorry!!!')
        # TODO
    else:
        raise ValueError('[Error]: Not Done!!! Sorry!!!')
        # TODO
    
    # Randomly choose test data from test dataset with a specified quantity.
    num_inferences = args_config.num_inferences
    X_inference = randomly_choose_from_test(X_test, num_inferences)


    # Test circles.
    count = 0
    gen_new_fig, inference_loss = inference_single_epoch(vae_model, X_inference)


    # Output inference results for comparison.
    plt.figure(figsize=(20, 4))
    for i in range(num_inferences):
        # Initial figs.
        ax = plt.subplot(2, num_inferences, i + 1)
        plt.imshow(X_inference[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Generated figs.
        ax = plt.subplot(2, num_inferences, i + 1 + num_inferences)
        plt.imshow(gen_new_fig[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(f"{result_dir}/compare_{args_config.model_name}.png")
    plt.close()