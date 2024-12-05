'''Define the test process !!!'''
#######################################################################
import os
import torch
import matplotlib.pyplot as plt

from load_model.data_loader import load_data
from load_model.model_loader import load_old_model
from others.utils import randomly_choose_from_test
#######################################################################

# Check the work state.
def check(config):
    if config['settings']['is_train'] is True:
        raise ValueError('[Error]: Not Test Process!!! Please Check!!!')


# Begin
def begin_test():
    print("-------------------------------------")
    print("[INFO] Begin Test!!!")


# Check training directory and result directory.
def check_dir(config):
    train_dir = config['settings']['train_dir']
    if not os.path.exists(train_dir):
        raise ValueError('[Error]: Training Directory Does Not Exist!!! Please Check!!!')
    result_dir = config['settings']['result_dir']
    if not os.path.exists(result_dir):
        raise ValueError('[Error]: Training Directory Does Not Exist!!! Please Check!!!')


# Inference for each epoch.
def inference_single_epoch(model, loss_func, X_test):
    model.eval()
    loss = 0.0
    with torch.no_grad():
        X_new, mu , sigma = model(X_test)
        
        loss_ = loss_func(X_new, X_test, mu, sigma)
        loss += loss_.cpu().data.numpy()

        return X_new.cpu().data.numpy(), loss


# Plot the figs for each epoch.
def plot_figs(num_inferences, X_inference, gen_new_fig, result_dir, config):
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
    plt.savefig(f"{result_dir}/compare_{config['settings']['model_name']}.png")
    plt.close()


# Print the result.
def output_result(inference_loss, num_inferences):

    test_loss = inference_loss / num_inferences
    print(f"[INFO] Test Loss {test_loss}")

    return test_loss


# Save the test result.
def save_result(test_loss, result_dir, config):
    with open(result_dir + f'/{config["settings"]["model_name"]}_test.txt', 'w') as f:
        f.write(f"[INFO] Test Loss {test_loss}")


def test_job(model, config, test_data):
    train_dir, result_dir = check_dir(config)
    # Randomly choose test data from test dataset with a specified quantity.
    num_inferences = config['settings']['num_inferences']
    X_inference = randomly_choose_from_test(test_data, num_inferences)
    # Test circles.
    gen_new_fig, inference_loss = inference_single_epoch(model, X_inference)
    # Output figs.
    plot_figs(num_inferences, X_inference, gen_new_fig, result_dir, config)
    # Print the results.
    test_loss = output_result(inference_loss, num_inferences)
    # Save the results.
    save_result(test_loss, result_dir, config)


def test_process(config):
    # Preprocess and begin test.
    check(config)
    begin_test()
    # Load data.
    X_train, X_val, X_test = load_data(config)
    # Load the model.
    vae_model = load_old_model(config)

    test_job(vae_model, config, X_test)