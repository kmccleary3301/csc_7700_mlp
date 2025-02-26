import requests
from io import BytesIO
import os
import PIL.Image
import pyarrow.parquet as pq
import numpy as np
from mlp import MultilayerPerceptron
from layer import Layer, LayerBatchNormed
from activations import Mish, Softmax, ReLU
from loss_functions import CrossEntropy
from optimizers import AdamW2017
from schedulers import CosineScheduler
import pickle
import matplotlib.pyplot as plt
from math import fmod


def augment_image(
    image : PIL.Image,
    trans_margin_x : float = 0.1,
    trans_margin_y : float = 0.1,
    rot_margin_degrees : float = 20
):
    # Get dimensions
    width, height = image.size
    
    # Random translation within margins
    dx = int(width * trans_margin_x * (2 * np.random.random() - 1))  # -10% to +10%
    dy = int(height * trans_margin_y * (2 * np.random.random() - 1))  # -10% to +10%
    
    # Random rotation angle
    angle = np.random.uniform(-rot_margin_degrees, rot_margin_degrees)
    
    # Create transform matrix
    translated = image.transform(image.size, PIL.Image.AFFINE, (1, 0, dx, 0, 1, dy))
    augmented = translated.rotate(angle, PIL.Image.BILINEAR, expand=False)
    
    return augmented



def unpack_data(
    data,
    use_augmentation : bool = False,
    trans_margin_x : float = 0.1,
    trans_margin_y : float = 0.1,
    rot_margin_degrees : float = 15,
    save_images : bool = False
):
    # Initialize arrays to store images and labels
    X = np.zeros((len(data), 784))  # 28x28 = 784 pixels
    y = np.zeros((len(data), 10))    # 10 classes for digits 0-9

    # Process each image
    for idx, row in data.iterrows():
        # Convert bytes to image
        img = PIL.Image.open(BytesIO(row['image']['bytes']))
        if use_augmentation:
            img = augment_image(
                img,
                trans_margin_x=trans_margin_x,
                trans_margin_y=trans_margin_y,
                rot_margin_degrees=rot_margin_degrees
            )
        if idx < 5 and save_images:
            img.save(f'outputs/x_train_{idx}.png')

        # Convert to grayscale numpy array and flatten
        img_array = np.array(img).flatten() / 255.0  # Normalize to [0,1]
        X[idx] = img_array
        
        # Create one-hot encoded labels
        y[idx, row['label']] = 1

    return X, y




if __name__ == "__main__":
    
    try:
        os.chdir(globals()['_dh'][0]) # Jupyter notebook
    except:
        os.chdir(os.path.dirname(__file__)) # Standard Python

    DATASET_FOLDER = "datasets"
    
    for folder in ['datasets', 'logs', "outputs"]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    if not os.path.exists(os.path.join(DATASET_FOLDER, 'mnist_train.parquet')):
        # Download the MNIST data
        print("Downloading MNIST Training Set...")
        url = "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/train-00000-of-00001.parquet?download=true"
        response = requests.get(url)
        # Cache the file locally
        with open(os.path.join(DATASET_FOLDER, 'mnist_train.parquet'), 'wb') as f:
            f.write(response.content)


    if not os.path.exists(os.path.join(DATASET_FOLDER, 'mnist_test.parquet')):
        # Download the MNIST data
        print("Downloading MNIST Test Set...")
        url = "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet?download=true"
        response = requests.get(url)
        # Cache the file locally
        with open(os.path.join(DATASET_FOLDER, 'mnist_test.parquet'), 'wb') as f:
            f.write(response.content)

    # Read the cached file
    mnist_train_data = pq.read_table(os.path.join(DATASET_FOLDER, 'mnist_train.parquet')).to_pandas()
    mnist_test_data = pq.read_table(os.path.join(DATASET_FOLDER, 'mnist_test.parquet')).to_pandas()
    
    # Unpack the training
    x_train, y_train = unpack_data(
        mnist_train_data, 
        # use_augmentation=True, 
        save_images=True,
        trans_margin_x=0.1,
        trans_margin_y=0.1,
        rot_margin_degrees=20
    )
    x_test, y_test = unpack_data(mnist_test_data, use_augmentation=False)

    model_mnist = MultilayerPerceptron([
        Layer(784, 256, Mish()),
        Layer(256, 256, Mish(), dropout_rate=0.2),
        Layer(256, 10, Softmax())
        
        # Layer(784, 256, ReLU()),
        # Layer(256, 256, ReLU(), dropout_rate=0.2),
        # Layer(256, 10, Softmax())
        
        # Layer(784, 1024, Mish()),
        # Layer(1024, 1024, Mish(), dropout_rate=0.2),
        # Layer(1024, 1024, Mish(), dropout_rate=0.2),
        # Layer(1024, 10, Softmax())
        
        # LayerBatchNormed(784, 1024, Mish(), use_batchnorm=True),
        # LayerBatchNormed(1024, 1024, Mish(), dropout_rate=0.2, use_batchnorm=True),
        # LayerBatchNormed(1024, 1024, Mish(), dropout_rate=0.2, use_batchnorm=True),
        # Layer(1024, 10, Softmax())
    ])
    
    print(model_mnist)
    
    heuristics = model_mnist.train(
        x_train,
        y_train,
        x_test,
        y_test,
        loss_func=CrossEntropy(),
        # learning_rate=1e-2,
        learning_rate_scheduler=CosineScheduler(
            initial_lr=1e-3,
            final_lr=1e-5,
            max_epochs=5,
            # reset_each_epoch=True,
            # exp_decay_factor=1e-1
        ),
        batch_size=20,
        epochs=5,
        # rmsprop=True, # This was implemented, but AdamW is better.
        optimizer=AdamW2017(1e-3),
        loss_averaging_window_fraction=0.1,
        continuous_validation=True
    )
    
    # Extract epochs and values for each metric
    # print(heuristics["training_loss"][:5])
    
    
    val_loss_epochs = [x['epoch'] for x in heuristics['val_loss']]
    val_loss_values = [x['value'] for x in heuristics['val_loss']]
    train_loss_epochs = [x['epoch'] for x in heuristics['training_loss']]
    train_loss_values = [x['value'] for x in heuristics['training_loss']]
    test_loss_epochs = [x['epoch'] for x in heuristics['test_loss_after_epoch']]
    test_loss_values = [x['value'] for x in heuristics['test_loss_after_epoch']]
    test_acc_epochs = [x['epoch'] for x in heuristics['test_accuracy']]
    test_acc_values = [x['value'] for x in heuristics['test_accuracy']]
    
    lr_schedule_epochs = [x['epoch'] for x in heuristics['lr_schedule']]
    lr_schedule_values = [x['value'] for x in heuristics['lr_schedule']]

    max_lr_value = 1e-2
    lr_schedule_values = np.array(lr_schedule_values)  
    lr_schedule_values = lr_schedule_values / max_lr_value # Normalize to initial learning rate

    # Create the plot
    plt.figure(figsize=(10, 6))
    # plt.plot(epochs, loss_values, marker='o', linestyle='-')
    plt.plot(train_loss_epochs, train_loss_values, label='Training Loss', linestyle='-')
    plt.plot(val_loss_epochs, val_loss_values, label='Val Loss', linestyle='-')
    plt.plot(lr_schedule_epochs, lr_schedule_values, label='Learning Rate', linestyle='-')
    # plt.plot(test_loss_epochs, test_loss_values, label='Test Loss')
    # plt.plot(test_acc_epochs, test_acc_values, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Metrics')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 2)  # Set y-axis limits from 0 to 2

    # Save the plot as PDF
    plt.savefig('outputs/training_metrics.png')
    plt.close()
    
    for i in range(20):
        result = model_mnist.forward(x_test[i])
        print(result)
        print(np.argmax(result), np.argmax(y_test[i]))
        print(y_test[i])
    
    # Save model weights and biases using pickle

    with open('outputs/mnist_model.pkl', 'wb') as f:
        pickle.dump(model_mnist, f)
    
    
    # scheduler = CosineScheduler(
    #     initial_lr=1e-3,
    #     final_lr=1e-4,
    #     max_epochs=20,
    #     reset_each_epoch=True,
    #     exp_decay_factor=1e-1
    # )
    
    
    # values_x = np.arange(0, scheduler.max_epochs, 0.01)
    # values_y = [scheduler.get_learning_rate(fmod(epoch, 1), epoch/scheduler.max_epochs) for epoch in values_x]
    
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(values_x, values_y)
    # plt.xlabel('Epoch')
    # plt.ylabel('Learning Rate')
    # plt.title('Learning Rate Schedule')
    # plt.grid(True)
    # plt.savefig('outputs/learning_rate_schedule.png')
    # plt.close()
    
    for i in range(10):
        
        
        y_test_values = np.argmax(y_test, axis=1)
        
        sample_input_i = np.where(y_test_values == i)[0]
        
        s_i_get = np.random.choice(sample_input_i)
        
        
        y_correct = y_test_values[s_i_get]
        x_sample = x_test[s_i_get]
        
        result = model_mnist.forward(x_sample)
        y_pred = np.argmax(result)
        
        x_sample = x_sample.reshape(28, 28)
        
        x_sample_pil_image = PIL.Image.fromarray((x_sample * 255).astype(np.uint8))
        
        print("Y Correct:", y_correct)
        print("Y Pred:", y_pred)
        
        with open(f'outputs/sample_{i}_correct_{y_correct}_pred_{y_pred}.png', 'wb') as f:
            x_sample_pil_image.save(f)
        