import numpy as np
from typing import Tuple, List
from activations import Softmax
from loss_functions import LossFunction, CrossEntropy
from optimizers import Optimizer, RMSProp
from schedulers import Scheduler
from tqdm import tqdm
from layer import Layer
from datetime import datetime


class MultilayerPerceptron():
    def __init__(
        self,
        layers: List[Layer],
    ):
        assert len(layers) > 0, "At least one layer is required"
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[0] == self.layers[0].fan_in, f"Expected input size {self.layers[0].fan_in}, got {x.shape[0]}"
        assert len(x.shape) == 1, f"Expected 1D array, got {len(x.shape)}D array"

        forward_value = x
        for layer in self.layers:
            forward_value = layer.forward(forward_value)

        return forward_value
 
    def backward(
        self, 
        input_data: np.ndarray, 
        loss_grad: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert input_data.shape[0] == self.layers[0].fan_in, f"Expected input size {self.layers[0].fan_in}, got {input_data.shape[0]}"
        assert len(input_data.shape) == 1, f"Expected 1D array, got {len(input_data.shape)}D array"

        dW_all, db_all = [], []
        
        current_delta = loss_grad
        for layer_i in range(len(self.layers) - 1, -1, -1):
            
            layer_outputs_get = self.layers[layer_i-1].outputs if layer_i > 0 \
                else input_data

            (dW, db) = self.layers[layer_i].backward(layer_outputs_get, current_delta)
            dW_all.append(dW)
            db_all.append(db)
            current_delta = self.layers[layer_i].delta

        return dW_all[::-1], db_all[::-1]

    def toogle_layer_training_flags(
        self,
        training: bool
    ):
        for layer in self.layers:
            layer.training = training
 
 
    def train(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        loss_func: LossFunction,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        rmsprop: bool = False,
        learning_rate_scheduler: Scheduler = None,
        loss_averaging_window_fraction: float = 0.02,
        optimizer: Optimizer = None,
        continuous_validation: bool = False
    ):
        """
        Train the neural network using provided data and parameters.
        This method trains the network with the given training data, validates performance with validation data,
        and supports various optimization techniques.
        Parameters
        ----------
        train_x : np.ndarray
            Training input data.
        train_y : np.ndarray
            Training target data.
        val_x : np.ndarray
            Validation input data.
        val_y : np.ndarray
            Validation target data.
        loss_func : LossFunction
            Loss function used during training (e.g., CrossEntropy, MSE).
        learning_rate : float, optional
            Initial learning rate (default: 1e-3).
        batch_size : int, optional
            Mini-batch size for gradient updates (default: 32).
        epochs : int, optional
            Number of training epochs (default: 100).
        rmsprop : bool, optional
            Whether to use RMSProp optimizer (default: False).
        learning_rate_scheduler : Scheduler, optional
            Learning rate scheduler for adjusting learning rate during training.
        loss_averaging_window_fraction : float, optional
            Fraction of data to use for calculating moving average of loss (default: 0.02).
        optimizer : Optimizer, optional
            Custom optimizer to use instead of standard gradient descent.
        continuous_validation : bool, optional
            Whether to perform validation continuously during training (default: False).
        Returns
        -------
        dict
            Dictionary containing training metrics over time:
            - training_loss: List of average training losses
            - val_loss: List of average validation losses
            - test_loss_after_epoch: Validation loss after each epoch
            - test_accuracy: Validation accuracy after each epoch
            - lr_schedule: Learning rate values over time
        """
        
        if rmsprop:
            optimizer = RMSProp(learning_rate)

        train_loss_over_time, val_loss_over_time = [], []
        heurestic_graph = {
            "training_loss": [],
            "val_loss": [],
            "test_loss_after_epoch": [],
            "test_accuracy": [],
            "lr_schedule": []
        }
        
        rolling_window_size_train = int(len(train_x) * loss_averaging_window_fraction)
        rolling_window_size_val = int(len(val_x) * loss_averaging_window_fraction)
        
        grad_beta_decay = 0.9
        grad_accumulation_steps = 1
        
        print(f"Training started at: {datetime.now()}")
        start_time = datetime.now()
        
        
        for epoch in range(epochs):
            
            print("Training Epoch [%3d/%3d] at %s" % (epoch+1, epochs, datetime.now()))
            pbar = tqdm(range(len(train_x)), desc="Epoch [%3d/%3d]" % (epoch+1, epochs))
            current_batch_size = 0
            current_delta_weights, current_delta_biases = None, None
            delta_weights_accumulation, delta_biases_accumulation = [], []
            
            # Disable the softmax layer for the training phase
            # In the event we are using softmax and cross-entropy, 
            # we need to disable the softmax layer
            self.toogle_layer_training_flags(True)
            current_val_sample_iterator = 0
            
            for i in pbar:
                epoch_progress = (i + 1) / len(train_x)
                val_sample_target = int(epoch_progress * len(val_x))
                
                x_sample, y_true = train_x[i], train_y[i]
                
                # Check if we'll be applying our batch this time.
                apply_micro_batch = ((i+1) % batch_size == 0) or (i == len(train_x) - 1)
                
                y_pred = self.forward(x_sample)
                
                # Calculate the loss; use the hack for softmax and cross-entropy
                if isinstance(loss_func, CrossEntropy) and isinstance(self.layers[-1].activation, Softmax):
                    loss_delta = y_pred - y_true
                else:
                    loss_delta = loss_func.derivative(y_true, y_pred)
                
                loss_single = loss_func.loss(y_true, y_pred)
                
                # Run the forward and backward pass
                d_weights, d_bias = self.backward(x_sample, loss_delta)

                # Accumulate the gradients
                if current_delta_weights is None:
                    current_delta_weights = d_weights
                    current_delta_biases = d_bias
                else:
                    for i in range(len(d_weights)):
                        current_delta_weights[i] += d_weights[i]
                        current_delta_biases[i] += d_bias[i]
                
                # Run validation within the training loop
                if continuous_validation:
                    for val_i in range(current_val_sample_iterator, val_sample_target):
                        x_val_sample, y_val_true = val_x[val_i], val_y[val_i]
                        
                        y_val_pred = self.forward(x_val_sample)
                        # if self.layers[-1].activation is Softmax:
                        #     val_loss = loss_func.loss(y_val_true, y_val_pred)
                        
                        val_loss = loss_func.loss(y_val_true, y_val_pred)
                        
                        val_loss_over_time.append(val_loss)
                 
                    # self.layers[-1].training = True
                    current_val_sample_iterator = val_sample_target
                
                
                
                current_batch_size += 1

                if apply_micro_batch:
                    
                    # Calculate the current progress, and use a cosine scheduler to adjust the learning rate
                    current_progress = (epoch / epochs) + (i / epochs*len(train_x))

                    learning_rate_scheduled = learning_rate
                    
                    # Use our scheduler to adjust the learning rate
                    if not learning_rate_scheduler is None:
                        learning_rate_scheduled = learning_rate_scheduler.get_learning_rate(
                            epoch_progress=i / len(train_x),
                            total_progress=current_progress
                        )

                    heurestic_graph["lr_schedule"].append({"epoch": epoch_progress+epoch, "value": learning_rate_scheduled})
        
                    if not optimizer is None:
                        optimizer.learning_rate = learning_rate_scheduled
                        apply_delta_weights, apply_delta_biases = \
                            optimizer.update(
                                dW=current_delta_weights, 
                                db=current_delta_biases,
                                params_W=[self.layers[i].W for i in range(len(self.layers))],
                                params_b=[self.layers[i].b for i in range(len(self.layers))]
                            )
                    else:
                        # Accumulate the gradients
                        delta_weights_accumulation.append(current_delta_weights)
                        delta_biases_accumulation.append(current_delta_biases)
                        
                        delta_weights_accumulation = delta_weights_accumulation[-grad_accumulation_steps:]
                        delta_biases_accumulation = delta_biases_accumulation[-grad_accumulation_steps:]
                        delta_weights_accumulation = [[e*grad_beta_decay for e in d] for d in delta_weights_accumulation]
                        delta_biases_accumulation = [[e*grad_beta_decay for e in d] for d in delta_biases_accumulation]
                    
                        # Apply the accumulated gradients
                        for d_w_acc in delta_weights_accumulation[:-1]:
                            current_delta_weights += d_w_acc

                        for d_b_acc in delta_biases_accumulation[:-1]:
                            current_delta_biases += d_b_acc
                        
                        apply_delta_weights = [ 
                            - learning_rate_scheduled * e for e in current_delta_weights
                        ]
                        apply_delta_biases = [ 
                            - learning_rate_scheduled * e for e in current_delta_biases
                        ]

                    for i in range(len(self.layers)):
                        self.layers[i].W += apply_delta_weights[i]
                        self.layers[i].b += apply_delta_biases[i]
                    
                    # Reset the batch
                    current_delta_weights, current_delta_biases, current_batch_size = None, None, 0
                

                # Update our heuristics
                train_loss_over_time.append(loss_single)
                
                recent_loss_train = None
                if len(train_loss_over_time) > 0:
                    recent_loss_train = np.mean(train_loss_over_time[-rolling_window_size_train:])
                    heurestic_graph["training_loss"].append({"epoch": epoch_progress+epoch, "value": recent_loss_train})
                
                recent_loss_val = None
                if len(val_loss_over_time) > 0:
                    recent_loss_val = np.mean(val_loss_over_time[-rolling_window_size_val:])
                    heurestic_graph["val_loss"].append({"epoch": epoch_progress+epoch, "value": recent_loss_val})
                
                pbar.set_postfix({'train_loss': recent_loss_train, "val_loss": recent_loss_val})

            loss_final = []
            correct_samples = []
            
            # Enable inference mode for the validation phase
            self.toogle_layer_training_flags(False)

            for i in tqdm(range(len(val_x))):
                x_sample, y_true = val_x[i], val_y[i]
                
                outputs = self.forward(x_sample)
                loss = loss_func.loss(y_true, outputs)

                loss_final.append(loss)
                sample_was_correct = np.argmax(outputs) == np.argmax(y_true)
                correct_samples.append(1 if sample_was_correct else 0)

            avg_loss = np.mean(loss_final)
            accuracy = np.mean(correct_samples)
            
            time_elapsed = datetime.now() - start_time
            
            print("\n\nEvaluations completed for epoch [%4d/%4d] at %s" % (epoch+1, epochs, datetime.now()))
            print("="*50)
            print(f"Training Loss: {np.mean(train_loss_over_time[-len(train_x):])}")
            print(f"Validation loss: {avg_loss}")
            print("Validation accuracy: %3.2f%%" % ((accuracy*100)))
            # print(f"Exact accuracy: {accuracy}")
            print("="*50)
            print(f"Time elapsed: {time_elapsed}")
            print("="*50+"\n\n")
            heurestic_graph["test_loss_after_epoch"].append({"epoch": epoch, "value": avg_loss})
            heurestic_graph["test_accuracy"].append({"epoch": epoch, "value": accuracy})
        
        return heurestic_graph
    
    
    def __repr__(self):
        layer_string = [str(layer) for layer in self.layers]
        layer_string = "" if "".join(layer_string) == "" else "\n\t" + ",\n\t".join(layer_string) + "\n"
        return f"MultilayerPerceptron({layer_string})"
