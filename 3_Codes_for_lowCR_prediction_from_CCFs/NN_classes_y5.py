import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers

from sklearn.model_selection import KFold, ParameterGrid, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import gc
import ctypes

import warnings
import time
from itertools import product
import matplotlib.pyplot as plt

def weighted_kl_divergence(target_weights):
    """
    KL(p_true || p_pred) = sum( p_true * log(p_true / p_pred) )
    weighted per-regime, then averaged over samples.
    
    Parameters:
    -----------
    target_weights : list of floats
        Weight for each regime, e.g. [1, 1, 1, 1, 0.3] to down-weight the 5th.
    """
    weights = tf.constant(target_weights, dtype=tf.float32)
 
    def loss(y_true, y_pred):
        # Clip both to avoid log(0) or division by zero
        y_true = tf.clip_by_value(y_true, 1e-7, 1.0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
 
        # Element-wise KL:  p * log(p / q)
        kl = y_true * tf.math.log(y_true / y_pred)
 
        # Apply per-regime weights  →  shape: (batch, n_regimes)
        weighted_kl = kl * weights[tf.newaxis, :]
 
        # Sum over regimes, mean over batch
        return tf.reduce_mean(tf.reduce_sum(weighted_kl, axis=-1))
 
    loss.__name__ = 'weighted_kl_divergence'
    return loss

class NeuralNetworkRegressor:
    """
    A TensorFlow-based neural network for regression tasks with early stopping.

    NEW (softmax / KL mode):
        - output activation = softmax  (guarantees sum-to-1)
        - loss = per-regime weighted KL divergence
        - output_dim should be set to 5 (4 low-cloud + 1 other)
        - other_regime_weight controls the loss weight on the 5th output
    
    Parameters:
    -----------
    input_dim : int, default=10
        Number of input features
    output_dim : int, default=4
        Number of output features
    hidden_units : int, default=15
        Number of units in the first hidden layer
    hidden_units_2 : int, default=None
        Number of units in the second hidden layer (None for single hidden layer)
    learning_rate : float, default=0.001
        Learning rate for the optimizer
    random_state : int, default=42
        Random seed for reproducibility
    """
    
    def __init__(self, input_dim=10, output_dim=5, hidden_units=(15,None),l2_reg_str=0.0001,
                 learning_rate=0.001, random_state=42,LR_scheduler=None,
                 other_regime_weight=0.2,  
                 target_names=['C1','C2','C3','C4','Other']):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units[0]
        self.hidden_units_2 = None if len(hidden_units)==1 else hidden_units[1:]  # Second hidden layer (None means single layer)
        self.learning_rate = learning_rate
        self.LR_scheduler= LR_scheduler
        self.l2_reg_str= l2_reg_str
        self.random_state = random_state
        self.target_names= target_names
        self.other_regime_weight = other_regime_weight
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        #np.random.seed(random_state)
        
        # Initialize components
        self.model = None
        self.history = None
        self.is_fitted = False        
        self._build_model()
    
    def _build_model(self):
        """Build the neural network architecture."""
        # Define a small L2 value (tune this later!)
        #print(self.l2_reg_str); sys.exit()
        l2_reg = regularizers.l2(self.l2_reg_str) #0.01)
        layers_list = [
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(self.hidden_units,  name='hidden_layer_1',
                         #kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.random_state),
                         kernel_initializer=tf.keras.initializers.HeNormal(seed=self.random_state), 
                         use_bias=True,
                         bias_initializer='zeros', #tf.keras.initializers.Zeros(),
                         kernel_regularizer=l2_reg, # <--- Added L2
                         ), #activation='relu',
            layers.LeakyReLU(alpha=0.1), #Activation('swish'), # 
            layers.Dropout(0.1),
        ]
        
        # Add second hidden layer if specified
        if self.hidden_units_2 is not None and self.hidden_units_2[0] is not None:
            for i,hu in enumerate(self.hidden_units_2): 
                layers_list.append(
                    layers.Dense(hu,  name=f'hidden_layer_{i+2}',
                             #kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.random_state+3),
                             kernel_initializer=tf.keras.initializers.HeNormal(seed=self.random_state+1), 
                             #kernel_initializer= 'he_normal',
                             use_bias=True,
                             bias_initializer='zeros', #tf.keras.initializers.Zeros(),
                             kernel_regularizer=l2_reg, # <--- Added L2
                             ) #activation='relu',
                )
                layers_list.append( layers.LeakyReLU(alpha=0.1)) #layers.Activation('swish')) #
                layers_list.append( layers.Dropout(0.1))
        
        # Add output layer
        clim_rfo = np.array([0.24, 0.24, 0.24, 0.24, 0.04])  # your actual values
        log_prior_bias = np.log(clim_rfo).astype('float32').tolist()

        layers_list.append(
            layers.Dense(self.output_dim, activation='softmax', name='output_layer',
                         #kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.random_state+7),
                         kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01,seed=self.random_state), 
                         #kernel_initializer= 'zeros', # 'he_normal',
                         use_bias=True,
                         bias_initializer=tf.keras.initializers.Constant(log_prior_bias))
        )
        
        self.model = keras.Sequential(layers_list)

        # ── Loss: weighted KL divergence ─────────────────────────── #
        # Weights: 1.0 for each low-cloud regime, reduced for "other"
        n_main = self.output_dim - 1  # 4 low-cloud regimes
        target_weights = [1.0] * n_main + [self.other_regime_weight,]
        loss_fn = weighted_kl_divergence(target_weights)
        
        # Compile the model
        if self.LR_scheduler is not None:
            LR= self.LR_scheduler
        else:
            LR= self.learning_rate
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LR)
        
        self.model.compile(
            optimizer=optimizer,
            #loss=keras.losses.Huber(delta=1.0), # delta=1.0 is standard // #
            loss=loss_fn, #'mse',
            metrics=['mae',] #'r2_score']
        )
    
    def fit(self, X, y, validation_split=0.2, epochs=1000, batch_size=32, callback2add=[],
            patience=10, min_delta=1e-4, restore_best_weights=True, verbose=1,
            validation_data=None, sample_weight=None, val_sample_weight=None):
        """
        Train the neural network with early stopping.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, input_dim)
            Training input features
        y : array-like, shape (n_samples, output_dim)
            Training target values
        validation_split : float, default=0.2
            Fraction of training data to use for validation
        epochs : int, default=1000
            Maximum number of training epochs
        batch_size : int, default=32
            Batch size for training
        patience : int, default=20
            Number of epochs to wait before early stopping
        min_delta : float, default=1e-4
            Minimum change to qualify as an improvement
        restore_best_weights : bool, default=True
            Whether to restore weights from the best epoch
        verbose : int, default=1
            Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)
        
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        ## Convert to numpy arrays
        #X = np.array(X)
        #y = np.array(y)
        
        # Validate input shapes
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} input features, got {X.shape[1]}")
        if y.shape[1] != self.output_dim:
            raise ValueError(f"Expected {self.output_dim} output features, got {y.shape[1]}")

        self.patience= patience
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=restore_best_weights,
            verbose=1 #verbose
        )
        progress_callback = CustomProgressCallback(
            display_metrics=['loss', 'mae','val_loss', 'val_mae',],
            display_every=1,  # Display every epoch            
        )
        callback= [early_stopping,progress_callback,]+callback2add
        # Train the model
        #if len(validation_data)==2:
        if validation_data is not None:
            X_val, y_val = validation_data
            if val_sample_weight is not None:
                val_data = (X_val, y_val, val_sample_weight)
            else:
                val_data = (X_val, y_val)
                
            self.history = self.model.fit(
                X, y,sample_weight=sample_weight,
                validation_data=val_data,
                epochs=epochs, batch_size=batch_size,
                callbacks=callback, verbose=0 #verbose
            )
        else:
            print('Validation Split Radio=',validation_split)
            self.history = self.model.fit(
                X, y,sample_weight=sample_weight,
                validation_split=validation_split,
                epochs=epochs, batch_size=batch_size,
                callbacks=callback, verbose=0 #verbose
            )
                
        self.is_fitted = True
        return self
    
    def predict(self, X, verbose=1):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, input_dim)
            Input features for prediction
        
        Returns:
        --------
        predictions : ndarray, shape (n_samples, output_dim)
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        y_pred = self.model.predict(X,verbose=verbose)
        
        return y_pred
    
    def evaluate(self, X, y, verbose=1,scatter_plot=True):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, input_dim)
            Test input features
        y : array-like, shape (n_samples, output_dim)
            True target values
        
        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        predictions = self.predict(X,verbose)
        
        # Calculate metrics
        mse = np.mean((y - predictions) ** 2)
        mae = np.mean(np.abs(y - predictions))
        rmse = np.sqrt(mse)
        
        # R² score and MAE for each output
        r2_scores = []; mae_scores=[]; corrs=[]
        for i in range(self.output_dim):
            ss_res = np.sum((y[:, i] - predictions[:, i]) ** 2)
            ss_tot = np.sum((y[:, i] - np.mean(y[:, i])) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            r2_scores.append(r2)
            mae_scores.append(np.mean(np.abs(y[:,i] - predictions[:,i])))
            corrs.append(np.corrcoef(y[:,i],predictions[:,i])[0,1])

        if scatter_plot:
            self._plot_regression_results(y, predictions, self.target_names)
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_scores': r2_scores,
            'mae_scores': mae_scores,
            'mean_r2': np.mean(r2_scores),
            'corrs': corrs,
        }

    def _plot_regression_results(self,y_true, y_pred, target_names):
        """
        Creates a 2x2 scatter plot for 4 regression outputs.
        y_true/y_pred: shape (n_samples, 4)
        target_names: list of 4 strings
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import r2_score
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten() # Flatten 2x2 into a list of 4
    
        for i in range(min(4,y_true.shape[1])):
            ax = axes[i]
            actual = y_true[:, i]
            pred = y_pred[:, i]
        
            # Calculate stats for the title
            r2 = r2_score(actual, pred)
            corr = np.corrcoef(actual, pred)[0, 1]
            mse = np.mean((actual - pred) ** 2)
            mae = np.mean(np.abs(actual - pred))
            rmse = np.sqrt(mse)
        
            # Plot the points (alpha=0.3 helps see density with 15k samples)
            ax.scatter(actual, pred, alpha=0.3, s=10, color='royalblue', label='Data')
        
            # Add the "Perfect Prediction" diagonal line
            mn, mx = actual.min(), actual.max()
            ax.plot([mn, mx], [mn, mx], color='red', linestyle='--', lw=2, label='Perfect (y=x)')
        
            # Formatting
            ax.set_title(f"{target_names[i]}\n$R^2$: {r2:.3f} | Corr: {corr:.3f} | RMSE: {rmse:.3f} | MAE: {mae:.3f}" )
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plt.show()
        return

    def plot_training_history(self, sup_tit, out_fn=None, figsize=(12, 4), show_val=True):
        """
        Plot training and validation loss curves.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 4)
            Figure size for the plots
        """
        if self.history is None:
            raise ValueError("Model must be trained before plotting history")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        plt.suptitle(sup_tit,fontsize=16,y=0.92,va='bottom') #,stretch='semi-condensed',x=0.1,ha='left')
        
        tot_len= len(self.history.history['loss'])
        xx= np.arange(tot_len)+1
        set_nm=  'Validation'
        # Plot loss
        ax1.plot(xx[1:],self.history.history['loss'][1:], label='Training Loss')
        if show_val:
            ax1.plot(xx[1:],self.history.history['val_loss'][1:], label=set_nm+' Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        ax1.axvline(x=tot_len-self.patience,ls='--',c='r',alpha=0.7)
        ymin,ymax= ax1.get_ylim()
        ax1.set_ylim(0.,ymax)
        
        # Plot MAE
        ax2.plot(xx[1:],self.history.history['mae'][1:], label='Training MAE')
        if show_val:
            ax2.plot(xx[1:],self.history.history['val_mae'][1:], label=set_nm+' MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        ax2.axvline(x=tot_len-self.patience,ls='--',c='r',alpha=0.7)
        ymin,ymax= ax2.get_ylim()
        ax2.set_ylim(0.,ymax)
            
        plt.tight_layout()        
        if out_fn is not None:
            plt.savefig(out_fn,bbox_inches='tight',dpi=100)
            print(out_fn)
        else:
            plt.show()
        return

    def plot_training_history_detail(self, sup_tit, out_fn=None, figsize=(12, 4)):
        """
        Plot training and validation loss curves.

        Parameters:
        -----------
        figsize : tuple, default=(12, 4)
            Figure size for the plots
        """
        if self.history is None:
            raise ValueError("Model must be trained before plotting history")

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        plt.suptitle(sup_tit,fontsize=16,y=0.92,va='bottom') #,stretch='semi-condensed',x=0.1,ha='left')

        tot_len= len(self.history.history['loss'])
        xx= np.arange(tot_len)+1
        set_nm=  'Validation'

        # Plot loss: o_target_1_loss, val_o_target_1_loss
        for i, tgt in enumerate(self.target_names):
            ax1= axes.flatten()[i]
            loss_nm= f'o_{tgt}_loss'
            val_loss_nm= f'val_o_{tgt}_loss'
            val_loss= self.history.history[val_loss_nm]
            tr_loss= self.history.history[loss_nm]
            ax1.plot(xx[1:],tr_loss[1:], label='Training Loss')
            ax1.plot(xx[1:],val_loss[1:], label=set_nm+' Loss')
            ax1.set_title('Model Loss: '+tgt)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            ax1.axvline(x=tot_len-self.patience,ls='--',c='0.4',alpha=0.7)
            ax1.axvline(x=xx[np.argmin(val_loss)],ls='--',c='r',alpha=0.7)
            ymin,ymax= ax1.get_ylim()
            ax1.set_ylim(0.,ymax)

        plt.tight_layout()
        if out_fn is not None:
            plt.savefig(out_fn,bbox_inches='tight',dpi=100)
            print(out_fn)
        else:
            plt.show()
        return

    def get_model_summary(self):
        """Get a summary of the model architecture."""
        if self.model is None:
            raise ValueError("Model not built yet")
        return self.model.summary()
    
    def save_model(self, filepath):
        """Save the trained model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        self.model.save(filepath)
    
    def load_model(self, filepath, compile=False):
        """Load a trained model from disk."""
        self.model = keras.models.load_model(
            filepath,
            custom_objects={'weighted_kl_divergence': weighted_kl_divergence(
                [1.0] * (self.output_dim - 1) + [self.other_regime_weight]
            )}, compile=compile,
        )
        self.is_fitted = True


    # ------------------------------------------------------------------ #
    #  Helper: append "other" column to targets   (call before fit/eval)  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def append_other_regime(y_4col):
        """
        Given y with 4 columns (low-cloud RFOs), append a 5th column = 1 - sum.
        Clips to [0, 1] for safety.
        
        Parameters:
        -----------
        y_4col : ndarray, shape (n_samples, 4)
        
        Returns:
        --------
        y_5col : ndarray, shape (n_samples, 5)
        """
        other = 1.0 - np.sum(y_4col, axis=1, keepdims=True)
        other = np.clip(other, 0.0, 1.0)
        return np.concatenate([y_4col, other], axis=1)
    

def create_cosine_schedule(initial_lr, final_lr, total_steps ,steps_per_epoch=0, warmup_epochs=0):
    """Create cosine annealing schedule with optional warmup."""
    
    # Calculate total steps
    # You'll need: steps_per_epoch = len(X_train) // batch_size
    #steps_per_epoch = sample_size // batch_size
    #total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    alpha= final_lr/initial_lr
    
    if warmup_epochs > 0:
        # Cosine decay with linear warmup
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_steps - warmup_steps,
            alpha=alpha,  # Minimum LR as fraction of initial_lr (0 = goes to 0)
        )
        
        # Add warmup
        lr_schedule = keras.optimizers.schedules.LinearWarmup(
            after_warmup_lr_sched=lr_schedule,
            warmup_steps=warmup_steps,
            warmup_learning_rate=0.0
        )
    else:
        # Simple cosine decay
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_steps,
            alpha=alpha,  # Goes to 0 at the end
        )
    
    return lr_schedule

class CustomProgressCallback(keras.callbacks.Callback):
    """
    Custom callback to display only selected metrics during training.
    Reduces clutter while still tracking all metrics in history.
    """
    
    def __init__(self, display_metrics=None, display_every=1):
        """
        Args:
            display_metrics: List of metric names to display (e.g., ['loss', 'val_loss'])
                           If None, displays loss and val_loss only
            display_every: Display progress every N epochs (default=1, every epoch)
        """
        super().__init__()
        self.display_metrics = display_metrics or ['loss', 'val_loss']
        self.display_every = display_every
            
    def on_epoch_begin(self, epoch, logs=None):
            #print(f"Starting epoch {epoch + 1}...")
            self.epoch_start_time = tf.timestamp()
            
    def on_epoch_end(self, epoch, logs=None):
        """Print selected metrics at the end of each epoch."""
        if (epoch + 1) % self.display_every != 0:
            return

        epoch_end_time = tf.timestamp()
        epoch_duration = epoch_end_time - self.epoch_start_time
        logs = logs or {}
        current_lr = self._get_learning_rate()
        
        # Build output string with only selected metrics
        output = f"Epoch {epoch + 1:3d}"
        #print(logs.keys())#; sys.exit()
        for metric in self.display_metrics:
            if metric in logs:
                output += f" - {metric}:{logs[metric]:.5f}"
        output+= f' - lr:{current_lr:.3e}'
        output+= f' - {epoch_duration.numpy():.2f}s'
        print(output)

    def _get_learning_rate(self):
        """
        Get the current learning rate from the optimizer.
        Handles both fixed and scheduled learning rates.
        """
        optimizer = self.model.optimizer
        
        # Method 1: Try to get lr directly (works for most cases)
        if hasattr(optimizer, 'learning_rate'):
            lr = optimizer.learning_rate
        elif hasattr(optimizer, 'lr'):
            lr = optimizer.lr
        else:
            return 0.0

        # If it's a Variable or Tensor, get the actual value
        if isinstance(lr, (tf.Variable, tf.Tensor)):
            return float(keras.backend.get_value(lr))
        # If it's a learning rate schedule, it will be called automatically
        elif hasattr(lr, '__call__'):
            # For learning rate schedules
            return float(keras.backend.get_value(lr(optimizer.iterations)))
        else:
            return float(lr)

class LearningRateLogger(tf.keras.callbacks.Callback):
    """Track all LR reductions but provide selective summary for CV results."""
    
    def __init__(self):
        super().__init__()
        self.lr_reductions = []  # Full record of all reductions
        self.previous_lr = None
        self.initial_lr = None
        #self.final_lr = None
        self.epoch_lr_history = {}  # Track LR at each epoch
        
    def on_train_begin(self, logs=None):
        self.initial_lr = float(self.model.optimizer.learning_rate)
        self.previous_lr = self.initial_lr
        
    def on_epoch_end(self, epoch, logs=None):
        current_lr = float(self.model.optimizer.learning_rate)

        # Record LR for this epoch
        self.epoch_lr_history[epoch + 1] = current_lr
        
        # Check for LR reduction and record ALL reductions
        if self.previous_lr is not None and current_lr < self.previous_lr:
            reduction_info = {
                'epoch': epoch + 1,
                'old_lr': self.previous_lr,
                'new_lr': current_lr,
                'reduction_factor': current_lr / self.previous_lr,
                'val_loss': logs.get('val_loss', None) if logs else None
            }
            self.lr_reductions.append(reduction_info)
            
            # Print for first reduction only
            if True: #len(self.lr_reductions) == 1:
                print(f"    📉 LR reduction #{len(self.lr_reductions)} at epoch {epoch + 1}: {self.previous_lr:.6f} → {current_lr:.6f}")
        
        self.previous_lr = current_lr
        #self.final_lr = current_lr

    def get_lr_at_epoch(self, epoch):
        """Get learning rate at specific epoch."""
        return self.epoch_lr_history.get(epoch, None)
    
    def get_full_record(self):
        """Get complete record of all LR reductions."""
        return {
            'initial_lr': self.initial_lr,
            #'final_lr': self.final_lr,
            'total_reductions': len(self.lr_reductions),
            'reductions': self.lr_reductions.copy(),
            'epoch_lr_history': self.epoch_lr_history.copy(),
        }
    
    def get_cv_summary(self, best_epoch=None):
        """Get only first reduction epoch and final LR for CV results."""
        first_reduction_epoch = None
        if self.lr_reductions:
            first_reduction_epoch = self.lr_reductions[0]['epoch']

        lr_at_best_epoch = None
        if best_epoch is not None:
            lr_at_best_epoch = self.get_lr_at_epoch(best_epoch)
            
        return {
            'first_reduction_epoch': first_reduction_epoch,
            'lr_at_best_epoch': lr_at_best_epoch,
        }
    
# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

class CrossValidationTuner:
    """
    Cross-validation hyperparameter tuning for NeuralNetworkRegressor.
    """
    
    def __init__(self, model_class, cv_folds=5, random_state=42):
        """
        Initialize the cross-validation tuner.
        
        Parameters:
        -----------
        model_class : class
            The NeuralNetworkRegressor class
        cv_folds : int, default=5
            Number of cross-validation folds
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.model_class = model_class
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results = []
        self.best_params = None
        self.best_score = float('inf')
        
    def tune_hyperparameters(self, X, y, param_grid, groups=[], #y_weights=[],
                             sample_weight=None, scoring='mse', callback2add=[],
                             max_epochs=250, verbose=0):
        """
        Perform cross-validation hyperparameter tuning.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples, n_targets)
            Training targets
        param_grid : dict
            Dictionary of hyperparameters to test
        scoring : str, default='mse'
            Scoring metric ('mse', 'mae', 'r2')
        max_epochs : int, default=200
            Maximum epochs for training
        verbose : int, default=0
            Verbosity level
        
        Returns:
        --------
        results_df : pandas.DataFrame
            DataFrame containing all results
        """
        print(f"Starting hyperparameter tuning with {self.cv_folds}-fold CV...")
        print(f"Total parameter combinations: {len(list(ParameterGrid(param_grid)))}")
        
        # Setup cross-validation
        if len(groups)>0:
            kf= GroupKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            #kf_split= kf.split(X,groups=groups)
            print('**  Group KFold initiated')
        else:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            #kf_split= kf.split(X)
            
        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        
        start_time = time.time()
        
        for i, params in enumerate(param_combinations):
            #print(params) #; sys.exit()
            model_params, training_params = self._parse_parameters(params)
            if model_params['learning_rate']<0.001 and training_params['batch_size']>128:
                continue
            elif model_params['learning_rate']>0.0031 and training_params['batch_size']<50:
                continue
                
            #if verbose > 0:
            print(f"\nTesting combination {i+1}/{len(param_combinations)}: {model_params},{training_params}")
                    
            fold_scores = []
            fold_times = []
            fold_epochs = []
            fold_val_loss= []
            # Cross-validation loop
            for fold, (train_idx, val_idx) in enumerate(kf.split(X,groups=groups)):
                fold_start = time.time()
                model= None
                # Split data
                #X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                #y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                try:
                    tf.keras.backend.clear_session()
                    gc.collect()
                    # Pre-fold cleanup (no data slicing yet)
                    #self._aggressive_memory_cleanup()
                    time.sleep(0.1)
                    
                    # Create and train model
                    model = self.model_class(
                        input_dim=X.shape[1],
                        output_dim=y.shape[1],
                        #y_weights= y_weights,
                        **model_params
                    )

                    # Create LR logger and add to callbacks
                    lr_logger = LearningRateLogger()
                    all_callbacks = callback2add + [lr_logger,]
        
                    # Train with early stopping
                    model.fit(
                        X[train_idx], y[train_idx],
                        #X_train_fold, y_train_fold,
                        #validation_split=0.15,  # Use small validation for early stopping
                        validation_data=(X[val_idx],y[val_idx]), #(X_val_fold, y_val_fold),
                        epochs=max_epochs, verbose=verbose,
                        callback2add=all_callbacks,
                        **training_params
                    )
                    time.sleep(0.2)
                    
                    # Predict and score
                    y_pred = model.predict(X[val_idx],verbose=verbose)
                    if model_params['other_regime_weight']<1.:
                        score = self._calculate_score(y[val_idx][:,:-1], y_pred[:,:-1], scoring) #, y_weights=y_weights)
                    else:
                        score = self._calculate_score(y[val_idx], y_pred, scoring) #, y_weights=y_weights)
                    fold_scores.append(score)
                    # Record the epoch when training terminated
                    if hasattr(model, 'history') and model.history is not None:
                        #last_epoch = len(model.history.history['loss'])-training_params['patience']
                        val_losses = model.history.history["val_loss"]
                        best_epoch = int(np.argmin(val_losses) + 1)
                        best_val_loss= val_losses[best_epoch-1]
                        #model.history = None  # Clear immediately
                        #del val_losses
                    else:
                        best_epoch=best_val_loss=9999

                    # Get LR info - full record available, but extract only what we need for CV
                    cv_lr_summary = lr_logger.get_cv_summary(best_epoch=best_epoch)
                    #full_lr_record = lr_logger.get_full_record()  # Available if needed for debugging
        
                    fold_val_loss.append(best_val_loss)
                    fold_epochs.append(best_epoch)
                    print(f'Fold={fold}: score={score}, best_epoch={best_epoch}, val_loss={fold_val_loss[-1]}, first_lr_reduction_epoch={cv_lr_summary["first_reduction_epoch"]}, lr_at_best={cv_lr_summary["lr_at_best_epoch"]:.3e}')
                    
                    # Store only CV summary data
                    if not hasattr(self, 'fold_lr_data'):
                        self.fold_lr_data = []
        
                    self.fold_lr_data.append({
                        'fold': fold,
                        'first_reduction_epoch': cv_lr_summary['first_reduction_epoch'],
                        'lr_at_best_epoch': cv_lr_summary['lr_at_best_epoch'],
                    })
        
                    # Optional: store full record if you need detailed analysis later
                    # You can uncomment this if you want access to complete reduction history
                    # if not hasattr(self, 'fold_lr_full_records'):
                    #     self.fold_lr_full_records = []
                    # self.fold_lr_full_records.append(full_lr_record)
        
                    del y_pred
                    
                except Exception as e:
                    #if verbose > 0:
                    print(f"Error in fold {fold}: {e}")
                    fold_scores.append(float('inf'))
                    fold_epochs.append(9999)
                    fold_val_loss.append(9999)

                    # Add empty data for failed folds
                    if not hasattr(self, 'fold_lr_data'):
                        self.fold_lr_data = []
                    self.fold_lr_data.append({
                        'fold': fold,
                        'first_reduction_epoch': None,
                        'lr_at_best_epoch': None
                    })

                finally:
                    # Model cleanup (same as before)
                    try:
                        if model is not None:
                            del model
                    except:
                        pass
        
                    self._aggressive_memory_cleanup()
                    fold_times.append(time.time() - fold_start)
                    time.sleep(0.5)
                    
            if hasattr(self, 'fold_lr_data'):
                # Extract data for CV results
                first_reduction_epochs = [data['first_reduction_epoch'] 
                                          for data in self.fold_lr_data 
                                          if data['first_reduction_epoch'] is not None]
    
                lr_at_best_epoch = [data['lr_at_best_epoch'] 
                             for data in self.fold_lr_data 
                             if data['lr_at_best_epoch'] is not None]

    
            # Calculate mean and std across folds
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            mean_time = np.mean(fold_times)
            mean_epochs = np.mean(fold_epochs)
            std_epochs = np.std(fold_epochs)
            min_epoch = np.min(fold_epochs)
            max_epoch = np.max(fold_epochs)
            mean_val_loss= np.mean(fold_val_loss)
            first_lr_reduction_epoch_mean= np.mean(first_reduction_epochs) if first_reduction_epochs else None
            lr_at_best_epoch_mean= np.mean(lr_at_best_epoch) if lr_at_best_epoch else None
            print(fold_epochs,mean_score,np.round(fold_scores,3))
            
            # Store results
            result = {
                **params,
                f'mean_{scoring}': mean_score,
                f'std_{scoring}': std_score,
                'mean_fit_time': mean_time,
                'mean_epochs': mean_epochs,
                'std_epochs': std_epochs,
                'min_epochs': min_epoch,
                'max_epochs': max_epoch,
                'fold_scores': fold_scores,
                'fold_epochs': fold_epochs,
                'mean_val_loss': mean_val_loss,
                'first_lr_reduction_epoch_mean': first_lr_reduction_epoch_mean,
                'lr_at_best_epoch_mean': lr_at_best_epoch_mean,
            }
            self.results.append(result)
            
            # Update best parameters
            if mean_score < self.best_score:
                self.best_score = mean_score
                self.best_params = params.copy()
            
            if verbose > 0:
                print(f"Mean {scoring}: {mean_score:.5f} (+/- {std_score:.5f})")
                print(f"Mean epochs: {mean_epochs:.1f} (+/- {std_epochs:.1f}) [max: {max_epochs}]")
            
        total_time = time.time() - start_time
        print(f"\nTuning completed in {total_time:.2f} seconds")
        print(f"Best {scoring}: {self.best_score:.5f}")
        print(f"Best parameters: {self.best_params}")
        
        ## Convert results to DataFrame
        #results_df = pd.DataFrame(self.results)
        return self.results #results_df
    
    def _parse_parameters(self, params):
        """
        Parse and separate model constructor parameters from training parameters.
        
        Parameters:
        -----------
        params : dict
            Parameter dictionary from hyperparameter grid
            
        Returns:
        --------
        model_params : dict
            Parameters for model constructor
        training_params : dict
            Parameters for model.fit() method
        """
        # Define which parameters go to model constructor vs training
        MODEL_CONSTRUCTOR_PARAMS = {
            'hidden_units', 'learning_rate', 'y_weights', 'l2_reg_str','random_state',
            'other_regime_weight',
            #'input_dim', 'output_dim'  # These are set automatically, but included for completeness
        }
        
        TRAINING_PARAMS = {
            'patience', 'min_delta', 'restore_best_weights', 'batch_size',
            'validation_split',  'sample_weight', #'epochs', 'shuffle',
        }
        
        model_params = {}
        training_params = {}
        
        for key, value in params.items():
            if key in MODEL_CONSTRUCTOR_PARAMS:
                model_params[key] = value
            elif key in TRAINING_PARAMS:
                training_params[key] = value
            else:
                # For unknown parameters, try to infer or give a warning
                print(f"Warning: Unknown parameter '{key}' - assuming it's a training parameter")
                training_params[key] = value
        
        return model_params, training_params
    
    def _calculate_score(self, y_true, y_pred, scoring,y_weights=None,):
        """Calculate the specified scoring metric."""
        #if y_weights is None:
        #    props= dict(multioutput='uniform_average')
        #else:
        #    props= dict(multioutput=y_weights)
        props= {}
        if scoring == 'mse':
            return mean_squared_error(y_true, y_pred, **props)
        elif scoring == 'mae':
            return mean_absolute_error(y_true, y_pred, **props)
        elif scoring == 'r2':
            return -r2_score(y_true, y_pred, **props)  # Negative for minimization
        else:
            raise ValueError(f"Unknown scoring metric: {scoring}")

    def _aggressive_memory_cleanup(verbose=False):
        """Comprehensive memory cleanup for TensorFlow."""
    
        if verbose:
            print("    Performing aggressive cleanup...")
    
        # 1. Clear Keras/TensorFlow session
        tf.keras.backend.clear_session()
    
        # 2. Reset TensorFlow graph (if available)
        try:
            tf.compat.v1.reset_default_graph()
        except:
            pass
    
        # 3. Multiple garbage collection passes
        for _ in range(3):
            gc.collect()
    
        # 4. Force Python to release memory to OS (if available)
        try:
            #import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass
    
        if verbose:
            print("    Cleanup completed.")
        return
        
    def get_best_model(self, X, y, max_epochs=200):
        """
        Train and return the best model with optimal hyperparameters.
        
        Parameters:
        -----------
        X : array-like
            Full training features
        y : array-like
            Full training targets
        max_epochs : int, default=200
            Maximum epochs for final training
        
        Returns:
        --------
        model : NeuralNetworkRegressor
            Trained model with best parameters
        """
        if self.best_params is None:
            raise ValueError("Must run tune_hyperparameters first")

        # Parse best parameters
        model_params, training_params = self._parse_parameters(self.best_params)
        
        print("Training final model with best parameters...")
        model = self.model_class(
            input_dim=X.shape[1],
            output_dim=y.shape[1],
            **model_params
        )
        
        model.fit(
            X, y,
            validation_split=0.2,
            epochs=max_epochs,
            verbose=1,
            **training_params
        )
        
        return model

