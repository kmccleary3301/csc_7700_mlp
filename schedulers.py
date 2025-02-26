import numpy as np

class Scheduler():
	def __init__(self, initial_lr: float):
		self.initial_lr = initial_lr

	def get_learning_rate(
     	self, 
      	epoch_progress : float, 
       	total_progress : float
    ) -> float:
		raise NotImplementedError("Subclasses must implement this method")



class CosineScheduler(Scheduler):
    def __init__(
        self, 
        initial_lr: float,
        final_lr: float,
        max_epochs: int,
        reset_each_epoch: bool = False,
        exp_decay_factor: float = None
    ):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.max_epochs = max_epochs
        self.reset_each_epoch = reset_each_epoch
        self.exp_decay = exp_decay_factor
        self.exp_decay_interval = [1, 1/exp_decay_factor] if not exp_decay_factor is None else None
        self.exp_decay_int_size = 1 - 1/exp_decay_factor if not exp_decay_factor is None else None
        

    def get_learning_rate(
        self,
        epoch_progress : float,
        total_progress : float,
        **kwargs
    ) -> float:
        """
        We want it to be a cosine function, but also optionally decay exponentially.
        """
        
        cos_input = epoch_progress if self.reset_each_epoch else total_progress
        
        cap_factor = self.initial_lr
        if not self.exp_decay is None:
            cap_factor *= np.exp(total_progress * np.log(self.exp_decay))
        
        
        cos_value = self.final_lr + 0.5 * (cap_factor - self.final_lr) * \
            (1 + np.cos(np.pi * cos_input))
        
        
        return cos_value