from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class Config:

    N: int = 1024 # Grid size
    L: float = 1.0 # length in meters
    lam: int = 0.02 #wavelength
    z: float = 0.4 #distance to wafer  # initial 0.3
    NA: float = 8.0 # frequency mask to filter destructive components # typical -> 0.3
    I_threshold: float = 1.0 # threshold to imprint on photoresist
    alpha: float = 1.0 # scheduled
    
    steps : int =  200# train steps
    save_every: int = 10
    lr: float = 1e-3 # learning rate


    style: str =  'magma'
    tv_lam: float = 2.0 # regularization weight # -> typical 0.6