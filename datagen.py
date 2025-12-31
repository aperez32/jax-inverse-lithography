import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from config import Config

def generate_X(cfg):
    N = cfg.N
    c1 = jnp.eye(N)
    c2 = c1[:, ::-1]
    T = c1 + c2
    mask = jnp.ones((N,N))
    mask = mask.at[:80,:].set(0)
    mask = mask.at[N-80:,:].set(0)

    T =1- mask * T
    
    jnp.savez("/home/jango/Coding/ILT/data/X.npz", target=T)

def generate_gaussian_blob(cfg):
    N = cfg.N
    x = jnp.linspace(-10,10,N)
    XX, YY = jnp.meshgrid(x, x)
    T = jnp.exp(-.03*(XX**2+YY**2))
    plt.imshow(T,cmap='viridis', interpolation='nearest')
    plt.show()
    jnp.savez("/home/jango/Coding/ILT/data/gaussian_blob.npz", target=T)

def generate_two_spirals(cfg):
    N = cfg.N
    x = jnp.linspace(-10, 10, N)
    XX, YY = jnp.meshgrid(x, x, indexing="ij")

    # Polar coords
    R = jnp.sqrt(XX**2 + YY**2)
    TH = jnp.arctan2(YY, XX)

    # Spiral params (tweak these)
    a = 1.1          # sets spacing (Archimedean: r ≈ a*theta)
    sigma = 0.4     # arm thickness
    rmax = 5.0       # fade out near edge

    def spiral_intensity(theta_shift):
        th = TH + theta_shift
        ks = jnp.arange(-6, 7)  # covers enough turns for [-10,10] range
        thk = th[..., None] + 2 * jnp.pi * ks[None, None, :]
        d = R[..., None] - a * thk
        dmin = jnp.min(jnp.abs(d), axis=-1)
        return jnp.exp(-0.5 * (dmin / sigma) ** 2)

    arm1 = spiral_intensity(0.0)
    arm2 = spiral_intensity(jnp.pi)
    window = jnp.exp(-0.5 * (R / rmax) ** 8)
    T = jnp.clip((arm1 + arm2) * window, 0.0, 1.0)

    plt.imshow(T, cmap='viridis', interpolation='nearest')
    plt.show()

    jnp.savez("/home/jango/Coding/ILT/data/two_spirals.npz", target=T)    


def image_to_npz(cfg):
    N=cfg.N
    img = Image.open("/home/jango/Coding/ILT/images/land.png").convert("L")
    img = img.resize((N, N))
    T = np.array(img) / 255.0
    jnp.savez("/home/jango/Coding/ILT/data/land.npz", target=T)
    plt.imshow(T,cmap='viridis', interpolation='nearest')
    plt.show()


# cfg = Config()
# generate_X(cfg)
# # image_to_npz(cfg)
# cfg = Config()
# # generate_gaussian_blob(cfg)
# generate_two_spirals(cfg)

