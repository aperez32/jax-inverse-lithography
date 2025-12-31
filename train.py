import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from config import Config
from datagen import generate_X, image_to_npz
from utils import init_phi, setup, tv, alpha_schedule


cfg = Config()
opt = optax.adam(learning_rate=cfg.lr)


@jax.jit
def step(phi, opt_state, T, t, pre):
    def loss_fn(phi):
        U = jnp.exp(-1j * phi) # Initialize phases
        U_hat = jnp.fft.fft2(U)

        Uz_hat = U_hat * pre["P"] * pre["H"] # Mask + compute at distance z
        Uz = jnp.fft.ifft2(Uz_hat)

        I = jnp.abs(Uz)**2 
        I = I / (jnp.mean(I) + 1e-8) # Normalize intensity
        S = jax.nn.sigmoid(alpha_schedule(t) * (I - pre["I_th"])) # Photoresist imprint threshold I_th. sigmoid approximates step_fxn but is differentiable
        mse = jnp.mean((S - T)**2)
        reg = pre["tv_lam"]*tv(phi)
        return mse + reg

    loss, grads = jax.value_and_grad(loss_fn)(phi)
    updates, opt_state = opt.update(grads, opt_state, phi)
    phi = optax.apply_updates(phi, updates)
    return phi, opt_state, loss


def peek(phi, t, pre):
    U = jnp.exp(-1j * phi) # Initialize phases
    U_hat = jnp.fft.fft2(U)

    Uz_hat = U_hat * pre["P"] * pre["H"] # Mask + compute at distance z
    Uz = jnp.fft.ifft2(Uz_hat)

    I = jnp.abs(Uz)**2 
    I = I / (jnp.mean(I) + 1e-8) # Normalize intensity
    S = jax.nn.sigmoid(alpha_schedule(t)  * (I - pre["I_th"])) # Photoresist imprint threshold I_th. sigmoid approximates step_fxn but is differentiable
    return S

def train(cfg, f_name):
    cfg = Config()
    key = jax.random.PRNGKey(43)
    phi, key = init_phi(key, cfg)
    pre = setup(cfg)
    opt_state = opt.init(phi)
    steps = cfg.steps
    save_every= cfg.save_every

    T = jnp.load(f"ILT/data/{f_name}.npz")["target"].astype(dtype=jnp.complex64)
    T = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(T)*pre["P"]))
    assert T.shape == (cfg.N, cfg.N)
    Ss = jnp.zeros((steps//save_every + 1, cfg.N, cfg.N))
    losses = jnp.zeros((steps,))
    for t in range(steps):
        phi, opt_state, loss = step(phi, opt_state, T,t, pre)
        losses = losses.at[t].set(loss.item())
        if t%save_every == 0:
            Ss = Ss.at[t//save_every].set(peek(phi, t, pre))


    plt.plot(jnp.arange(steps), losses)
    plt.show()

    S= peek(phi, steps-1, pre)
    Ss = Ss.at[-1].set(S)

    jnp.savez(f"/home/jango/Coding/ILT/results/results_{f_name}.npz", S=S, phi=phi, T=T, Ss=Ss)


