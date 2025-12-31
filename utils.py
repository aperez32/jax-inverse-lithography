import jax.numpy as jnp
import jax
from config import Config
cfg = Config()
def init_phi(key, cfg, std=0.3, kcut_frac=0.1):
    N = cfg.N

    key, sub = jax.random.split(key)
    phi = jax.random.normal(sub, (N, N), dtype=jnp.float32) * std

    # Build a radial low-pass mask in FFT index space (unitless, stable)
    fx = jnp.fft.fftfreq(N)  # cycles/sample (since d not given)
    FX, FY = jnp.meshgrid(fx, fx, indexing="ij")
    R2 = FX**2 + FY**2
    mask = (R2 <= (kcut_frac * 0.5)**2).astype(jnp.float32)  # 0.5 is Nyquist in these units

    phi_hat = jnp.fft.fft2(phi)
    phi_blur = jnp.fft.ifft2(phi_hat * mask).real
    phi_blur = phi_blur - jnp.mean(phi_blur)  # zero-mean

    return phi_blur, key
    
def setup(cfg):
    N = cfg.N
    L = cfg.L
    lam = cfg.lam
    NA = cfg.NA
    z = cfg.z
    k = 2*jnp.pi/lam
    dtype = jnp.float32

    dx = L/N
    f = jnp.fft.fftfreq(N, d=dx)
    kx = 2*jnp.pi*f
    KX, KY = jnp.meshgrid(kx, kx, indexing="ij")
    K_sq = KX**2 + KY**2
    R = jnp.sqrt(K_sq)

    k = 2 * jnp.pi / lam          # radians / unit length
    r1 = k * NA                   # radian cutoff
    roll = jnp.array(0.1, dtype=dtype)
    r0 = (1.0 - roll) * r1

    P_core = (R <= r0).astype(dtype)
    m = (R > r0) & (R < r1)
    x = (R - r0) / (r1 - r0 + 1e-12)
    taper = 0.5 * (1.0 + jnp.cos(jnp.pi * x))

    P = jnp.where(m, taper.astype(dtype), P_core)
    P = jnp.where(R >= r1, jnp.array(0.0, dtype=dtype), P)


    H = jnp.exp(-1j*z/(2*k)*K_sq)
    pre = {"P":P, "H":H, "lambda": lam, "I_th":cfg.I_threshold, "alpha":cfg.alpha, "tv_lam":cfg.tv_lam}
    return pre

@jax.jit
def tv(phi, eps=1e-6):
    dx = phi - jnp.roll(phi, -1, axis=0)
    dy = phi - jnp.roll(phi, -1, axis=1)
    return jnp.mean(jnp.sqrt(dx*dx + dy*dy + eps))

# Hold then ramp
# @jax.jit
# def alpha_schedule(t, T=cfg.steps, a0=5.0, a1=30.0, hold_frac=0.5):
#     frac = t / (T - 1)
#     ramp = jnp.clip((frac - hold_frac) / (1 - hold_frac), 0.0, 1.0)
#     return a0 + (a1 - a0) * ramp

#Geometric
# @jax.jit
# def alpha_schedule(t, T=cfg.steps, a0=5.0, a1=30.0):
#     frac = t / (T - 1)
#     return a0 * (a1 / a0) ** frac

# Saturation (Sigmoid)
# @jax.jit
# def alpha_schedule(t, T=cfg.steps, a0=5.0, a1=10.0, sharp=10.0):
#     x = (t / (T - 1)) * 2 - 1          # map to [-1, 1]
#     s = jax.nn.sigmoid(sharp * x)      # [~0, ~1]
#     return a0 + (a1 - a0) * s

# constant
@jax.jit
def alpha_schedule(t, T=cfg.steps, a0=5.0, a1=10.0, sharp=10.0):
    return 5.0