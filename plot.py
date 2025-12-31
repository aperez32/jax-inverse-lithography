import matplotlib.pyplot as plt
import numpy as np
from config import Config
from matplotlib.animation import FuncAnimation
cfg = Config()


def plot(cfg, f_name):
    
    f = np.load(f"/home/jango/Coding/ILT/results/results_{f_name}.npz")

    phi = f["phi"]
    S = f["S"]
    T = f["T"]
    Ss = f["Ss"]  # (frames, N, N)

    fig, axs = plt.subplots(2, 2, figsize=(10, 4))

    n = 1.5
    t = (cfg.lam/(2*np.pi*(n-1))) * phi
    axs[1, 0].imshow(t, cmap=cfg.style, interpolation='nearest')
    axs[1, 0].set_title('Height Map')

    # dphix = phi - np.roll(phi, 1, axis=0)
    # dphiy = phi - np.roll(phi, 1, axis=1)
    # g = np.sqrt(dphix**2 + dphiy**2 + 1e-12)
    # axs[1, 1].imshow(g, cmap='viridis', interpolation='nearest')
    # axs[1, 1].set_title('Phase Gradient')
    axs[1, 1].imshow(T, cmap=cfg.style, interpolation='nearest')
    axs[1, 1].set_title('Target')


    axs[0, 0].imshow(S, cmap=cfg.style, interpolation='nearest')
    axs[0, 0].set_title('Simulated Imprint')

    im = axs[0, 1].imshow(Ss[0], cmap=cfg.style, interpolation='nearest', vmin=S.min(), vmax=S.max(), animated=True)
    axs[0, 1].set_title('Target Imprint (animated Ss)')
    axs[0, 1].set_axis_off()

    def _update(i):
        im.set_data(Ss[i])
        axs[0, 1].set_title(f"Ss frame {i+1}/{Ss.shape[0]}")
        return (im,)

    anim = FuncAnimation(fig, _update, frames=range(Ss.shape[0]), interval=150, blit=True)

    anim.save(f"/home/jango/Coding/ILT/gifs/{f_name}_full.gif")
    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    im2 = ax2.imshow(Ss[0], cmap=cfg.style, interpolation='nearest', vmin=S.min(), vmax=S.max(), animated=True)
    ax2.set_axis_off()
    ax2.set_title(f"Imprint Evolution - {f_name}")

    def _update2(i):
        im2.set_data(Ss[i])
        return (im2,)

    anim_only = FuncAnimation(fig2, _update2, frames=range(Ss.shape[0]), interval=150, blit=True)
    anim_only.save(f"/home/jango/Coding/ILT/gifs/{f_name}.gif")
    plt.tight_layout()
    plt.close(fig2)



