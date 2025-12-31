from train import train
from config import Config
from plot import plot

cfg = Config()
fname= "land"
train(cfg, fname)
plot(cfg, fname)