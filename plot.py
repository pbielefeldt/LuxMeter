import numpy as np
import matplotlib.pyplot as plt
import scipy as sy
from scipy import optimize

d,e = np.genfromtxt("./data/data_lima_beide_masten.csv", skip_header=1, unpack=True)

def exponential(x, a, t):
    return a*np.exp(x/t)

start = [800, -80]
p_op, cov_mat = optimize.curve_fit(exponential, d, e, p0=start, sigma=0.2*e+1)

xplot = np.linspace(min(d), max(d), 1000)

plt.errorbar(d,e, linewidth=0, marker="x")
plt.plot(xplot, exponential(xplot, *p_op))
plt.grid()
plt.yscale("log")
plt.xlabel("distanz / m")
plt.ylabel("bel. stärke / lx")
plt.savefig("plot.png")
