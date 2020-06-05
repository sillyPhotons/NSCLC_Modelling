import matplotlib.pyplot as plt
import numpy as np

plt.rc("text", usetex=True)
plt.rcParams.update({'font.size': 18,
                     'figure.autolayout': True})

def func(x):

    return np.exp(-np.log(1/0.1) * np.exp(-2*x))


def func_prime(x):

    return 2 * np.log(1/0.1) * np.exp(-np.log(1/0.1) * np.exp(-2*x)) * np.exp(-2*x)


x = np.arange(-1, 3, 0.01)
y = func(x)
y_p = func_prime(x)

plt.plot(x, y, label="$V(t)$",  linewidth=3)
plt.plot(x, y_p, label = "$V^{\prime}(t)$",  linewidth=3)
plt.plot(x, [1/np.e]*len(x), linestyle = 'dashed',linewidth=2)
plt.ylabel("Volume $[$cm$^{3}]$")
plt.xlabel("Time [days]")
plt.legend()
plt.savefig("./Figures/fig1.pdf")
