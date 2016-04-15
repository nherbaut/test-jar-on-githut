#from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def eq31(alpha=0, s=0, beta=0, r=0):
    return alpha * s + beta * (s -  r)


def eq32(alpha=0, s=0, beta=0, l=0):
    return alpha * s + beta * (s + l)


def eq33(alpha=0, s=0, beta=0, d=0):
    return alpha * s + beta * (s + d)


def eq41(alpha=0, beta=0, t=0, s=0, r=0, l=0, d=0):
    return min(
        eq31(alpha, s, beta, r) + (alpha - beta) * t,
        eq32(alpha, s, beta, l) + (alpha - beta) * t,
        eq33(alpha, s, beta, d) + (alpha - beta) * t)


def eq42(alpha=0, beta=0, t=0, s=0, r=0, l=0, d=0):
    return min(
        eq31(alpha, s, beta, r) + (l - t) * beta,
        eq32(alpha, s, beta, l) + (l - t) * beta,
        eq33(alpha, s, beta, d) + (l - t) * beta)


def g(alpha=0, beta=0, t=0, s=0, r=0, l=0, d=0):
    return min(eq41(alpha, beta, t, s, r, l, d), eq42(alpha, beta, t, s, r, l, d))


plt.title('Optimality for UC1')

alpha = 0.80
beta = 0.10
t = 4.5*10**10  * 200000 / 30 * 0.0563 / 60 / 60.0 / 10 ** 6 * 0.63 * 8
s = 80000
r = 200000
l = 100000
d = 550000
premium_cdn_pricing = 100000

#ss = np.arange(s / 40.0, s * 4, (s * 4 - s / 40.0) / 200.0)
ss = np.logspace(np.log(s*4),np.log(s/1000.0) , 500,base=np.e)
# ds = np.arange(d/4.0,d*2,d*2/200.0)
ts = np.logspace(np.log(t*4),np.log(t/1000.0) , 500,base=np.e)
X, Y = np.meshgrid(ss, ts)

theCM = cm.get_cmap()
theCM._init()
theCM._lut[:-3, -1] = 0.2

Z = np.empty((len(ss), len(ts)))
Zko = np.empty((len(ss), len(ts)))
for i in range(0, len(X)):
    for j in range(0, len(Y)):
        if (g(alpha=alpha, beta=beta, t=Y[i][j], s=X[i][j], r=r, l=l, d=d) <= premium_cdn_pricing):
            Z[i][j] = g(alpha=alpha, beta=beta, t=Y[i][j], s=X[i][j], r=r, l=l, d=d)
            Zko[i][j] = None
        else:
            Z[i][j] = g(alpha=alpha, beta=beta, t=Y[i][j], s=X[i][j], r=r, l=l, d=d)
            Zko[i][j] = g(alpha=alpha, beta=beta, t=Y[i][j], s=X[i][j], r=r, l=l, d=d)




Zpremium = np.empty((len(ss), len(ts)))
for i in range(0, len(X)):
    for j in range(0, len(Y)):
        Zpremium[i][j] = premium_cdn_pricing


theCM = cm.get_cmap()
theCM._init()
alphas = np.abs(np.linspace(-1.0, 1.0, theCM.N))
theCM._lut[:-3,-1] = alphas



surf = ax.plot_surface(X, Y, Z, linewidth=0.1,color='#7dff63')
surf = ax.plot_surface(X, Y, Zpremium, linewidth=0,color='blue',alpha=0.5)
surf = ax.plot_surface(X, Y, Zko, linewidth=0.1,color='#ffcece')


ax.zaxis.set_major_formatter(FormatStrFormatter('%1.1e $'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%1.1e $'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%1.1e $'))



ax.xaxis.set_major_locator(LinearLocator(3))
ax.yaxis.set_major_locator(LinearLocator(3))
ax.zaxis.set_major_locator(LinearLocator(3))

plt.ylabel('Bandwidth Cost')

label = ax.set_xlabel('Infrastructure Costs')

label = ax.set_ylabel('Bandwidth Costs')

label = ax.set_zlabel('Premium CDN Pricing, g(.)')

for tick in ax.xaxis.get_major_ticks():
    # tick.label.set_fontsize(14)
    # specify integer or one of preset strings, e.g.
    tick.label.set_fontsize('x-small')
    tick.label.set_rotation('15')

for tick in ax.yaxis.get_major_ticks():
    # tick.label.set_fontsize(14)
    # specify integer or one of preset strings, e.g.
    tick.label.set_fontsize('x-small')
    tick.label.set_rotation('3')

for tick in ax.zaxis.get_major_ticks():
    # tick.label.set_fontsize(14)
    # specify integer or one of preset strings, e.g.
    tick.label.set_fontsize('x-small')
    tick.label.set_rotation('70')

for ii in xrange(0, 220, 1):
    ax.view_init(elev=15., azim=ii)






#plt.savefig("uc.png", dpi=None, facecolor='w', edgecolor='w',                orientation='landscape', papertype="A4", format="png",                transparent=False, bbox_inches=None, pad_inches=0.1,                frameon=None)
plt.savefig("uc.pdf", dpi=None, facecolor='w', edgecolor='w',                orientation='landscape', papertype="A4", format="pdf",                transparent=False, bbox_inches=None, pad_inches=0.1,                frameon=None)
#plt.show()
#fig.show()
print premium_cdn_pricing-g(alpha=alpha, beta=beta, t=t, s=s, r=r, l=l, d=d)
