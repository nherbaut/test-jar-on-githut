import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, AutoLocator
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def eq31(alpha=0, s=0, beta=0, r=0):
    return alpha * s - beta * r


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
t = 40000000000.0*200000/30*0.0563/60/60.0/10**6*0.63*8
s = 80000
r = 200000
l = 100000
d = 550000
premium_cdn_pricing=100000


ss = np.arange(s/40.0,s*4,(s*4-s/40.0)/20.0)
#ds = np.arange(d/4.0,d*2,d*2/200.0)
ds = np.arange(0,d*2,(d*2)/20.0)
X, Y = np.meshgrid(ss, ds)



theCM = cm.get_cmap()
theCM._init()
theCM._lut[:-3,-1] = 0.2





Z = np.empty((len(ss), len(ds)))
for i in range(0, len(X)):
    for j in range(0, len(Y)):
        if( g(alpha, beta, t, X[i][j], r, Y[i][j], d) <= premium_cdn_pricing):
            Z[i][j] = g(alpha, beta, t, X[i][j], r, l, Y[i][j])
        else:
            Z[i][j] = None


surf = ax.plot_surface(X, Y, Z,linewidth=0)

Z = np.empty((len(ss), len(ds)))
for i in range(0, len(X)):
    for j in range(0, len(Y)):
        Z[i][j] = premium_cdn_pricing

surf = ax.plot_surface(X, Y, Z,linewidth=0,cmap=theCM)




ax.zaxis.set_major_formatter(FormatStrFormatter('%1.1e $'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%1.1e $'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%1.1e $'))


Z = np.empty((len(ss), len(ds)))
for i in range(0, len(X)):
    for j in range(0, len(Y)):
        Z[i][j] = g(alpha, beta, t, X[i][j], r, l, Y[i][j])



surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                       linewidth=0.00, antialiased=True,cmap=theCM)


ax.xaxis.set_major_locator(LinearLocator(3))
ax.yaxis.set_major_locator(LinearLocator(3))
ax.zaxis.set_major_locator(LinearLocator(3))




plt.ylabel('Software development Cost')



label = ax.set_xlabel('Substrate Cost')


label = ax.set_ylabel('Software development Cost')


label = ax.set_zlabel('Premium CDN Pricing, g(.)')



for tick in ax.xaxis.get_major_ticks():
                #tick.label.set_fontsize(14)
                # specify integer or one of preset strings, e.g.
                tick.label.set_fontsize('x-small')
                tick.label.set_rotation('15')

for tick in ax.yaxis.get_major_ticks():
                #tick.label.set_fontsize(14)
                # specify integer or one of preset strings, e.g.
                tick.label.set_fontsize('x-small')
                tick.label.set_rotation('3')

for tick in ax.zaxis.get_major_ticks():
                #tick.label.set_fontsize(14)
                # specify integer or one of preset strings, e.g.
                tick.label.set_fontsize('x-small')
                tick.label.set_rotation('70')


for ii in xrange(0,220,1):
        ax.view_init(elev=15., azim=ii)



X = np.arange(s-10, s+10, 1)
xlen = len(X)
Y = np.arange(d-10, d+10, 1)
ylen = len(Y)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = g(alpha, beta, t, X[i][j], r, Y[i][j], d)

colortuple = ('y', 'b')
colors = np.empty(X.shape, dtype=str)
for y in range(ylen):
    for x in range(xlen):
        colors[x, y] = colortuple[(x + y) % len(colortuple)]

#surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,                       linewidth=0, antialiased=False)



#plt.savefig("uc.png", dpi=None, facecolor='w', edgecolor='w',                orientation='landscape', papertype="A4", format="png",                transparent=False, bbox_inches=None, pad_inches=0.1,                frameon=None)
plt.show()
fig.show()
