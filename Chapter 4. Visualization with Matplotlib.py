
# coding: utf-8

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt


# In[2]:


plt.style.use('classic')


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import numpy as np


# In[6]:


x = np.linspace(0, 10, 100)
fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--');


# In[7]:


import os


# In[10]:


os.chdir('D:\\Python Examples\\')


# In[11]:


fig.savefig('myfigure.png')


# In[12]:


from IPython.display import Image
Image('my_figure.png')


# In[13]:


plt.figure() # create a plot figure
plt.subplot(2, 1, 1) # rows, columns, panel number
plt.plot(x, np.sin(x))

plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x));


# In[14]:


# First create a grid of plots
# ax will be an array of two Axes objects
fig, ax = plt.subplots(2)

# Call plot() method on the appropiate object
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x));


# In[15]:


plt.style.use('seaborn-whitegrid')


# In[19]:


fig = plt.figure()
ax = plt.axes()
x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x));


# In[21]:


plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x));


# In[22]:


plt.plot(x, np.sin(x))
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5);


# In[23]:


plt.plot(x, np.sin(x))
plt.xlim(10, 0)
plt.ylim(1.2, -1.2);


# In[24]:


plt.plot(x, np.sin(x))
plt.axis([-1, 11, -1.5, 1.5]);


# In[26]:


plt.plot(x, np.sin(x))
plt.axis('equal');


# In[30]:


plt.plot(x, np.sin(x))
plt.title('A sine curve')
plt.xlabel('x')
plt.ylabel('sin(x)');


# In[31]:


plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.axis('equal')

plt.legend();


# In[33]:


ax = plt.axes()
ax.plot(x, np.sin(x))
ax.set(xlim=(0, 10), ylim=(-2, 2),
      xlabel='x', ylabel='sin(x)',
      title='A simple plot');


# In[36]:


plt.plot(x, np.sin(x), 'ok');


# In[39]:


rng = np.random.RandomState(0)
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker,
            label="marker='{0}'".format(marker))

plt.legend(numpoints=1)
plt.xlim(0, 1.8);


# In[45]:


x = np.linspace(0, 10, 20)


# In[46]:


plt.plot(x, np.sin(x), '-p', color='gray',
        markersize=15, linewidth=4,
        markerfacecolor='white',
        markeredgecolor='gray',
        markeredgewidth=2)
plt.ylim(-1.2, 1.2);


# In[49]:


plt.scatter(x, np.sin(x), marker='o');


# In[51]:


rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
           cmap='viridis')
plt.colorbar();


# In[53]:


x = np.linspace(0, 50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)

plt.errorbar(x, y, yerr=dy, fmt='.k');


# In[54]:


plt.errorbar(x, y, yerr=dy, fmt='o', color='black',
            ecolor='lightgray', elinewidth=3, capsize=0);


# In[55]:


def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)


# In[56]:


x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)


# In[57]:


plt.contour(X, Y, Z, colors='black');


# In[60]:


plt.contour(X, Y, Z, 20, cmap='jet_r');


# In[61]:


plt.contourf(X, Y, Z, 20, cmap='jet_r')
plt.colorbar();


# In[63]:


plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
          cmap='jet_r')
plt.colorbar()
plt.axis(aspect='image');


# In[66]:


contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)

plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
          cmap='RdGy', alpha=0.5)
plt.colorbar();


# In[67]:


plt.style.use('seaborn-white')
data = np.random.randn(1000)


# In[68]:


plt.hist(data);


# In[73]:


plt.hist(data, bins=30, density=True, alpha=0.5,
        histtype='stepfilled', color='steelblue',
        edgecolor='none');


# In[74]:


plt.hist(data, histtype='bar');


# In[76]:


plt.hist(data, histtype='stepfilled');


# In[77]:


x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)

kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs);


# In[78]:


counts, bin_edges = np.histogram(data, bins=5)


# In[79]:


counts


# In[80]:


bin_edges


# In[83]:


mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T


# In[87]:


plt.hist2d(x, y, bins=30, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')


# In[88]:


plt.hexbin(x, y, gridsize=30, cmap='Blues')
cb = plt.colorbar(label='count in bin')


# In[91]:


from scipy.stats import gaussian_kde
data = np.vstack([x, y])
kde = gaussian_kde(data)

xgrid = np.linspace(-3.5, 3.5, 40)
ygrid = np.linspace(-6, 6, 40)
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

plt.imshow(Z.reshape(Xgrid.shape),
          origin='lower', aspect='auto',
          extent=[-3.5, 3.5, -6, 6],
          cmap='Blues')
cb = plt.colorbar()
cb.set_label('density')


# In[92]:


plt.style.use('classic')


# In[93]:


x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), '-b', label='Sine')
ax.plot(x, np.cos(x), '--r', label='Cosine')
ax.axis('equal')
leg = ax.legend();


# In[94]:


ax.legend(loc='upper left', frameon=False)
fig


# In[96]:


ax.legend(loc='lower center', frameon=False, ncol=2)
fig


# In[100]:


ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
fig


# In[101]:


y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5))
lines = plt.plot(x, y)
# lines is a list of plt.line2D instances
plt.legend(lines[:2], ['first', 'second']);


# In[106]:


plt.plot(x, y[:, 0], label='first')
plt.plot(x, y[:, 1], label='second')
plt.plot(x, y[:, 2:])
plt.legend(framealpha=1, frameon=True);


# In[107]:


y.shape


# In[108]:


import pandas as pd


# In[109]:


cities = pd.read_csv('california_cities.txt')


# In[110]:


cities.head()


# In[116]:


# Extract the data we're interested in
lat, lon = cities['latd'], cities['longd']
population, area = cities['population_total'], cities['area_total_km2']

# Scatter the points, using size and color but not label
plt.scatter(lon, lat, label=None,
           c=np.log10(population), cmap='viridis',
           s=area, linewidth=0, alpha=0.5)
plt.axis(aspect='equal')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar(label='log$_{10}$(population)')
plt.clim(3, 7)

# Here we create a legend:
# we'll plot empty lists with the desired size and label
for area in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.3, s=area,
               label=str(area) + ' km$^2$')
plt.legend(scatterpoints=1, frameon=False,
          labelspacing=1, title='City Area')

plt.title('California Cities: Area and Population');


# In[117]:


fig, ax = plt.subplots()

lines = []
styles = ['-', '--', '-.', ':']
x = np.linspace(0, 10, 1000)

for i in range(4):
    lines += ax.plot(x, np.sin(x - i * np.pi / 2),
                     styles[i], color='black')

    ax.axis('equal')

# specify the lines and labels of the first legend
ax.legend(lines[:2], ['line A', 'line B'],
         loc='upper right', frameon=False)

# Create the second legend and add the artist manually.
from matplotlib.legend import Legend
leg = Legend(ax, lines[2:], ['line C', 'line D'],
            loc='lower right', frameon=False)
ax.add_artist(leg);


# In[118]:


plt.style.use('classic')


# In[119]:


x = np.linspace(0, 10, 1000)
I = np.sin(x) * np.cos(x[:, np.newaxis])

plt.imshow(I)
plt.colorbar();


# In[1]:


import matplotlib.pyplot as plt
import numpy as np
plt.style.use('classic')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


x = np.linspace(0, 10, 1000)
I = np.sin(x) * np.cos(x[:, np.newaxis])

plt.imshow(I)
plt.colorbar();


# In[5]:


plt.imshow(I, cmap='gray');


# In[127]:


plt.cm.binary


# In[12]:


plt.imshow(I, cmap='jet');


# In[14]:


plt.imshow(I, cmap='cubehelix');


# In[15]:


# make noise in 1% of the image pixels
speckles = (np.random.random(I.shape) < 0.01)
I[speckles] = np.random.normal(0, 3, np.count_nonzero(speckles))

plt.figure(figsize=(10, 3.5))

plt.subplot(1, 2, 1)
plt.imshow(I, cmap='RdBu')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(I, cmap='RdBu')
plt.colorbar(extend='both')
plt.clim(-1, 1);


# In[21]:


I[I > 2]


# In[22]:


plt.imshow(I, cmap=plt.cm.get_cmap('Blues', 6))
plt.colorbar()
plt.clim(-1, 1)


# In[23]:


# load images of the digits 0 through 5 and visualize several of them
from sklearn.datasets import load_digits
digits = load_digits(n_class=6)

fig, ax = plt.subplots(8, 8, figsize=(6, 6))
for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap='binary')
    axi.imshow(digits.images[i], cmap='binary')
    axi.set(xticks=[], yticks=[])


# In[25]:


# project the digits into 2 dimensions using IsoMap
from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
projection = iso.fit_transform(digits.data)


# In[26]:


# plot the results
plt.scatter(projection[:, 0], projection[:, 1], lw=0.1,
           c=digits.target, cmap=plt.cm.get_cmap('cubehelix', 6))
plt.colorbar(ticks=range(6), label='digits value')
plt.clim(-0.5, 5.5)


# In[27]:


plt.style.use('seaborn-white')


# In[28]:


ax1 = plt.axes() # standard axes
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])


# In[42]:


fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
                  xticklabels=[], ylim=(-1.2, 1.2))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
                  ylim=(-1.2, 1.2))

x = np.linspace(0, 10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x));


# In[43]:


for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.text(0.5, 0.5, str((2, 3, i)),
            fontsize=18, ha='center')


# In[44]:


fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 7):
    ax = fig.add_subplot(2, 3, i)
    ax.text(0.5, 0.5, str((2, 3, i)),
           fontsize=18, ha='center')


# In[46]:


fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')


# In[50]:


# axes are in a two-dimensional array, indexed by [row, col]
for i in np.arange(2):
    for j in np.arange(3):
        ax[i, j].text(0.5, 0.5, str((i, j)),
                     fontsize=18, ha='center')
fig


# In[51]:


grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)


# In[55]:


plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2]);


# In[58]:


# Create some normally distributed data
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 3000).T

# Set up the axes with gridspec
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

# scatter points on the main axes
main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)

# histogram on the attached axes
x_hist.hist(x, 40, histtype='stepfilled',
           orientation='vertical', color='gray')
x_hist.invert_yaxis()
y_hist.hist(y, 40, histtype='stepfilled',
           orientation='horizontal', color='gray')
y_hist.invert_xaxis()


# In[59]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd


# In[60]:


import os


# In[61]:


os.chdir('D:\\Python Examples\\')


# In[62]:


births = pd.read_csv('births.txt')


# In[63]:


quartiles = np.percentile(births['births'], [25, 50, 75])
mu, sig = quartiles[1], 0.74 * (quartiles[2] - quartiles[0])
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')

births['day'] = births['day'].astype(int)

births.index = pd.to_datetime(10000 * births.year +
                             100 * births.month +
                             births.day, format='%Y%m%d')
births_by_date = births.pivot_table('births',
                                   [births.index.month, births.index.day])
births_by_date.index = [pd.datetime(2012, month, day)
                       for (month, day) in births_by_date.index]


# In[69]:


df = pd.DataFrame(np.random.randn(10, 2), columns=list('ab'))


# In[70]:


df


# In[71]:


df[df.a > df.b]


# In[72]:


df.query('a > b')


# In[74]:


fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax);


# In[78]:


fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)

# Add labels to the plot
style = dict(size=10, color='gray')

ax.text('2012-1-1', 3950, "New Year's Day", **style)
ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)
ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
ax.text('2012-12-25', 4600, "Christmas", ha='right', **style)

# Label the axes
ax.set(title='USA births by day of the year (1969-1988)',
      ylabel='average daily births');


# In[79]:


fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)

# Add labels to the plot
style = dict(size=10, color='gray')

ax.text('2012-1-1', 3950, "New Year's Day", **style)
ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)
ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
ax.text('2012-12-25', 4600, "Christmas", ha='right', **style)

# Label the axes
ax.set(title='USA births by day of the year (1969-1988)',
      ylabel='average daily births')

# Format the x axis with centered month labels
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'));


# In[81]:


fig, ax = plt.subplots(facecolor='lightgray')
ax.axis([0, 10, 0, 10])

# transform=ax.transData is the default, but we'll specify it anyway
ax.text(1, 5, '.Data: (1, 5)', transform=ax.transData)
ax.text(0.5, 0.1, '.Axes: (0.5, 0.1)', transform=ax.transAxes)
ax.text(0.2, 0.2, '.Figure: (0.2, 0.2)', transform=fig.transFigure);


# In[82]:


ax.set_xlim(0, 2)
ax.set_ylim(-6, 6)
fig


# In[86]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[87]:


ax.set_xlim(0, 2)
ax.set_ylim(-6, 6)
fig


# In[88]:


fig, ax = plt.subplots()

x = np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')

ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),
           arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('local minimum', xy=(5* np.pi, -1), xytext=(2, -6),
           arrowprops=dict(arrowstyle='->',
                          connectionstyle='angle3,angleA=0, angleB=90'));


# In[89]:


fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)

# Add labels to the plot
ax.annotate("New Year's Day", xy=('2012-1-1', 4100),  xycoords='data',
            xytext=(50, -30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.2"))

ax.annotate("Independence Day", xy=('2012-7-4', 4250),  xycoords='data',
            bbox=dict(boxstyle="round", fc="none", ec="gray"),
            xytext=(10, -40), textcoords='offset points', ha='center',
            arrowprops=dict(arrowstyle="->"))

ax.annotate('Labor Day', xy=('2012-9-4', 4850), xycoords='data', ha='center',
            xytext=(0, -20), textcoords='offset points')
ax.annotate('', xy=('2012-9-1', 4850), xytext=('2012-9-7', 4850),
            xycoords='data', textcoords='data',
            arrowprops={'arrowstyle': '|-|,widthA=0.2,widthB=0.2', })

ax.annotate('Halloween', xy=('2012-10-31', 4600),  xycoords='data',
            xytext=(-80, -40), textcoords='offset points',
            arrowprops=dict(arrowstyle="fancy",
                            fc="0.6", ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90"))

ax.annotate('Thanksgiving', xy=('2012-11-25', 4500),  xycoords='data',
            xytext=(-120, -60), textcoords='offset points',
            bbox=dict(boxstyle="round4,pad=.5", fc="0.9"),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle,angleA=0,angleB=80,rad=20"))


ax.annotate('Christmas', xy=('2012-12-25', 3850),  xycoords='data',
             xytext=(-30, 0), textcoords='offset points',
             size=13, ha='right', va="center",
             bbox=dict(boxstyle="round", alpha=0.1),
             arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1));

# Label the axes
ax.set(title='USA births by day of year (1969-1988)',
       ylabel='average daily births')

# Format the x axis with centered month labels
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'));

ax.set_ylim(3600, 5400);


# In[90]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np


# In[91]:


ax = plt.axes(xscale='log', yscale='log')


# In[92]:


print(ax.xaxis.get_major_locator())
print(ax.xaxis.get_minor_locator())


# In[93]:


print(ax.xaxis.get_major_formatter())
print(ax.xaxis.get_minor_formatter())


# In[96]:


ax = plt.axes()
ax.plot(np.random.rand(50))

ax.yaxis.set_major_locator(plt.NullLocator())
ax.xaxis.set_major_formatter(plt.NullFormatter())


# In[97]:


fig, ax = plt.subplots(5, 5, figsize=(5, 5))
fig.subplots_adjust(hspace=0, wspace=0)

# Get some face data from scikit-learn
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces().images

for i in range(5):
    for j in range(5):
        ax[i, j].xaxis.set_major_locator(plt.NullLocator())
        ax[i, j].yaxis.set_major_locator(plt.NullLocator())
        ax[i, j].imshow(faces[10 * i + j], cmap='bone')


# In[98]:


fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)


# In[100]:


# For every axis, set the x and y major locator
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(3))
    axi.yaxis.set_major_locator(plt.MaxNLocator(3))

fig


# In[103]:


# Plot a sine and cosine curve
fig, ax = plt.subplots()
x = np.linspace(0, 3 * np.pi, 1000)
ax.plot(x, np.sin(x), lw=3, label='Sine')
ax.plot(x, np.cos(x), lw=3, label='Cosine')

# Set up grid, legend, and limits
ax.grid(True)
ax.legend(frameon=False)
ax.axis('equal')
ax.set_xlim(0, 3 * np.pi);


# In[107]:


ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
fig


# In[108]:


def format_func(value, tick_number):
    # find number of multiples of pi/2
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return  r"${0}\pi$".format(N // 2)


# In[109]:


ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
fig


# In[110]:


import matplotlib.pyplot as plt
plt.style.use('classic')
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[112]:


x = np.random.randn(1000)
plt.hist(x);


# In[118]:


# use a a gray background
ax = plt.axes(facecolor='#E6E6E6')
ax.set_axisbelow(True)

# draw solid white grid lines
plt.grid(color='w', linestyle='solid')

# hide axis spines
for spine in ax.spines.values():
    spine.set_visible(False)

# hide top and right ticks
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
    
# lighten ticks and labels
ax.tick_params(colors='gray', direction='out')
for tick in ax.get_xticklabels():
    tick.set_color('gray')
for tick in ax.get_yticklabels():
    tick.set_color('gray')
    
# control face and edge color of histogram
ax.hist(x, edgecolor='#E6E6E6', color='#EE6666');


# In[1]:


import matplotlib.pyplot as plt
plt.style.use('classic')
import numpy as np


# In[2]:


IPython_default = plt.rcParams.copy()


# In[3]:


from matplotlib import cycler
colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)


# In[4]:


x = np.random.normal(10, 5, 1000)


# In[6]:


plt.hist(x);


# In[7]:


for i in range(4):
    plt.plot(np.random.randn(10))


# In[9]:


plt.style.available


# In[10]:


def hist_and_lines():
    np.random.seed(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.random.randn(1000))
    for i in range(3):
        ax[1].plot(np.random.rand(10))
    ax[1].legend(['a', 'b', 'c'], loc='lower left')


# In[11]:


# reset rcParams
plt.rcParams.update(IPython_default);


# In[12]:


hist_and_lines()


# In[13]:


with plt.style.context('fivethirtyeight'):
    hist_and_lines()


# In[14]:


with plt.style.context('ggplot'):
    hist_and_lines()


# In[15]:


with plt.style.context('bmh'):
    hist_and_lines()


# In[16]:


with plt.style.context('dark_background'):
    hist_and_lines()


# In[17]:


with plt.style.context('grayscale'):
    hist_and_lines()


# In[19]:


import seaborn
hist_and_lines()


# In[22]:


with plt.style.context('tableau-colorblind10'):
    hist_and_lines()


# In[23]:


import matplotlib.pyplot as plt
plt.style.use('classic')
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd


# In[28]:


# Create some data
rng = np.random.RandomState(0)
x = np.linspace(0, 10, 500)
y = np.cumsum(rng.randn(500, 6), 0)


# In[29]:


# Plot the data with Matplotlib defaults
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');


# In[12]:


import seaborn as sns
sns.set()


# In[32]:


# same plotting code as above!
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');


# In[36]:


data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])


# In[38]:


for col in 'xy':
    plt.hist(data[col], density=True, alpha=0.5)


# In[39]:


for col in 'xy':
    sns.kdeplot(data[col], shade=True)


# In[40]:


sns.distplot(data['x'])
sns.distplot(data['y']);


# In[41]:


sns.kdeplot(data);


# In[46]:


with sns.axes_style('white'):
    sns.jointplot('x', 'y', data, kind='kde');


# In[49]:


with sns.axes_style('white'):
    sns.jointplot('x', 'y', data, kind='hex');


# In[50]:


with sns.axes_style('white'):
    sns.jointplot('x', 'y', data, kind='resid');


# In[52]:


with sns.axes_style('white'):
    sns.jointplot('x', 'y', data, kind='reg');


# In[53]:


iris = sns.load_dataset('iris')
iris.head()


# In[57]:


sns.pairplot(iris, hue='species', size=2.5);


# In[13]:


tips = sns.load_dataset('tips')
tips.head()


# In[1]:


import seaborn as sns


# In[15]:


tips = sns.load_dataset('tips')


# In[3]:


tips.head()


# In[16]:


tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']


# In[17]:


tips.head()


# In[18]:


grid = sns.FacetGrid(tips, row='sex', col='time', margin_titles=True)
grid.map(plt.hist, 'tip_pct', bins=np.linspace(0, 40, 15));


# In[9]:


import numpy as np
import matplotlib.pyplot as plt


# In[20]:


with sns.axes_style(style='ticks'):
    g = sns.factorplot('day', 'total_bill', 'sex', data=tips, kind='box')
    g.set_axis_labels('Day', 'Total Bill');


# In[22]:


with sns.axes_style('white'):
    sns.jointplot('total_bill', 'tip', data=tips, kind='hex')


# In[23]:


sns.jointplot('total_bill', 'tip', data=tips, kind='reg');


# In[24]:


planets = sns.load_dataset('planets')
planets.head()


# In[25]:


with sns.axes_style('white'):
    g = sns.factorplot('year', data=planets, aspect=2,
                      kind='count', color='steelblue')
    g.set_xticklabels(step=5)


# In[26]:


with sns.axes_style('white'):
    g = sns.factorplot('year', data=planets, aspect=4.0, kind='count',
                      hue='method', order=range(2001, 2015))
    g.set_ylabels('Number of Planets Discovered')


# In[1]:


import os


# In[2]:


os.chdir('D:\\Python Examples\\')


# In[3]:


import pandas as pd


# In[4]:


data = pd.read_csv('marathon-data.txt')


# In[5]:


data.head()


# In[6]:


data.dtypes


# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[12]:


def convert_time(s):
    h, m, s = map(int, s.split(':'))
    return pd.datetools.timedelta(hours=h, minutes=m, seconds=s)

data = pd.read_csv('marathon-data.txt',
                  converters={'split':convert_time, 'final':convert_time})
data.head()


# In[14]:


data.dtypes


# In[23]:


1

