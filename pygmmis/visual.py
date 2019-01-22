import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def plot_ellipse(ax, mu, covariance, color, linewidth=2, alpha=0.5):
    var, U = np.linalg.eig(covariance)
    angle = 180. / np.pi * np.arccos(np.abs(U[0, 0]))
    e = Ellipse(mu, 2 * np.sqrt(5.991 * var[0]),
                2 * np.sqrt(5.991 * var[1]),
                angle=angle)
    e.set_alpha(alpha)
    e.set_linewidth(linewidth)
    e.set_edgecolor(color)
    e.set_facecolor(color)
    e.set_fill(False)
    ax.add_artist(e)
    return e


def plot_centre(ax, mu, color):
    return ax.scatter(*mu.T, s=10, alpha=0.5, color=color)

def plot_direction(ax, old, new, **kwargs):
    return ax.arrow(*old, *new-old, **kwargs)


class GMMTracker(object):
    def __init__(self, backend, data, dim_view=None, ax_labels=(None, None)):
        self.backend = backend
        self.data = data
        self.artists = []
        self.n = None
        self.fig, self.axes = None, None
        self.n_components = self.backend.mu.shape[1]
        self.ndim = self.backend.mu.shape[2]
        self.dim_view = dim_view or [0, 1]
        assert len(self.dim_view) == 2, "Can only visualise a 2d slice"
        self.ax_labels = ax_labels

    def figure(self):
        a = np.sqrt(self.n_components)
        shape = [int(np.floor(a)), int(np.ceil(a))]
        if np.prod(shape) < self.n_components:
            shape[0] += 1
        self.fig = plt.figure()
        self.axes = []
        ax = None
        for i in range(self.n_components):
            ax = self.fig.add_subplot(shape[0], shape[1], i+1, sharex=ax, sharey=ax)
            ax.scatter(*self.data[:, self.dim_view].T, s=1, alpha=0.4)
            ax.set_title(i)
            ax.set_xlabel(self.ax_labels[0])
            ax.set_ylabel(self.ax_labels[1])
            self.axes.append(ax)


    def plot(self, n, clear=True, color='k'):
        if clear:
            self.clear()
        if n < 0:
            n = len(self.backend) + n
        self.n = n
        if self.axes is None:
            self.figure()
        for i, ax in enumerate(self.axes):
            try:
                v = self.backend.V[n][i][np.ix_(self.dim_view, self.dim_view)]
                e = plot_ellipse(ax, self.backend.mu[n][i][self.dim_view], v, color)
                c = plot_centre(ax, self.backend.mu[n][i][self.dim_view], color)
                self.artists.append(e)
                self.artists.append(c)
                direction = plot_direction(ax, self.backend.mu[n][i][self.dim_view], self.backend.mu[n+1][i][self.dim_view], color=color, label='EMstep')
                self.artists.append(direction)
            except IndexError:
                pass


    def clear(self):
        for a in self.artists:
            a.remove()
        self.artists = []

    def next(self):
        if self.n == len(self.backend)-1:
            raise IndexError("No more iterations left")
        if self.n is None:
            self.n = 0
            self.plot(0)
        self.n += 1
        self.plot(self.n)

    def previous(self):
        if (self.n is None) or (self.n == 0):
            raise IndexError("You are at the start")
        self.n -= 1
        self.plot(self.n)

    def __len__(self):
        return len(self.backend)

    def plot_trace(self, start=0, stop=-1, step=1):
        self.clear()
        if stop < 0:
            stop += len(self)
        if start < 0:
            start += len(self)
        cmap = matplotlib.cm.get_cmap('viridis')
        from matplotlib.colors import Normalize
        mappable = matplotlib.cm.ScalarMappable(norm=Normalize(0, stop), cmap=cmap)
        mappable.set_array(np.arange(0, stop))
        ranged = range(start, stop, step)
        for i in ranged:
            color = cmap((i - ranged.start) / (ranged.stop - ranged.start))
            self.plot(i, clear=False, color=color)
