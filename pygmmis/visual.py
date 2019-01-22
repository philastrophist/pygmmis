import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def orientation_from_covariance(cov, sigma):
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * sigma * np.sqrt(vals)
    return w, h, theta

def plot_ellipse(ax, mu, covariance, color, linewidth=2, alpha=0.5):
    x, y, angle = orientation_from_covariance(covariance, 2)
    e = Ellipse(mu, x, y, angle=angle)
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
    def __init__(self, backend, data, ax_labels=None):
        self.backend = backend
        self.data = data
        self.artists = []
        self.n = None
        self.figs, self.axes = [], []
        self.n_components = self.backend.mu.shape[1]
        self.ndim = self.backend.mu.shape[2]
        self.ax_labels = ax_labels

    def figure(self, k):
        fig = plt.figure()
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(self.ndim, self.ndim)
        subplots = np.empty((self.ndim, self.ndim), dtype=object)
        for i in range(self.ndim):
            for j in range(i+1, self.ndim):
                sharex = subplots[i, j-1] if i > 0 else None
                sharey = subplots[i-1, j] if j > 0 else None
                subplots[i, j] = subplots[j, i] = ax = fig.add_subplot(gs[j, i], sharex=sharex, sharey=sharey)
                ax.scatter(*self.data[:, [i, j]].T, s=1, alpha=0.4)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i == 0:
                    ax.get_yaxis().set_visible(True)
                    ax.set_ylabel(self.ax_labels[j])
                if j == self.ndim-1:
                    ax.get_xaxis().set_visible(True)
                    ax.set_xlabel(self.ax_labels[i])

        fig.suptitle("component-"+str(k))
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        return fig, subplots

    def figures(self):
        for k in range(self.n_components):
            fig, axs = self.figure(k)
            self.figs.append(fig)
            self.axes.append(axs)

    def plot(self, n, clear=True, color='k'):
        if clear:
            self.clear()
        if n < 0:
            n = len(self.backend) + n
        self.n = n
        if self.axes is None:
            self.figures()

        for k, (fig, axs) in enumerate(zip(self.figs, self.axes)):
            for i in range(self.ndim):
                for j in range(i + 1, self.ndim):
                    ax = axs[i, j]
                    try:
                        v = self.backend.V[n][k][np.ix_([i, j], [i, j])]
                        e = plot_ellipse(ax, self.backend.mu[n][k][[i, j]], v, color)
                        c = plot_centre(ax, self.backend.mu[n][k][[i, j]], color)
                        self.artists.append(e)
                        self.artists.append(c)
                        direction = plot_direction(ax, self.backend.mu[n][k][[i, j]],
                                                   self.backend.mu[n+1][k][[i, j]], color=color, label='EMstep')
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
