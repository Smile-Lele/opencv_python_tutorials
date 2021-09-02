import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# configure full path for ImageMagick
rcParams['animation.convert_path'] = r'/usr/bin/convert'

TWOPI = 2 * np.pi

fig, ax = plt.subplots()

t = np.arange(0.0, TWOPI, 0.001)
s = np.sin(t)
l = plt.plot(t, s)

# ax = plt.axis([0, TWOPI, -1, 1])

redDot, = plt.plot([0], [np.sin(0)], 'ro')


def animate(i):
    redDot.set_data(i, np.sin(i))
    return redDot,


# create animation using the animate() function with no repeat
myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(0.0, TWOPI, 0.1), \
                                      interval=10, blit=True, repeat=False)

# save animation at 30 frames per second
myAnimation.save('myAnimation.gif', writer='pillow', fps=30)
