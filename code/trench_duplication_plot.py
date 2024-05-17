import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('pgf')

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
x, y = np.meshgrid(x, y)

vx = 4
vy = -1.5

u = (vx*y-vy*x)**2 + vx*x+vy*y

y_crit_1 = 1/(2*vy)
y_crit_2 = -vy/(2*vx**2)
lerp = lambda t, a, b: a + t*(b-a)
quiver_t = np.arange(-20, 20) / 2
quiver_y = lerp(quiver_t, y_crit_1, y_crit_2)
quiver_x = np.zeros_like(quiver_y)
quiver_grad_x = -2*vy*(vx*quiver_y-vy*quiver_x) + vx
quiver_grad_y = 2*vx*(vx*quiver_y-vy*quiver_x) + vy
quiver_grad_norm = np.sqrt(quiver_grad_x**2+quiver_grad_y**2)
quiver_grad_x /= quiver_grad_norm
quiver_grad_y /= quiver_grad_norm

fig = plt.figure(figsize=(5, 2.5))
ax = fig.add_axes([0, 0, 1, 1])
ax.contourf(x, y, u, levels=20, cmap='Greys')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
is_axis_aligned = (quiver_t==0.0) | (quiver_t==1.0)
ax.quiver(quiver_x, quiver_y, quiver_grad_x, quiver_grad_y, is_axis_aligned, cmap='copper')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_axis_off()
fig.savefig('latex-build/trench-duplication.pgf', format='pgf')
