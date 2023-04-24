import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages('speed-eao2019.pdf')
plt.rc('font',family='Times New Roman')

fig, ax = plt.subplots()  # type:figure.Figure, axes.Axes
ax.set_title('The Performance $vs.$ Speed on VOT-2019', fontsize=15)
ax.set_xlabel('Tracking Speed (FPS)', fontsize=15)
ax.set_ylabel('EAO', fontsize=15)


trackers = ['Ocean-offline-25.9M', 'SiamRPN++-11.2M', 'DaSiamRPN-19.6M', 'SiamDW-34M', 'SiamBAN-10.8M', 'LightTrack-1.97M', 'SiamMask-16.6M', 'Ours (Small)-0.28M', 'Ours (Large)-0.54M']
speed = np.array([72, 35, 160, 67, 40, 52.6, 120, 131, 135])
#speed_norm = np.array([50, 75, 65, 35, 50, 45, 72, 58, 25]) / 48
params=np.array([518,224,392,680,216,39.4,332,5.6,10.8])
performance = np.array([0.273, 0.285, 0.276, 0.299, 0.327, 0.333, 0.287, 0.321, 0.327])

circle_color = ['cornflowerblue', 'deepskyblue',  'turquoise', 'gold', 'yellowgreen', 'orange', 'pink', 'r', 'r']
# Marker size in units of points^2
volume = params #(300 * speed_norm/5 * performance/0.6)  ** 2

ax.scatter(speed, performance, c=circle_color, s=volume*5, alpha=0.4)
ax.scatter(speed, performance, c=circle_color, s=20, marker='o')
# text
ax.text(speed[0] - 7.2, performance[0] - 0.0131, trackers[0], fontsize=10, color='k')
ax.text(speed[1] - 3.5, performance[1] - 0.005, trackers[1], fontsize=10, color='k')
ax.text(speed[2] - 16.0, performance[2] - 0.005, trackers[2], fontsize=10, color='k')
ax.text(speed[3] - 6.7, performance[3] - 0.0032, trackers[3], fontsize=10, color='k')
ax.text(speed[4] - 4.0, performance[4] - 0.0040, trackers[4], fontsize=10, color='k')
ax.text(speed[5] - 5.26, performance[5] - 0.0042, trackers[5], fontsize=10, color='k')
ax.text(speed[6] - 12.0, performance[6] + 0.0041, trackers[6], fontsize=10, color='k')
ax.text(speed[7] - 13.1, performance[7] -0.0050, trackers[7], fontsize=12, color='k')
ax.text(speed[8] - 13.5, performance[8] -0.0035, trackers[8], fontsize=12, color='k')


ax.grid(which='major', axis='both', linestyle='-.') # color='r', linestyle='-', linewidth=2
ax.set_xlim(10, 175)
ax.set_ylim(0.245, 0.34)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)

# plot lines
ystart, yend = ax.get_ylim()
ax.plot([25, 25], [ystart, yend], linestyle="--", color='k', linewidth=0.7)
ax.plot([131, 135], [0.321,  0.327], linestyle="--", color='r', linewidth=0.7)
#ax.plot([58, 72], [0.467, 0.438], linestyle="--", color='r', linewidth=0.7)
ax.text(26, 0.25, 'Real-time line', fontsize=13, color='k')
ax.text(140, 0.254, 'parameters', fontsize=16, color='k')



fig.savefig('speed-eao2018.svg')


pdf.savefig()
pdf.close()
plt.show()
