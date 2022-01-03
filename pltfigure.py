import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def pltfigure(x1,x2,T,name1,name2,savename):  
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    line, = ax.plot([], [], linestyle="dotted", linewidth=2)
    line2, = ax.plot(
        [],
        [],
    )
    ax.autoscale_view()
    xlimmin = np.min(T)
    xlimmax = np.max(T)
    ax.set_xlim(xlimmin, xlimmax)
    ax.legend([name1, name2], loc='upper right')


    def animate(i):
        line.set_xdata(T)
        line.set_ydata(x1[i,:])
        line2.set_xdata(T)
        line2.set_ydata(x2[i,:])
        ax.set_ylim(-np.max(x2[i,:]),np.max(x2[i,:]))
        return line, line2


    ani = animation.FuncAnimation(fig,
                                  animate,
                                  frames=x1.shape[0],
                                  interval=1,
                                  blit=True)
    plt.show()

    ani.save(savename, writer='PillowWriter', fps=5)