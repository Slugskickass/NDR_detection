import Detect as dt
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

file_name = 'Example_data.tif'

data = dt.loadtiffs(file_name)
#empty_events = dt.collect_non_event(data, 80)
#lines = dt.fit_line(empty_events)
#slopes, intercepts, breakpoints = dt.change_point(data[71, 98, :], sigma=5, diff=1)

data = data[110:140, 160:190, :]
XS, YS, yy = np.shape(data)
holdall = np.zeros((XS, YS))
for indexx, X in enumerate(range(np.shape(holdall)[0])):
    for indexy, Y in enumerate(range(np.shape(holdall)[0])):
        data_point = [X, Y]
        my_data = np.diff(data[data_point[0], data_point[1], :])
        out = np.asarray(dt.ck_filter(my_data, 3))
        slopes, intercepts, breakpoints = dt.change_point(out, sigma=5)
        holdall[indexx, indexy] = len(breakpoints)


plt.imshow(holdall)
plt.show()

[x, y] = np.where(holdall == np.max(holdall))
for I in range(-4, 5):
    line_data = data[x[0]+I, y[0], :]
    out = np.asarray(dt.ck_filter(np.diff(line_data), 3))
    slopes, intercepts, breakpoints = dt.change_point(out, sigma=5)

    slopes, intercepts = dt.get_gradients(line_data, breakpoints)
    plt.plot(line_data, '.')
    dt.plot_CP(line_data, breakpoints, slopes, intercepts)
plt.show()


