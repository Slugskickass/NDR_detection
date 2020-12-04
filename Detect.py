from PIL import Image
import numpy as np
from scipy import stats
import ruptures as rpt
import utilities as util
import matplotlib.pyplot as plt
from scipy import stats


def loadtiffs(file_name):
    """
    This function returns an array of images in numpy format
    :param file_name:
    :return: an array containing an image
    """
    img = Image.open(file_name)
    print('The Image is', img.size, 'Pixels.')
    print('With', img.n_frames, 'frames.')

    imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.int16)
    for I in range(img.n_frames):
        img.seek(I)
        imgArray[:, :, I] = np.asarray(img)
    img.close()
    return(imgArray)

def collect_non_event(data, per):
    """

    :param data: The data as a 3D stack, this is the NDR data
    :param per: The percentile of the data to be filteres
    :return: This returns a list containg all the voxels which are in the % percentile data
    basically all the lines with no event in them
    """
    final_frame = data[:, :, 299] - data[:, :, 0]
    perce = np.percentile(final_frame.flatten(), per)
    positions = np.where(final_frame < perce)

    X = positions[0]
    Y = positions[1]
    empty = []
    for item in range(len(X)):
        empty.append(data[X[item], Y[item], :])


    return (empty)

def fit_line(line_data):
    '''
    My fitting program because I dont like elliots
    :param line_data: A list of pixel values, intended to take data from the above function (collect non event
    :return: The slopes of all the data
    '''
    line_length = len(line_data)
    # Place to store the results
    hold_all = np.zeros(line_length)

    # build an array to supply as the X values for fitting
    x = np.linspace(0, len(line_data[0])-1, len(line_data[0]))
    #start_time = time.clock()
    for I in range(line_length):
        y = line_data[I]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        hold_all[I] = slope
    return hold_all

def change_point(signal, sigma=5):
    n = len(signal)
    model = 'l2'
    algorithm = rpt.Pelt(model=model, min_size=10).fit((signal))
    breakpoints = algorithm.predict(pen=np.log(n) * sigma ** 2)

    data = np.split(signal, breakpoints)
    breakpoints.insert(0, 0)

    def fit(p):
        y, start = p
        x = util.make_lin_fit_x(start=start, length=len(y))
        y = util.make_lin_fit_y(y)
        return util.lin_fit(x, y)

    intercepts, slopes,  = zip(*map(fit, zip(data, breakpoints[:-1])))

    return list(slopes), list(intercepts), breakpoints

def ck_filter(data, width):
    filtered_data = []
    for I in range(width, len(data)-width):
        before = data[I-width:I]
        after = data[I:width+I]
        step = np.int(width /2)
        current = data[I-step:I+step]
        if (np.mean(current) - np.mean(before)) < (np.mean(current) - np.mean(after)):
            filtered_data.append(np.mean(before))
        else:
            filtered_data.append(np.mean(after))
    return filtered_data

def get_gradients(y_data, breakpoints):
    slopes = []
    intercepts =[]
    breakpoints = np.append(breakpoints, len(y_data))
    x_data = np.linspace(0, len(y_data)-1, len(y_data))
    for I in range(len(breakpoints)-1):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data[breakpoints[I]:breakpoints[I+1]], y_data[breakpoints[I]:breakpoints[I+1]])
        slopes.append(slope)
        intercepts.append(intercept)
    return slopes, intercepts

def plot_CP(data, breakpoints, slopes, intercepts):
    x_data = np.linspace(0, len(data) - 1, len(data))
    for I in range(len(breakpoints) - 1):
        x_data_s = x_data[breakpoints[I]:breakpoints[I + 1]]
        y_data = x_data_s * slopes[I] + intercepts[I]
        plt.plot(x_data_s, y_data)
    return 0