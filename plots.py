'''
Plots
'''

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def save_data(plot_data, file_path):
    '''Save the plot_data to file_path
       plot_data's element: [length of subset, Ot of subset, the subset list]
    Attributes:
        plot_data (list)
        file_path (str)
    '''
    print('start saving data')
    with open(file_path, 'w') as f:
        for data in plot_data:
            f.write(str(data[0]) + ',' + str(data[1]) + '\n')


def visualize_selection(counter, grid_len, primary, secondary, results, sensors):
    '''Visualize the legal transmitters and sensors selected
    Args:
        counter (int):    iteration number
        grid_len (int):   length of grid
        primary (list):   list of integers (index of transmitter)
        secondary (list): list of integers (index of transmitter)
        results (list):   results[-1] is list of (x, y), results[1] is O_t aux
        sensors (list):   list of Sensor object
    '''
    grid = np.zeros((grid_len, grid_len))

    for s in sensors:
        grid[s.x][s.y] = 0.2

    for s in results[-1]:
        x = sensors[s].x
        y = sensors[s].y
        grid[x][y] = 1

    for t in primary:
        x = t//grid_len
        y = t%grid_len
        grid[x][y] = -1

    for t in secondary:
        x = t//grid_len
        y = t%grid_len
        grid[x][y] = -0.8

    sns.set(style="white")
    f, ax = plt.subplots(figsize=(16, 16))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(grid, cmap=cmap, vmax=1, center=0, square=True, linewidth=1, cbar_kws={"shrink": .5})
    #plt.show()
    plt.title('O_aux= {}'.format(results[1]), fontsize=20)
    plt.savefig('visualize/{}'.format(counter))


def visualize_sensor_output(grid_len, intruders, sensor_outputs, sensors, threshold, fig):
    '''Visualize the intruders and sensor_output to have a better sense on deciding the threshold
    Args:
        grid_len (int):       length of grid
        intruders (list):     list of Transmitter objects
        sensor_output (list): list of float (RSSI)
        sensors (lists):      list of Sensor objects
    '''
    grid = np.zeros((grid_len, grid_len))
    maximum = np.max(sensor_outputs)
    minimum = np.min(sensor_outputs)
    for index, sensor in enumerate(sensors):
        if sensor_outputs[index] > threshold:
            color = (sensor_outputs[index] - minimum) / (maximum - minimum) + 0.2
        else:
            color = 0
        grid[sensor.x][sensor.y] = color
        #print((sensor[0], sensor[1]), sensor_output[index], '--', color)
    for intr in intruders:
        grid[intr.x][intr.y] = -1

    sns.set(style="white")
    f, ax = plt.subplots(figsize=(8, 8))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(grid, cmap=cmap, center=0, square=True, linewidth=1, cbar_kws={"shrink": .5})
    #plt.show()
    plt.savefig('visualize/localization/{}-sensor-output'.format(fig))


def visualize_cluster(grid_len, intruders, sensor_to_cluster, labels):
    '''Visualize the clustering results
    Args:
        grid_len (int):           length of grid
        intruders (list):         list of Transmitter objects
        sensor_to_cluster (list): list of position (x, y)
        labels (list):            list of labels
    '''
    grid = np.zeros((grid_len, grid_len))
    num_labels = len(np.unique(labels))
    step = 1.5/num_labels
    for i in range(len(sensor_to_cluster)):
        sensor = sensor_to_cluster[i]
        color = -1 + (labels[i]+1)*step
        if color > -0.25:
            color = 0.25 + (num_labels - labels[i])*step
        grid[sensor[0]][sensor[1]] = color
    for intr in intruders:
        grid[intr.x][intr.y] = -1
    sns.set(style="white")
    f, ax = plt.subplots(figsize=(8, 8))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(grid, cmap=cmap, center=0, square=True, linewidth=1, cbar_kws={"shrink": .5})
    plt.show()


def visualize_localization(grid_len, true_locations, pred_locations, fig):
    '''Visualize the localization
    Args:
        true_locations (list): each element is a coordinate (x, y)
        pred_locations (list): each element is a coordinate (x, y)
    '''
    grid = np.zeros((grid_len, grid_len))
    for true in true_locations:
        grid[true[0]][true[1]] = -1       # miss
    for pred in pred_locations:
        if grid[pred[0]][pred[1]] == -1:
            grid[pred[0]][pred[1]] = 0.4  # accurate prediction
        else:
            grid[pred[0]][pred[1]] = 1    # false alarm
    sns.set(style="white")
    f, ax = plt.subplots(figsize=(8, 8))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(grid, cmap=cmap, center=0, square=True, linewidth=1, cbar_kws={"shrink": .5})
    #plt.show()
    plt.savefig('visualize/localization/{}-localization'.format(fig))


def save_data_AGA(plot_data, file_path):
    '''Save the plot_data to file_path for offline greedy
        plot_data's element: [length of subset, ot approx, ot real]
    Attributes:
        plot_data (list)
        file_path (str)
    '''
    print('start saving data')
    with open(file_path, 'w') as f:
        for data in plot_data:
            # length, ot_approx, ot_real
            f.write(str(data[0]) + ',' + str(data[1]) + ',' + str(data[2]) + '\n')


def figure_localization_error():
    x = [0, 1, 2]
    loc_error = [50, 50 * 2.4722643030095783, 50 * 3.6372462313440623]
    plt.bar(x, loc_error)
    plt.xticks(x, ('Our', 'Clustering', 'SPLOT'))
    plt.ylabel('Localization Error (m)')
    plt.xlabel('Algorithm')
    #misses = [0, 0, 0]
    #plt.bar(loc_error)
    plt.show()

if __name__ == '__main__':
    figure_localization_error()
    pass
