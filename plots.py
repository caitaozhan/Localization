'''
Plots
'''

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


def visualize_unused_sensors(grid_len, intruders, sensor_outputs, sensors, sensors_used, previous_identified, threshold, fig):
    '''Visualize the intruders and UNUSED sensor_output
    Args:
        grid_len (int):                 length of grid
        intruders (list<Transmitter>):  list of Transmitter objects
        sensor_output (list):           list of float (RSSI)
        sensors (lists):                list of Sensor objects
        sensors_used (np.array<bool>):
    '''
    grid = np.zeros((grid_len, grid_len))
    if np.max(sensor_outputs) > threshold:
        maximum = np.max(sensor_outputs[sensor_outputs > threshold])
        minimum = np.min(sensor_outputs[sensor_outputs > threshold])
    for index, sensor in enumerate(sensors):
        if sensors_used[index] or sensor_outputs[index] < -80:
            color = -0.2
        else:
            color = (sensor_outputs[index] - minimum) / (maximum - minimum) + 0.2
        grid[sensor.x][sensor.y] = color
    intruders = set([(intru.x, intru.y) for intru in intruders])
    intru_unidentified = intruders - set(previous_identified)
    for intru in intruders:
        grid[intru[0]][intru[1]] = -0.6
    for intru in intru_unidentified:
        grid[intru[0]][intru[1]] = -1.2

    sns.set(style="white")
    plt.subplots(figsize=(10, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(grid, cmap=cmap, center=0, square=True, linewidth=1, cbar_kws={"shrink": .5})
    plt.xlabel('Unused Sensors and Unidentified Intruders')
    #plt.show()
    plt.savefig('visualize/localization/{}-proc-1-1'.format(fig))


def visualize_sensor_output(grid_len, intruders, sensor_outputs, sensors, threshold, fig):
    '''Visualize the intruders and sensor_output to have a better sense on deciding the threshold
    Args:
        grid_len (int):       length of grid
        intruders (list):     list of Transmitter objects
        sensor_output (list): list of float (RSSI)
        sensors (lists):      list of Sensor objects
    '''
    grid = np.zeros((grid_len, grid_len))
    # if np.max(sensor_outputs) > threshold:
    #     maximum = np.max(sensor_outputs[sensor_outputs < 0])
    #     minimum = np.min(sensor_outputs[sensor_outputs > threshold])
    # for index, sensor in enumerate(sensors):
    #     if sensor_outputs[index] > threshold:
    #         color = (sensor_outputs[index] - minimum) / (maximum - minimum)
    #     else:
    #         color = -0.2
    #     grid[sensor.x][sensor.y] = color
    #     #print((sensor[0], sensor[1]), sensor_output[index], '--', color)
    # for intr in intruders:
    #     grid[intr.x][intr.y] = -1

    for index, sensor in enumerate(sensors):
        color = sensor_outputs[index]
        grid[sensor.x][sensor.y] = color
    for intr in intruders:
        grid[intr.x][intr.y] = 1

    grid2 = np.copy(grid)
    for i in range(grid_len):
        for j in range(grid_len):
            grid2[i, j] = grid[j, grid_len-1-i]
    sns.set(style="white")
    plt.subplots(figsize=(10, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(grid2, cmap=cmap, center=0, square=True, linewidth=1, cbar_kws={"shrink": .5}, annot=True)
    plt.xlabel('red (>0) = sensor outputs; -1.2 = intruders (dark blue); -0.2 = is noise (light blue) ')
    #plt.show()
    plt.title('Intruders: ' + ' '.join(map(lambda intru: '({:2d}, {:2d})'.format(intru.x, intru.y), intruders)), fontsize=20)
    plt.savefig('visualize/localization/{}-sensor-output'.format(fig))


def visualize_q(grid_len, posterior, fig):
    '''Args: grid_len (int) posterior (np.array), 1D array
    '''
    grid = np.zeros((grid_len, grid_len))
    for x in range(grid_len):
        for y in range(grid_len):
            grid[x][y] = np.log10(posterior[x*grid_len + y])
    grid[grid == -np.inf] = -330

    grid2 = np.copy(grid)
    for i in range(grid_len):
        for j in range(grid_len):
            grid2[i, j] = grid[j, grid_len-1-i]

    plt.subplots(figsize=(8, 8))
    sns.heatmap(grid2, vmin=np.min(grid2), vmax=np.max(grid2), square=True, linewidth=0.5)
    plt.title('Q: ploting the exponent of Q, \n min exponent = -infinity (modify to -330 for plotting), max = {}'.format(round(np.max(grid), 3)))
    plt.savefig('visualize/localization/{}-q'.format(fig))


def visualize_q_prime(posterior, fig):
    '''
    '''
    grid_len = len(posterior)
    grid2 = np.copy(posterior)
    for i in range(grid_len):
        for j in range(grid_len):
            grid2[i, j] = posterior[j, grid_len-1-i]

    plt.subplots(figsize=(10, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(grid2, vmin=0, vmax=1, cmap=cmap, center=0, square=True, linewidth=0.5)
    plt.title('Q prime')
    plt.savefig('visualize/localization/{}-q-prime'.format(fig))


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
    f, ax = plt.subplots(figsize=(10, 10))
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

    grid2 = np.copy(grid)
    for i in range(grid_len):
        for j in range(grid_len):
            grid2[i, j] = grid[j, grid_len-1-i]

    sns.set(style="white")
    f, ax = plt.subplots(figsize=(10, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(grid2, cmap=cmap, center=0, square=True, linewidth=1, cbar_kws={"shrink": .5})
    plt.xlabel('1 = false alarm; 0.4 = accurate prediction; -1 = miss')
    #plt.show()
    plt.savefig('visualize/localization/{}-localization'.format(fig))


def visualize_splot(weight_global, folder, fig):
    '''Visualize the weights estimated in SPLOT
    Args:
        weight_global (np.array)
    '''
    weight_global[weight_global==0] = np.min(weight_global)
    
    grid_len = len(weight_global)
    grid2 = np.copy(weight_global)
    for i in range(grid_len):
        for j in range(grid_len):
            grid2[i, j] = weight_global[j, grid_len-1-i]

    plt.subplots(figsize=(10, 10))
    sns.heatmap(grid2, vmin=np.min(grid2), vmax=np.max(grid2), square=True, linewidth=0.5)
    plt.title('The estimated power (the weights of ridge regression)')
    plt.savefig('visualize/{}/{}'.format(folder, fig))


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
