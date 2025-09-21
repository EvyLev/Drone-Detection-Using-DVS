import numpy as np
import pandas as pd
import os
import cv2
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
from sctn.resonator import resonator
from my_resonators import clk_freq
# from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.distance import cdist
import random
# from openpyxl import load_workbook
# from openpyxl.styles import PatternFill
# import xlsxwriter
# from matplotlib.colors import ListedColormap
# import datashader as ds
# from datashader import transfer_functions as tf
# from datashader.colors import Greys9
# from holoviews.operation.datashader import datashade
# from matplotlib.colors import to_rgb
import matplotlib.cm as cm

color_list = [
    (255, 153, 153),  # #FF9999
    (153, 255, 153),  # #99FF99
    (153, 153, 255),  # #9999FF
    (255, 255, 153),  # #FFFF99
    (255, 178, 102),  # #FFB266
    (178, 102, 255),  # #B266FF
    (102, 255, 178),  # #66FFB2
    (255, 102, 178),  # #FF66B2
    (102, 178, 255),  # #66B2FF
    (178, 255, 102),  # #B2FF66
    (255, 102, 102),  # #FF6666
    (102, 255, 102),  # #66FF66
    (102, 102, 255),  # #6666FF
    (255, 255, 102),  # #FFFF66
    (255, 102, 255),  # #FF66FF
    (102, 255, 255),  # #66FFFF
    (178, 102, 102),  # #B26666
    (102, 178, 102),  # #66B266
    (102, 102, 178),  # #6666B2
    (178, 178, 102),  # #B2B266
]


#====Function to load data into workable 
def load_data(file_path, time = 1e6, time_frame = 0.33):
    '''
    Function intended to load data, gets: file path of the input data as csv, and the time the csv's time stamps speed
    we are using data with time stamps that are 1e6, so that is the defult, used to keep time_bin the same and change presived data rate
    the output is data, panda file with data names as x, y, p and t
    also gives the time_bin aka the time frame, in each time frame we eill look for a drone
    '''
    data = pd.read_csv(file_path, header=None, names=['x', 'y', 'p', 't'])

    # Convert time to seconds and normalize
    data['t'] = data['t'] / time 
    data['t'] -= data['t'].min()

    # Convert data types for efficiency
    data = data.astype({'x': np.uint16, 'y': np.uint16, 'p': np.int8, 't': np.float32})

    # Compute number of frames (not som important data)
    total_duration = data['t'].max()
    num_frames = int(total_duration / time_frame)
    print(f"Total frames: {num_frames}, Total duration: {total_duration:.2f} seconds")

    return data

#===== Create Square Signal =======
def create_signal(polarities, timestamps, pixel_x, pixel_y, folder):  # Timestamps are in seconds
    '''
    Function to create signal from the polarity of a selected pixel
    inputs:  
        polarities is an array of the polarities of the pixel, while timestamps is an array of the corresponding timestamps
        pixel_x, pixel_y is the placement of the selcted pixel, used for plot name, and folder is the save directory

    idea:
        normalize timestamsps to start at time 0, then move the first time stamp back so it will be picked up
        using the clk_freq as the sample_rate, create array t with same start as timestamps, and end time 20 smaples later 
        each time element in p is detrmaned on which is the last element in polaraities that t passes
        creates plot of the cretad signal (p is the y axis and t is the x axis) - remarks the number of events on plot
        returns p and t  
    '''
    # Inputs
    sample_rate = clk_freq

    # Define output time range
    timestamps -= timestamps[0]
    timestamps[0] -= 20/sample_rate
    t = np.arange(0, timestamps[-1] + 20 / sample_rate, 1 / sample_rate)

    current_event = 0
    p = np.zeros_like(t)
    for idx, i in enumerate(t):
        if current_event < len(timestamps) - 1:
            if timestamps[current_event + 1] < i:
                current_event += 1
        p[idx] = polarities[current_event]

    # Plot the result
    fig, ax = plt.subplots()
    ax.plot(t, p)
    ax.set_title(f"Pixel ({pixel_x}, {pixel_y}) polarity")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.text(0.98, 0.98,f"Number of Events: {len(polarities)}", 
            transform = ax.transAxes, 
            fontsize = 10,
            verticalalignment  = 'top',
            horizontalalignment = 'right',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.6))
    plt.savefig(os.path.join(folder, "signal.png"), dpi=300, bbox_inches='tight')
    plt.close()
    return p, t

#==== Optimized FFT Function ===
def event_pixel_fft( polarity_values, pixel_x, pixel_y, folder, is_spike = False):
    '''
    
    '''
    file_name = "fft_spectrum.png"
    if is_spike:
        file_name  = "fft_spike_spectrum.png"
    # Remove DC component
    signal_no_dc = polarity_values - np.mean(polarity_values)

    # Compute FFT
    N = len(signal_no_dc)
    fft_values = fft(signal_no_dc)
    frequencies = fftfreq(N, 1/clk_freq)

    # Get magnitude spectrum
    fft_magnitude = np.abs(fft_values)[:N // 2]
    frequencies = frequencies[:N // 2]

    # Find central frequency
    central_frequency = round(frequencies[np.argmax(fft_magnitude)])
    limit = min(np.argmax(fft_magnitude) + 5, N)

    # Save FFT plot
    fig, ax = plt.subplots()
    ax.plot(frequencies[:limit], fft_magnitude[:limit] / np.max(fft_magnitude))
    ax.scatter(central_frequency, np.max(fft_magnitude) / np.max(fft_magnitude), color='red')
    ax.set_title(f"FFT - Pixel ({pixel_x}, {pixel_y})")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Normalized Amplitude")
    ax.text(0.98, 0.98,f"Max Frequency:\n{central_frequency}", 
            transform = ax.transAxes, 
            fontsize = 10,
            verticalalignment  = 'top',
            horizontalalignment = 'right',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.6))
    plt.savefig(os.path.join(folder, file_name), dpi=300, bbox_inches="tight")
    plt.close()
    return central_frequency

#==== Plot Resonator On Signal ================
def plot_resonator_on_signal(
        my_resonator, input_signal, resonator_freq, pixel_x = 0, pixel_y = 0, clk_freq = clk_freq, 
        show=False,
        noise_std=0, end_with_n_zeros=0, start_with_n_zeros=0,
        folder = ""
        
):
    my_resonator.forget_logs()

    # Add Gaussian noise if specified
    signal = np.array(input_signal, dtype=np.float32)
    if noise_std > 0:
        signal += np.random.normal(0, noise_std, len(signal))

    # Normalize the signal (to avoid saturation or scale issues)
    if np.max(np.abs(signal)) > 0:
        signal /= np.max(np.abs(signal))

    # Optional zero padding at start
    start_zeros = []
    if start_with_n_zeros > 0:
        start_zeros = np.zeros(int(clk_freq / resonator_freq * start_with_n_zeros))
        my_resonator.input_full_data(start_zeros)

    my_resonator.input_full_data(signal)

    # Optional zero padding at end
    end_zeros = []
    if end_with_n_zeros > 0:
        end_zeros = np.zeros(int(clk_freq / resonator_freq * end_with_n_zeros))
        my_resonator.input_full_data(end_zeros)

    # Plot spikes for each logged neuron
    spikes_window_size = 500
    output_neuron = my_resonator.neurons[-1]
    y_events = output_neuron.out_spikes()
    total_len = len(signal) + len(start_zeros) + len(end_zeros)
    y_spikes = np.zeros(total_len)
    y_spikes[y_events] = 1
    y_spikes = np.convolve(y_spikes, np.ones(spikes_window_size, dtype=int), 'valid')

    if pixel_x == 0 and pixel_y == 0:
        plt.title(f'Resonator Response to Input Signal')
    else:
        plt.title(f'Resonator, {resonator_freq}Hz, Response to Input Signal in Pixel ({pixel_x}, {pixel_y})')
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.plot( np.linspace(0, total_len / (clk_freq / resonator_freq), len(y_spikes)), y_spikes)


    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(folder, str(resonator_freq)), dpi = 300, bbox_inches = 'tight')
        plt.close()
    threshold1 = 300
    threshold2 = 200
    # Rising edge: signal goes from below to above threshold
    rising_crossings = np.where((y_spikes[:-1] < threshold1) & (y_spikes[1:] >= threshold1))[0]

    # Falling edge: signal goes from above to below threshold
    falling_crossings = np.where((y_spikes[:-1] >= threshold2) & (y_spikes[1:] < threshold2))[0]

    return np.max(y_spikes), np.min(y_spikes), np.mean(y_spikes) #, len(rising_crossings), len(falling_crossings)

# Worker function
def process_resonator(args):
    j, my_signal, save_path, my_resonators, my_resonators_freq, fft_freq, pixel_x, pixel_y = args
    folder = save_path
    max_amp, min_amp, mean_amp = plot_resonator_on_signal(
        my_resonators[j], my_signal, my_resonators_freq[j], pixel_x, pixel_y,
        folder=folder, end_with_n_zeros=5, start_with_n_zeros=10
    )
    diff = max_amp - min_amp
    fft_diff = abs(fft_freq - my_resonators_freq[j])  # Use global or temp variable
    return (j, max_amp, min_amp, mean_amp, diff, fft_diff)

# Choose Pixel
def pixel_with_most_transitions(events):
    # Extract position, time, polarity columns
    xy = events[["x", "y"]].values
    t = events["t"].values
    p = events["p"].values  # 0 or 1

    combined = np.column_stack((xy, t, p))
    sort_idx = np.lexsort((combined[:, 2], combined[:, 1], combined[:, 0]))
    combined_sorted = combined[sort_idx]

    xy_keys, indices, counts = np.unique(combined_sorted[:, :2], axis=0, return_index=True, return_counts=True)

    max_transitions = -1
    best_pixel = None

    for i in range(len(xy_keys)):
        start, end = indices[i], indices[i] + counts[i]
        polarity = combined_sorted[start:end, 3]
        transitions = np.sum(polarity[1:] != polarity[:-1])
        if transitions > max_transitions:
            max_transitions = transitions
            best_pixel = xy_keys[i]

    return int(best_pixel[0]), int(best_pixel[1])

def cluster_fetures(coords, labels):
    # get importent features of clusters
    features = []
    for label in set(labels):
        if label == -1:
            continue
        cluster_points = coords[labels == label]
        centroid = cluster_points.mean(axis=0).astype(int)

        xmin, ymin = cluster_points.min(axis=0)
        xmax, ymax = cluster_points.max(axis=0)
        box_size = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)  # diagonal size

        features.append({
            "label": label,
            "centroid": centroid,
            "Vstart": None, # Vector start only in moving cluster
            "Vend": None, # Vector end only in moving cluster
            "min": (xmin, ymin),
            "max":(xmax, ymax),
            "points": cluster_points,
            "box_size": box_size,
            "size": len(cluster_points)
        })
    return features

# def associate_bubbles(current_clusters, previous_clusters, max_distance = 10):

def merge_clusters(pos_cluster, neg_cluster):
    # Mearge 2 clusters (pos & neg) into one moving cluster
    merged_points = np.vstack([pos_cluster["points"], neg_cluster["points"]])
    xmin, ymin = merged_points.min(axis=0)
    xmax, ymax = merged_points.max(axis=0)
    centroid = merged_points.mean(axis=0).astype(int)

    merged_cluster = {
        "label": f"merged_{pos_cluster['label']}_{neg_cluster['label']}",
        "centroid": centroid,
        "min": (xmin, ymin),
        "max": (xmax, ymax),
        "Vstart": pos_cluster["centroid"],
        "Vend": neg_cluster["centroid"],
        "points": merged_points,
        "box_size": np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2),
        "size": len(merged_points)
    }

    return merged_cluster

def cluster_correlation(features1, features2, size_diff = 2, dist_threshold = 1.5, angle_thresh = 30, speed_thresh = 0.3, size_tol = 5000):
    # Checks if 2 clusters are close enough and similer size to be labeld as correlated
    # Size consistancy 
    ratio = min(features1["size"]/  features2["size"],features2["size"] / features1["size"]) if features2["size"] > 0 and features1["size"] > 0 else 0 # The size of the object is consistent
    if ratio < 1/size_diff:
        return False
    avg_size = (features1["box_size"] + features2["box_size"])/2 # Avreage Box size
    # Checking if centers of mass close enogh (enogh to check if 2 non moving clusters are correlated)
    x1, y1 = features1["centroid"]
    x2, y2 = features2["centroid"]  
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    if dist > dist_threshold * avg_size: # Avreage box size times the threshold
        return False
    
    # Corralation between 2 moving clusters
    if not(features1["Vstart"] is None or features2["Vstart"] is None):         # Movement Vectors
        v1 = np.array(features1["Vend"]) - np.array(features1["Vstart"]) 
        v2 = np.array(features2["Vend"]) - np.array(features2["Vstart"])

        # Magnitudes
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        # Normalize vectors
        u1, u2 = v1 / n1, v2 / n2

        # --- Angle difference (allow flip) ---
        cos_theta = np.dot(u1, u2)
        angle_diff = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

        # Adaptive angle threshold: smaller clusters → looser tolerance
        adaptive_angle_thresh = angle_thresh + size_tol / (avg_size + 1)  # big tolerance for small objects
        if not (angle_diff <= adaptive_angle_thresh or abs(angle_diff - 180) <= adaptive_angle_thresh):
            return False

        # speed_diff = abs(scaled_n1 - scaled_n2) / max(scaled_n1, scaled_n2, 1e-6)
        speed_diff = n1/n2
        adaptive_speed_thresh = max(speed_thresh - 15 / avg_size, speed_thresh/20)  # smaller cluster → bigger tolerance

        if speed_diff < adaptive_speed_thresh or speed_diff > 1 / adaptive_speed_thresh:
            return False

    return True

# Draw cluster outling 
def draw_cluster_outline(img, points, color=(0,255,0), thickness=2, epsilon = 2.0):
    pts = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(pts)
    approx = cv2.approxPolyDP(hull, epsilon, True)  # simplify outline
    cv2.polylines(img, [approx], isClosed=True, color=color, thickness=thickness)

#=====================================================================================
#================================= TEST ==============================================
#=====================================================================================
# Input frame width and height and list of all boxes found, outputs pixel that is not in any box
def get_random_background_pixel(width, height, object_boxes):
    while True:
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        inside = False
        for (xmin, ymin, xmax, ymax) in object_boxes:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                inside = True
                break
        if not inside:
            return x, y
        
def cluster_matrix(save_path, height, width, features, cluster_type, frame_number, heatmap_threshold=50000):
    n_clusters = len(features)
    colors = cm.get_cmap('tab20', n_clusters)

    fig, ax = plt.subplots(figsize=(10, 7))
    
    for idx, f in enumerate(features):
        points = f["points"]
        ax.scatter(points[:,0], points[:,1], s=1, 
                   color=colors(idx), label=f"Cluster {f['label']}: {f['size']} events")
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()  # match image coords
    ax.set_xlabel("X Pixel Coordinate")
    ax.set_ylabel("Y Pixel Coordinate")
    ax.set_title(f"Frame {frame_number}, {cluster_type} Cluster Points", fontsize=14)
    ax.legend(loc="upper right", markerscale=4, fontsize=9, frameon=True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
#=====================================================================================
#================================= NOT IN USE ========================================
#=====================================================================================
#==== util function to plot how resonator react on a signal with given frequencies====
def plot_resonator_on_freq(
        my_resonator, signal_freq, clk_freq, resonator_freq, time=20, show=True, weights=None,
        neurons_log_spikes=None, noise_std=0,
        end_with_n_zeros=0, start_with_n_zeros=0
):
    my_resonator.forget_logs()
    neurons_log_spikes = neurons_log_spikes or []
    for i in neurons_log_spikes:
        my_resonator.log_out_spikes(i)

    if type(signal_freq) is not list:
        signal_freq = [signal_freq]

    phases = time * resonator_freq
    weights = weights or np.ones(len(signal_freq))

    x = np.linspace(0, phases / resonator_freq, int(phases * clk_freq / resonator_freq))
    t = x * 2 * np.pi * signal_freq[0]
    sine_wave = np.sin(t) * weights[0]
    for w, f in zip(weights[1:], signal_freq[1:]):
        t = x * 2 * np.pi * f
        sine_wave += np.sin(t) * w
    sine_wave += np.random.normal(0, noise_std, len(sine_wave))
    sine_wave /= np.max(sine_wave)

    start_zeros = []
    if start_with_n_zeros > 0:
        start_zeros = np.zeros(int(clk_freq / resonator_freq * start_with_n_zeros))
        my_resonator.input_full_data(start_zeros)
    my_resonator.input_full_data(sine_wave)
    ends_zeros = []
    if end_with_n_zeros > 0:
        ends_zeros = np.zeros(int(clk_freq / resonator_freq * end_with_n_zeros))
        my_resonator.input_full_data(ends_zeros)

    spikes_window_size = 500

    for i in neurons_log_spikes:
        output_neuron = my_resonator.neurons[i]
        y_events = output_neuron.out_spikes()
        y_spikes = np.zeros(len(sine_wave) + len(ends_zeros) + len(start_zeros))
        y_spikes[y_events] = 1
        y_spikes = np.convolve(y_spikes, np.ones(spikes_window_size, dtype=int), 'valid')
        plt.title(f'signal freq {signal_freq}')
        plt.plot(np.linspace(0, phases + end_with_n_zeros + start_with_n_zeros, len(y_spikes)), y_spikes,
                 label=f'SN {i}')

    if show:
        plt.show()

#====Function to plot the emitad spikes====
def plot_emitted_spikes(network, x_stop, nid=-1, label=None, spare_steps=10, folder = None, frequency = "0", is_spike = False):
    spikes_neuron = network.neurons[nid]
    y_events = spikes_neuron.out_spikes()
    if len(y_events) == 0:
        return "N/A", "N/A"
    y_spikes = np.zeros(y_events[-1] + 1)
    y_spikes[y_events] = 1
    y_spikes = np.convolve(y_spikes, np.ones(500, dtype=int), 'valid')
    y_spikes = y_spikes[::spare_steps]
    x = np.linspace(0, x_stop, len(y_spikes))
    plt.title('Resonator Output')
    plt.ylabel('Spikes per W500')
    plt.xlabel('Frequency')
    plt.plot(x, y_spikes, label=label)
    if folder == None:
        plt.close()
        return "N/A"
    if is_spike:
        frequency += "_spike"
    frequency = frequency + "_resonator.png"
    plt.savefig(os.path.join(folder, frequency), dpi = 300, bbox_inches = 'tight')
    plt.close()
    return np.max(y_spikes), np.max(y_spikes) - np.mean(y_spikes), np.max(y_spikes) - np.min(y_spikes) #CHANGE: Give back max - mean & max - min instead of only max
