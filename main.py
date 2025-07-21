import numpy as np
import pandas as pd
import cv2
import os
from sklearn.cluster import DBSCAN
from datetime import datetime
import support_functions as sf
import my_resonators as mr
from concurrent.futures import ProcessPoolExecutor
import matplotlib
from scipy.spatial.distance import cdist

#====Main Function==== 
def generate_frames(events, resolution, time_bin, my_resonators, my_resonators_freq, save_path):
    # Create df to be saved as CSV
    hedars = ["Frame Number", "Pixel Found", "FFT Freq"]
    values = ["N/A", "N/A", "N/A"]
    for i in range(0, len(my_resonators)):
        hedars.append(str(my_resonators_freq[i]) +" Peak to Peak")

        values += ["N/A"]
    hedars.append("Drone in frame")
    values.append("No")
    row_len = len(hedars)
    df = pd.DataFrame(columns = hedars)

    # Generics
    width, height = resolution
    frames = []
    start_time = 0
    end_time = events['t'].max()
    current_time = start_time

    # Array of moving objects
    mem_moving_obj = [] # [[{fetures of lest moving cluster}, number of time frames not used, drone was detacted]]

    # While loop over time frames
    while current_time < end_time:
        # Update mem_move
        i = 0
        while i < len(mem_moving_obj):
            if mem_moving_obj[i][1] == 2:
                mem_moving_obj.pop(i)
            else:
                mem_moving_obj[i][1] += 1
                i += 1
            
        frame = np.ones((height, width, 3), dtype=np.uint8) * 128
        # Filter events for the current time window
        mask = (events['t'] >= current_time) & (events['t'] < current_time + time_bin)
        current_events = events[mask]

        # Vectorized event mapping
        frame[current_events["y"].values, current_events["x"].values] = np.where(
            current_events["p"].values[:, None] == 0, [0, 0, 0], [255, 255, 255]
        )

        # Create diffrent posetive and negative event arrays
        pos_events = current_events[current_events['p'] == 1]
        neg_events = current_events[current_events['p'] == 0]
        # Get coordinates of posetive and negative events
        pos_coords = pos_events[['x', 'y']].values
        neg_coords = neg_events[['x', 'y']].values
        # Cluster using DBSCAN
        if len(pos_coords) > 0:
            pos_clustering = DBSCAN(eps=16, min_samples=150, n_jobs=-1).fit(pos_coords) # Pos events are more sparse 
            pos_labels = pos_clustering.labels_
        if len(neg_coords) > 0:
            neg_clustering = DBSCAN(eps=10, min_samples=150, n_jobs=-1).fit(neg_coords)
            neg_labels = neg_clustering.labels_

        # Set defult csv output
        for i in range(row_len) :
            values[i] = "N/A"
        values[-1] = "No"
        
        ### Boxes for all clustres check
        pos_features = sf.cluster_fetures(pos_coords, pos_labels)
        neg_features = sf.cluster_fetures(neg_coords, neg_labels)
        # Pos box 
        
        # Draw bounding box on pos
        for f in pos_features:
            cv2.rectangle(frame, f["min"], f["max"], (0, 255, 0), thickness=2)
        
        # Draw bounding box on neg
        for f in neg_features:
            cv2.rectangle(frame, f["min"], f["max"], (0, 0, 255), thickness=2)
        
        # Create movement vector
        pos_mid_points = np.array([f["centroid"] for f in pos_features])
        neg_mid_points = np.array([f["centroid"] for f in neg_features])
        
        # work on pos and negative cluster if there are any
        if len(pos_mid_points) > 0 and len(neg_mid_points) > 0:
            dist = cdist(neg_mid_points, pos_mid_points)
            nearest_pos_idx = np.argmin(dist, axis= 1)
            nearest_dist = dist[np.arange(len(neg_mid_points)), nearest_pos_idx]

            # Filter by size and distance
            for neg_idx, pos_idx in enumerate(nearest_pos_idx):
                if sf.cluster_correlation(neg_features[neg_idx],pos_features[pos_idx], size_diff = 3, dist_threshold = 2):
                    neg_mid = tuple(map(int, neg_mid_points[neg_idx]))
                    pos_mid = tuple(map(int, pos_mid_points[pos_idx]))
                    cv2.arrowedLine(frame, pos_mid, neg_mid, (255,0,0), thickness=2, tipLength=0.4)
                    merged_features = sf.merge_clusters(pos_features[pos_idx], neg_features[neg_idx])
                    cv2.rectangle(frame, merged_features["min"], merged_features["max"], (255, 50, 204), thickness=2)

                    updated = 0
                    for i in mem_moving_obj:
                        if sf.cluster_correlation(merged_features, i[0]):
                            i[0] = merged_features
                            i[1] = 0
                            updated = 1
                            break
                    if updated == 0:
                        mem_moving_obj.append([merged_features, 0, 0])
            frame_folder = f"{save_path}/frame_{int(current_time/time_bin)}"
            os.makedirs(frame_folder, exist_ok=True)
            cv2.imwrite(f"{frame_folder}/selected_clusters.png", frame)
            
            for k in range(len(mem_moving_obj)):
                if mem_moving_obj[k][1] == 0:
                    if mem_moving_obj[k][2] == 1:
                        values[-1] = "Yes"
                    current_features = mem_moving_obj[k][0]
                    # Create a mask for the merged cluster
                    merged_coords = current_features["points"]
                    mask = np.isin(current_events[['x', 'y']].values, merged_coords).all(axis=1)
                    cluster_events = current_events[mask].copy()
                    pixel_x, pixel_y = sf.pixel_with_most_transitions(cluster_events)

                    object_folder = f"{frame_folder}/cluster_{k}"
                    os.makedirs(object_folder, exist_ok=True)
                    
                    if mem_moving_obj[k][2] == 1:
                        cx, cy = int(current_features["centroid"][0]), int(current_features["centroid"][1])
                        xmin, ymin = current_features["min"]
                        xmax, ymax = current_features["max"]
                        cv2.line(frame, (xmin, cy), (xmax, cy), (0, 255, 255), thickness=2)
                        cv2.line(frame, (cx, ymin), (cx, ymax), (0, 255, 255), thickness=2)

                    # Extract events for the selected pixel
                    pixel_data = current_events[(current_events["x"] == pixel_x) & (current_events["y"] == pixel_y)]
                    if len(pixel_data) < 20:
                        continue
                    pixel_t = pixel_data['t'].values - pixel_data['t'].min()
                    pixel_p = pixel_data['p'].values * 2 - 1


                    # Create and Save signal 
                    values[1] = f"x{pixel_x}y{pixel_y}"
                    folder = f"{object_folder}/frame_{int(current_time/time_bin)}_pixel_{values[1]}"
                    
                    os.makedirs(folder, exist_ok=True)
                    my_signal, signal_xexis = sf.create_signal(pixel_p, pixel_t, pixel_x, pixel_y, folder)
                    # spike_signal, spike_signal_xexis = sf.create_spike_signal(pixel_p, pixel_t, pixel_x, pixel_y, folder)
                    # Polarity and FFT of signal
                    fft_freq = sf.event_pixel_fft(my_signal.copy(), pixel_x, pixel_y, folder) # Get max freq of FFT

                    if k == 0:
                        values[2] = fft_freq

                    args = [(j, my_signal, folder, my_resonators, my_resonators_freq, fft_freq, pixel_x, pixel_y) for j in range(len(my_resonators))]
                    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                        results = list(executor.map(sf.process_resonator, args))

                    # Fill `values`
                    if k == 0:
                        for j, max_amp, min_amp, mean_amp, diff, fft_diff in results: # Most values in result are not currently used
                            i = 3 + j
                            values[i] = diff
                            if diff > 300:
                                mem_moving_obj[k][2] = 1
                                values[-1] = "Yes"
                    if mem_moving_obj[k][2] == 1:
                        cx, cy = int(current_features["centroid"][0]), int(current_features["centroid"][1])
                        xmin, ymin = current_features["min"]
                        xmax, ymax = current_features["max"]
                        cv2.line(frame, (xmin, cy), (xmax, cy), (0, 255, 255), thickness=2)
                        cv2.line(frame, (cx, ymin), (cx, ymax), (0, 255, 255), thickness=2)

        # Save data to csv
        values[0] = int(current_time/time_bin) 
        df.loc[len(df)] = values

        frames.append(frame)
        current_time += time_bin

    df.to_csv(f"{save_path}/data.csv", index=False)
    return frames

if __name__ == "__main__":
    # Diable matplotlib gui for parrlel processing
    matplotlib.use('Agg')

    # Check run Time
    start_run_time  = datetime.now()
    #====Create resonators====
    r105 = mr.resonator_105()
    r110 = mr.resonator_110()
    r115 = mr.resonator_115()
    r128 = mr.resonator_128()
    r130 = mr.resonator_130()
    r166 = mr.resonator_166()
    r175 = mr.resonator_175()
    r221 = mr.resonator_221()
    r250 = mr.resonator_250()
    r268 = mr.resonator_268()
    r462 = mr.resonator_462()
    r509 = mr.resonator_509() 
    r636 = mr.resonator_636()
    r694 = mr.resonator_694()

    my_resonators = [
                    # r105, 
                    # r110, 
                    # r115, 
                    # r128,
                    # r130,
                    # r166,
                    # r175,
                    # r221,
                    # r250,
                    # r268,
                    # r462,
                    r509,
                    # r636,
                    # r694
                    ]
    my_resonators_freq = [
                        # 105,
                        # 110,
                        # 115,
                        # 128,
                        # 130,
                        # 166,
                        # 175,
                        # 221,
                        # 250,
                        # 268,
                        # 462,
                        509,
                        # 636,
                        # 694
                        ]

    #====Load Data====
    file_path = input("Enter data input path: ")
    save_path = input("Enter data output path: ")
    time_frame = 0.033 # Choose length of time for the time frame worked on, also determeans videos fps
    data = sf.load_data(file_path, 1e6, time_frame)
    resolution = (1280, 720)

    # Generate frames with optimized code
    frames = generate_frames(data, resolution, time_frame, my_resonators, my_resonators_freq, save_path)

    # Define video parameters
    output_path = "event_video.mp4"
    fps = int(1 / time_frame)  # Calculate FPS based on time bin

    # Initialize video writer
    video_writer = cv2.VideoWriter(f"{save_path}/event_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, resolution, isColor=True)

    # Write each frame to the video
    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved at: {save_path}/event_video.mp4")
    print(f"total run time: {datetime.now() - start_run_time}")