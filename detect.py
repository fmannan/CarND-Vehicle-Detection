import numpy as np
import cv2
from scipy.misc import imread
from utils import draw_boxes, convert_color
from feature_extractor import extract_features
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import pickle


class VehicleDetection:
    def __init__(self, trained_model, heatmap_threshold=3):
        # load model
        # load training data
        with open(trained_model, 'rb') as fid:
            res_loaded = pickle.load(fid)

        self.clf = res_loaded['svc']
        self.scaler = res_loaded['scaler']
        self.bin_params = res_loaded['bin_params']
        self.color_params = res_loaded['color_params']
        self.hog_params = res_loaded['hog_params']
        self.hog_params['feature_vector'] = True
        self.hog_params['visualize'] = False
        self.feature_extractor_params = {'bin_params': self.bin_params,
                                         'color_params': self.color_params,
                                         'hog_params': self.hog_params}
        self.heatmap_threshold = heatmap_threshold
        self.frame = 0
        self.windows = None

    def detect(self, image, windows=None, window_img_filename=None):
        if windows is None:
            windows_32 = slide_window(image, x_start_stop=None, y_start_stop=[398, 434],
                                      xy_window=(32, 32), xy_overlap=(0.5, 0.5))

            windows_64 = slide_window(image, x_start_stop=None, y_start_stop=[390, 450],
                                      xy_window=(64, 64), xy_overlap=(0.5, 0.5))

            windows_96 = slide_window(image, x_start_stop=None, y_start_stop=[390, 540],
                                      xy_window=(96, 96), xy_overlap=(0.5, 0.5))

            windows_128 = slide_window(image, x_start_stop=None, y_start_stop=[350, image.shape[0]],
                                       xy_window=(128, 128), xy_overlap=(0.25, 0.25))

            windows = windows_32 + windows_64 + windows_96 + windows_128
            self.windows = windows  # store the window coords for next time

        if window_img_filename:
            window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
            plt.figure()
            plt.imshow(window_img)
            plt.title('Anchor Windows')
            plt.savefig(window_img_filename)

        on_windows = search_windows(image, windows, self.clf, self.scaler, feature_extractor_params=self.feature_extractor_params)

        # window_img = draw_boxes(image, on_windows, color=(0, 0, 255), thick=6)
        # plt.figure()
        # plt.imshow(window_img)

        heatmap = build_heatmap(image.shape[:2], on_windows)
        # plt.figure()
        # plt.imshow(heatmap)
        #
        # plt.figure()
        # plt.imshow(heatmap > 3)

        heatmap[heatmap < self.heatmap_threshold] = 0

        labels = label(heatmap)
        imb_bbox = draw_labeled_bboxes(np.copy(image), labels)

        return imb_bbox

    def process_frame(self, frame):
        self.frame += 1
        return self.detect(frame, self.windows)


def process_video(video_filename, output_filename, vehicle_detector):
    clip = VideoFileClip(video_filename)
    clip = clip.fl_image(vehicle_detector.process_frame)
    clip.write_videofile(output_filename, audio=False)


def slide_window(img, x_start_stop=None, y_start_stop=None,
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    # Compute the span of the region to be searched
    # Compute the number of pixels per step in x/y
    # Compute the number of windows in x/y

    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
    # Calculate each window position
    # Append window position to list
    # Return the list of windows
    xy_step = np.array(xy_window) * np.array(xy_overlap)
    if x_start_stop is None:
        x_start_stop = [0, img.shape[1]]
    if y_start_stop is None:
        y_start_stop = [0, img.shape[0]]

    W = max(x_start_stop) - min(x_start_stop)
    H = max(y_start_stop) - min(y_start_stop)

    X, Y = np.mgrid[0:W - xy_window[1] + 1:xy_step[0], 0:H - xy_window[0] + 1:xy_step[1]].astype(np.int32)

    X += min(x_start_stop)
    Y += min(y_start_stop)

    X_bottom_right = X + xy_window[0]
    Y_bottom_right = Y + xy_window[1]

    window_list = list(np.stack(
        (np.stack((X.ravel(), Y.ravel()), axis=1), np.stack((X_bottom_right.ravel(), Y_bottom_right.ravel()), axis=1)),
        axis=1))
    return window_list


def search_windows(img, windows, clf, scaler, color_space='RGB', patch_size=(64, 64), feature_extractor_params=None):
    if color_space != 'RGB':
        img = convert_color(img, color_space)

    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], patch_size)
        # 4) Extract features for that window using single_img_features()
        features = extract_features(test_img, **feature_extractor_params)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def build_heatmap(image_shape, bbox_list):
    heatmap = np.zeros(image_shape)

    for bbox in bbox_list:
        heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1

    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img

def test_run():
    image = imread('./test_images/test1.jpg')
    vd = VehicleDetection('training_res.pkl', heatmap_threshold=2)
    img = vd.detect(image, window_img_filename='anchor_windows.png')

    plt.figure()
    plt.imshow(img)
    plt.title('Detected Cars')
    plt.savefig('detection_result.png')

    video_src = './test_video.mp4'
    video_out = './test_video_detect.mp4'
    #process_video(video_src, video_out, vd)

    #process_video('./project_video.mp4', './project_video_detect.mp4', vd)


def main():
    import sys
    import argparse
    import argcomplete

    # detect.py --src <video-in> --out <video-out>  --model <trained-model>
    parser = argparse.ArgumentParser(description='Run vehicle detection on video.\n Usage: ' + sys.argv[0] +
                                                 '--src <video-in> --out <video-out> --model <trained-model>')
    parser.add_argument('--src', type=str, help='Input video filename', required=True)
    parser.add_argument('--out', type=str, help='Output video filepath', required=True)
    parser.add_argument('--model', type=str, help='Model pickle file', required=True)
    parser.add_argument('--thresh', type=int, help='Heatmap threshold', default=2)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    print(args)

    vd = VehicleDetection(args.model, heatmap_threshold=args.thresh)

    process_video(args.src, args.out, vd)



if __name__ == '__main__':
    test_run()
    #main()
