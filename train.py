import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from utils import load_image_filenames
from feature_extractor import get_car_notcar_scaled_feature_vectors


def train_car_detector(cars_dirpath, notcars_dirpath, train_out, cspace='RGB', orient=9, pix_per_cell=8,
                       cell_per_block=2,
                       classifier_fn=LinearSVC):
    cars = load_image_filenames(cars_dirpath)
    notcars = load_image_filenames(notcars_dirpath)

    bin_params = {'size': (32, 32)}
    color_params = {'nbins': 32, 'bins_range': (0, 256)}
    hog_params = {'pixels_per_cell': (pix_per_cell, pix_per_cell), 'cells_per_block': (cell_per_block, cell_per_block),
                  'orientations': orient, 'visualize': False,
                  'feature_vector': True}
    res = get_car_notcar_scaled_feature_vectors(cars_dataset=cars, notcars_dataset=notcars, cspace=cspace,
                                                bin_params=bin_params, color_params=color_params, hog_params=hog_params,
                                                display_sample=False)

    car_features = res['car_features']
    notcar_features = res['notcar_features']
    scaled_X = res['scaled_X']

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    clf = classifier_fn()
    # Check the training time for the classifier
    t = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train the classifier...')
    # Check the score of the classifier
    print('Test Accuracy of the classifier = ', round(clf.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', clf.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with classifier')

    import pickle
    # save the classifier
    with open(train_out, 'wb') as fid:
        pickle.dump({'clf': clf, 'bin_params': bin_params, 'color_params': color_params, 'hog_params': hog_params,
                     'scaler': res['X_scaler']}, fid)

    # load it again
    with open(train_out, 'rb') as fid:
        res_loaded = pickle.load(fid)

    clf_loaded = res_loaded['clf']
    print('(Reloading Test) Test Accuracy of SVC = ', round(clf_loaded.score(X_test, y_test), 4))


def main():
    train_car_detector(cars_dirpath='./vehicles', notcars_dirpath='./non-vehicles', cspace='HSV', orient=9, pix_per_cell=8,
                       cell_per_block=2, train_out='train_svc_hsv_o9_pc8_cb2.pkl')
    train_car_detector(cars_dirpath='./vehicles', notcars_dirpath='./non-vehicles', cspace='YCrCb', orient=9,
                       pix_per_cell=8,
                       cell_per_block=2, train_out='train_svc_ycrcb_o9_pc8_cb2.pkl')
    train_car_detector(cars_dirpath='./vehicles', notcars_dirpath='./non-vehicles', cspace='RGB', orient=9, pix_per_cell=8,
                       cell_per_block=2, train_out='train_svc_rgb_o9_pc8_cb2.pkl')
    train_car_detector(cars_dirpath='./vehicles', notcars_dirpath='./non-vehicles', cspace='RGB', orient=11,
                       pix_per_cell=8,
                       cell_per_block=2, train_out='train_svc_rgb_o11_pc8_cb2.pkl')
    train_car_detector(cars_dirpath='./vehicles', notcars_dirpath='./non-vehicles', cspace='RGB', orient=9,
                       pix_per_cell=12,
                       cell_per_block=2, train_out='train_svc_rgb_o9_pc12_cb2.pkl')

if __name__ == '__main__':
    main()