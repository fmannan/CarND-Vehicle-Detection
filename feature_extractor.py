import numpy as np
import cv2
from skimage.feature import hog
from scipy.misc import imread
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from utils import convert_color, load_images
import copy


def check_param(dict_var, key, default_val):
    return dict_var[key] if key in dict_var else default_val


class VehicleFeatureExtractor(sklearn.base.TransformerMixin):
    def __init__(self, cspace='RGB', spatial_bin_size=(32, 32), color_nbins=32, color_bin_range=(0, 256),
                 hog_pix_per_cell=(8, 8), hog_cells_per_block=(2, 2), hog_orientation=9,
                 hog_visualize=False, hog_feature_vector=True):
        self.set_params(cspace=cspace, spatial_bin_size=spatial_bin_size, color_nbins=color_nbins,
                        color_bin_range=color_bin_range,
                        hog_pix_per_cell=hog_pix_per_cell, hog_cells_per_block=hog_cells_per_block,
                        hog_orientation=hog_orientation,
                        hog_visualize=hog_visualize, hog_feature_vector=hog_feature_vector)

    @staticmethod
    def __construct_params(**params):
        cspace = params['cspace']
        bin_params = {'size': params['spatial_bin_size']}
        color_params = {'nbins': params['color_nbins'], 'bins_range': params['color_bin_range']}
        hog_params = {'pixels_per_cell': params['hog_pix_per_cell'],
                      'cells_per_block': params['hog_cells_per_block'],
                      'orientations': params['hog_orientation'],
                      'visualize': check_param(params, 'hog_visualize', False),
                      'feature_vector': check_param(params, 'hog_feature_vector', True)}
        return {'cspace': cspace, 'bin_params': bin_params, 'color_params': color_params,
                'hog_params': hog_params}

    def fit(self, X, y=None, **fit_params):
        print('VehicleFeatureExtractor.fit', y, fit_params)
        return self  #.transform(X)

    @staticmethod
    def extract_feature(im, **params):
        #print(params)
        color_bins_features = bin_spatial(im, **params['bin_params']) if params['bin_params'] is not None else []
        color_hist_features = color_hist(im, **params['color_params']) if params['color_params'] is not None else []
        im_hog_features = hog_features(im, **params['hog_params']) if params['hog_params'] is not None else []
        return np.concatenate((color_bins_features, color_hist_features, im_hog_features))

    def transform(self, X):
        print(X.shape)
        params = self.__construct_params(**self.get_params(deep=False))
        feature_list = []
        for idx in range(X.shape[0]):
            #if idx % 1000 == 0:
            #    print('iter', idx)
            im = np.squeeze(X[idx])
            if params['cspace'] != 'RGB':
                im = convert_color(im, params['cspace'])
            feature_list.append(self.extract_feature(im, **params))
        print('transform done...scaling...')
        feature_vec = np.vstack(feature_list).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(feature_vec)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(feature_vec)
        print('scaling done ', scaled_X.shape)
        return scaled_X

    # def fit_transform(self, X, y=None, **fit_params):
    #     params = self.__construct_params(**fit_params)
    #     color_bins_features = bin_spatial(X, **params['bin_params']) if params['bin_params'] is not None else []
    #     color_hist_features = color_hist(X, **params['color_params']) if params['color_params'] is not None else []
    #     im_hog_features = hog_features(X, **params['hog_params']) if params['hog_params'] is not None else []
    #     return np.concatenate((color_bins_features, color_hist_features, im_hog_features))

    def get_params(self, deep=True):
        print('VehicleFeatureExtractor.get_params')
        param_dict = dict(cspace=self.cspace, spatial_bin_size=self.spatial_bin_size, color_nbins=self.color_nbins,
                          color_bin_range=self.color_bin_range,
                          hog_pix_per_cell=self.hog_pix_per_cell, hog_cells_per_block=self.hog_cells_per_block,
                          hog_orientation=self.hog_orientation,
                          hog_visualize=self.hog_visualize, hog_feature_vector=self.hog_feature_vector)
        print(param_dict)
        if deep:
            return copy.deepcopy(param_dict)
        return param_dict

    def set_params(self, cspace='RGB', spatial_bin_size=(32, 32), color_nbins=32, color_bin_range=(0, 256),
                 hog_pix_per_cell=(8, 8), hog_cells_per_block=(2, 2), hog_orientation=9,
                 hog_visualize=False, hog_feature_vector=True):
        #print('VehicleFeatureExtractor.set_params')
        print(cspace, spatial_bin_size, hog_pix_per_cell, hog_cells_per_block)
        self.cspace = cspace
        self.spatial_bin_size = spatial_bin_size
        self.color_nbins = color_nbins
        self.color_bin_range = color_bin_range
        self.hog_pix_per_cell = hog_pix_per_cell
        self.hog_cells_per_block = hog_cells_per_block
        self.hog_orientation = hog_orientation
        self.hog_visualize = hog_visualize
        self.hog_feature_vector = hog_feature_vector


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def hog_features(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False,
                 feature_vector=True):
    if len(img.shape) > 2:
        img = np.mean(img, axis=-1)
    return hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block, visualise=visualize, feature_vector=feature_vector)


def extract_features(img, bin_params=None, color_params=None, hog_params=None):
    """Extract features from a list of images
    NOTE: The caller performs the color conversion
    Used for extracting features from an image patch

    :param img: A single image
    :param bin_params: parameters for spatial binning
    :param color_params: parameters for color histogram binning
    :param hog_params: parameters for hog feature extractor
    :return:
    """
    color_bins_features = bin_spatial(img, **bin_params) if bin_params is not None else []
    color_hist_features = color_hist(img, **color_params) if color_params is not None else []
    im_hog_features = hog_features(img, **hog_params) if hog_params is not None else []
    return np.concatenate((color_bins_features, color_hist_features, im_hog_features))


def extract_features_imgs(imgs, cspace='RGB', **params):
    """Extract features from a list of images
    Useful for extracting features from training and test images

    :param imgs:
    :param bin_params: parameters for spatial binning
    :param color_params: parameters for color histogram binning
    :param hog_params: parameters for hog feature extractor
    :return:
    """
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    # Read in each one by one
    # apply color conversion if other than 'RGB'
    # Apply bin_spatial() to get spatial color features
    # Apply color_hist() to get color histogram features
    # Get HOG features
    # Append the new feature vector to the features list
    # Return list of feature vectors
    for image in imgs:
        if type(image) is str:
            image = imread(image)

        if cspace is not 'RGB':
            image = convert_color(image, cspace)

        features.append(extract_features(image, **params))

    return features


def get_car_notcar_scaled_feature_vectors(cars_dataset, notcars_dataset, cspace, bin_params, color_params, hog_params,
                                          display_sample=False):

    car_features = extract_features_imgs(cars_dataset, cspace=cspace, bin_params=bin_params,
                                         color_params=color_params, hog_params=hog_params)
    notcar_features = extract_features_imgs(notcars_dataset, cspace=cspace, bin_params=bin_params,
                                            color_params=color_params, hog_params=hog_params)
    example_fig_filename = 'im_feature_norm.png'
    # Normalize the features
    if len(car_features) > 0:
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        ### For display
        if display_sample:
            car_ind = np.random.randint(0, len(cars_dataset))
            notcar_ind = np.random.randint(0, len(notcars_dataset))
            img = imread(cars_dataset[car_ind])
            notcar_img = imread(notcars_dataset[notcar_ind])
            # Plot an example of raw and scaled features
            fig = plt.figure(figsize=(12, 4))
            plt.subplot(131)
            plt.imshow(img)
            plt.title('Original Image')
            plt.subplot(132)
            plt.plot(X[car_ind])
            plt.title('Raw Features')
            plt.subplot(133)
            plt.plot(scaled_X[car_ind])
            plt.title('Normalized Features')
            fig.tight_layout()
            plt.savefig(example_fig_filename)

            # show corresponding HoG image
            vis_hog_params = hog_params
            vis_hog_params['visualize'] = True
            vis_hog_params['feature_vector'] = False
            _, hog_img = hog_features(img, **vis_hog_params)
            plt.figure()
            plt.imshow(hog_img)
            plt.gray()
            plt.title('HoG Features')
            plt.savefig('hog_image.png')

            plt.figure()
            plt.subplot(121)
            plt.imshow(img)
            plt.title('Car')
            plt.subplot(122)
            plt.imshow(notcar_img)
            plt.title('Not Car')
            plt.savefig('car_notcar.png')

    else:
        ValueError('Empty feature vector')

    return {'X': X, 'scaled_X': scaled_X, 'X_scaler': X_scaler,
            'car_features': car_features, 'notcar_features': notcar_features}


def example_run():
    from utils import load_image_filenames
    cars = load_image_filenames('./vehicles')
    notcars = load_image_filenames('./non-vehicles')

    bin_params = {'size': (32, 32)}
    color_params = {'nbins': 32, 'bins_range': (0, 256)}
    hog_params = {'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2), 'orientations': 9, 'visualize': False,
                  'feature_vector': True}
    get_car_notcar_scaled_feature_vectors(cars_dataset=cars, notcars_dataset=notcars, cspace='RGB',
                                          bin_params=bin_params, color_params=color_params, hog_params=hog_params,
                                          display_sample=True)

def show_car_notcar_sample(n_samples, out_filename):
    from utils import load_image_filenames
    cars = load_image_filenames('./vehicles')
    notcars = load_image_filenames('./non-vehicles')

    cars_inds = np.random.randint(0, len(cars), n_samples)
    notcars_inds = np.random.randint(0, len(notcars), n_samples)

    plt.figure(figsize=(2, 4))
    for idx in range(n_samples):
        car_img = imread(cars[cars_inds[idx]])
        notcar_img = imread(notcars[notcars_inds[idx]])

        plt.subplot(n_samples, 2, idx * 2 + 1)
        plt.imshow(car_img)
        plt.axis('off')
        if idx == 0:
            plt.title('Cars')

        plt.subplot(n_samples, 2, idx * 2 + 2)
        plt.imshow(notcar_img)
        plt.axis('off')
        if idx == 0:
            plt.title('Not Cars')

    plt.savefig(out_filename)

if __name__ == '__main__':
    show_car_notcar_sample(4, 'car_notcar_samples.png')
    #example_run()