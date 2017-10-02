from feature_extractor import VehicleFeatureExtractor
from utils import load_dataset

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from pprint import pprint
from time import time
import logging

if __name__ == '__main__':
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # pipeline experiment
    # set of transformers and then estimator
    # test grid search with pipeline

    pipeline = Pipeline([('feature', VehicleFeatureExtractor()),
                         ('clf', SVC()),
                         ])

    parameters = {'feature__cspace': ('RGB', 'HSV', 'YCrCb'),
                  'feature__hog_pix_per_cell': ((4, 4), (8, 8), (9, 9)),
                  'feature__hog_cells_per_block': ((1, 1), (2, 2), (3, 3)),
                  'clf__kernel': ('rbf', 'linear')}

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=2, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)

    print('loading dataset')
    X, y = load_dataset('./vehicles', './non-vehicles')
    print('dataset loaded.')

    t0 = time()
    grid_search.fit(X, y)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))





