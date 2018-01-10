import json
import subprocess
import os.path
import os
import pandas as pd
import numpy as np
from pyproj import Proj, transform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from datetime import datetime

class Segmenter(object):
    def __init__(self, points, src_epsg, dst_epsg, src_path):
        self.points = points
        self.src_epsg = src_epsg
        self.dst_epsg = dst_epsg
        self.src_path = src_path

    def create_polygon(self):

        polygon = "POLYGON(("
        count = 0

        for newPoint in self.points[0]:

            if count % 10 == 0:
                inProj = Proj(init=self.src_epsg)
                outProj = Proj(init=self.dst_epsg)

                x1, y1 = newPoint[0], newPoint[1]
                x2, y2 = transform(inProj,
                                   outProj,
                                   x1,
                                   y1)

                polygon += str(x2) + " " + str(y2) + ","

            count += 1

        inProj = Proj(init=self.src_epsg)
        outProj = Proj(init=self.dst_epsg)

        x1, y1 = self.points[0][0][0], self.points[0][0][1]
        x2, y2 = transform(inProj,
                           outProj,
                           x1,
                           y1)

        polygon += str(x2) + " " + str(y2) + ","
        polygon = polygon[:-1]
        polygon += "))"

        return polygon

    def segment_in(self, out_name):
        polygon = self.create_polygon()
        pdal_Command = ("docker run -v /home/daire/Code/CS401_Projects/Project:/data " +
                        "pdal/pdal:1.5 pdal pipeline " +
                        "data/assets/pipelines/in.json " +
                        "--readers.las.filename=data/" +
                        self.src_path + " "
                        "--filters.crop.polygon=\"" + polygon + "\" " +
                        "--writers.las.filename=data/scratch/" +
                        out_name)
        os.system(pdal_Command)

    def segment_out(self, out_name):
        polygon = self.create_polygon()
        pdal_Command = ("docker run -v /home/daire/Code/CS401_Projects/Project:/data " +
                        "pdal/pdal:1.5 pdal pipeline " +
                        "data/assets/pipelines/out.json " +
                        "--readers.las.filename=data/" +
                        self.src_path + " "
                        "--filters.crop.polygon=\"" + polygon + "\" " +
                        "--writers.las.filename=data/scratch/" +
                        out_name)
        os.system(pdal_Command)

    def subsample(self, out_name):
        polygon = self.create_polygon()
        pdal_Command = ("docker run -v /home/daire/Code/CS401_Projects/Project:/data " +
                        "pdal/pdal:1.5 pdal pipeline " +
                        "data/assets/pipelines/subsample.json " +
                        "--readers.las.filename=data/" +
                        self.src_path + " "
                        "--writers.las.filename=data/scratch/" +
                        out_name)
        os.system(pdal_Command)
        return out_name


class Data_Preprocessor(object):

    def __init__(self, src_epsg, dst_epsg, scratch_path):
        self.src_epsg = src_epsg
        self.dst_epsg = dst_epsg
        self.src_path = scratch_path

    def las2txt(self, src_path, dst_path=None):

        if dst_path is None:
            dst_path = src_path[:-3] + "txt"

        cmd = ("las2txt " +
               "-i " +
               src_path + " " +
               "--parse xyzinrM " +
               "--delimiter , " +
               "--labels " +
               "--header " +
               "-o " +
               dst_path)

        if not os.path.exists(dst_path):
            print("Got here")
            subprocess.call(cmd.split(" "))
            return dst_path
        else:
            print("File Already Excists")
            return dst_path

    def split_data(self):
        print("Datasplitter")


if __name__ == "__main__":
    with open('assets/road_edge.geojson') as data_file:
        data = json.load(data_file)

    road_edge_coords = data["features"][0]["geometry"]["coordinates"]

    road_points = "road.las"
    bank_points = "bank.las"

    my_segmenter = Segmenter(road_edge_coords,
                             "epsg:3857",
                             "epsg:2157",
                             "scratch/segmented_road.las")
    new_path = "scratch/" + my_segmenter.subsample("subsample.las")
    my_segmenter.src_path = new_path

    my_segmenter.segment_in(road_points)
    my_segmenter.segment_out(bank_points)


    my_preprocessor = Data_Preprocessor("EPSG:2157",
                                        "EPSG:3857",
                                        "/home/daire/Code/CS401_Projects/Project/scratch")

    road_points = my_preprocessor.las2txt("scratch/road.las")
    bank_points = my_preprocessor.las2txt("scratch/bank.las")

    ###########################################################################
    #  Partitioning Data                                                     #
    ###########################################################################

    df_road = pd.read_csv(road_points)
    df_road["class"] = 1

    print("\n\n\nHead of df_road")
    print(df_road.head())

    df_bank = pd.read_csv(bank_points)
    df_bank["class"] = 0

    print("\n\n\nHead of df_bank")
    print(df_bank.head())

    df_combined = pd.concat([df_road, df_bank])
    print("\n\n\nHead of df_combined")
    print(df_combined.head())

    X  = df_combined.loc[:, 'Z':'Return Number']
    y = df_combined.loc[:, "class"]

    print('\n\n\nClass labels:', np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=1,
                                                        stratify=y)

    ###########################################################################
    #  Scaling Data                                                           #
    ###########################################################################
    # Fit the StandardScaler class only once on the training data and use those
    # parameters to transform the test set or any new data point.
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)
    ###########################################################################
    #  Dimensionality reduction                                               #
    ###########################################################################
    from sklearn.decomposition import PCA as sklearnPCA
    sklearn_pca = sklearnPCA(n_components=2)
    X_train_pca = sklearn_pca.fit_transform(X_train_std)

    print(X_train_pca)
    ###########################################################################
    #  Random Forest                                                     #
    ###########################################################################
    from sklearn.ensemble import RandomForestClassifier
    param_grid = {"n_estimators" :[10, 50, 100],
                  "max_depth":[3, 10,  None],
                  "min_samples_split": [2, 3, 10, 30],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}


    random_forest = RandomForestClassifier(random_state=1)

    startTime = datetime.now()

    grid_search = GridSearchCV(random_forest,
                               param_grid,
                               n_jobs=-1,
                               cv=10,
                               scoring='accuracy')

    grid_search_RandomForest = grid_search.fit(X_train, y_train)

    print("Random Forest took :  "  str(datetime.now() - startTime))
    print(grid_search_RandomForest.best_score_)

    print(grid_search_RandomForest.best_params_)
    ###########################################################################
    # SVM                                                                     #
    ###########################################################################
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    pipe_svc = make_pipeline(SVC(random_state=1))
    param_range = [0.0001, 0.001, 0.01, 0.1,
                   1.0, 10.0, 100.0, 1000.0]
    param_range = [1.0]
    param_grid = [{'svc__C': param_range,
                  'svc__kernel': ['linear']},
                  {'svc__C': param_range,
                   'svc__gamma': param_range,
                   'svc__kernel': ['rbf']}]
    param_grid = [{'svc__C': param_range,
                  'svc__kernel': ['linear']}
                  ]

    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=-1)
    print("Fitting SVM")
    print(param_range)
    gs = gs.fit(X_train, y_train)
    print(gs.best_score_)

    print(gs.best_params_)
    ###########################################################################
    # Random Forest                                                           #
    ###########################################################################
##################################
    # Nueal Network                                                           #
    ###########################################################################
    from sklearn.neural_network import MLPClassifier
    """mlp = MLPClassifier(hidden_layer_sizes=(4,4,4),
                        max_iter=500)

    mlp.fit(X_train,y_train)

    predictions = mlp.predict(X_test)

    from sklearn.metrics import classification_report,confusion_matrix

    print(confusion_matrix(y_test,predictions))

    print(classification_report(y_test,predictions))"""

    pipe_mlp = MLPClassifier() #max_iter=500))
    parameters = {'hidden_layer_sizes': [(100,1), (100,2)  ]} # , (5, 5, 5), (3, 3, 3)]}
    """parameters={
                'learning_rate': ["constant", "invscaling", "adaptive"],
                'hidden_layer_sizes': [(100,1), (100,2), (100,3)],
                'alpha': [10.0 ** -np.arange(1, 7)],
                'activation': ["logistic", "relu", "Tanh"]
                }"""
    gs = GridSearchCV(estimator=pipe_mlp,
                      param_grid=parameters,
                      n_jobs=-1)
    print("Fitting MLP")
    print(parameters)
    gs = gs.fit(X_train, y_train)
    print(gs.best_score_)

    print(gs.best_params_)
