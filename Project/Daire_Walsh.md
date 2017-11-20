# Daire Walsh

#### Project Name
Auto Road Edge Extraction

#### Author
Daire Walsh


#### Introduction
A Mobile Mapping System(MMS) is usually mounted on a van or car. Data is recorded by the MMS as it moves across a road. A MMS will have one or more laser scanners, a GPS and orientation system and possibly one or more camera's. LiDAR is collected by MMS's. LIDAR, which stands for Light Detection and Ranging, is a is a surveying method that uses  light to measure distances to a targert.This process generates millions of 3D points which allows for a highly accurancte 3D model of a section of road to be genereated. This project proposes to use machine learnig techniques to extract road edges from a lidar data source.



##### Methods
I propose to try an inplement a Multi-layer Perceptron to process the data using scikit-learn or tensor flow.Other algorithms will be used to compare accuracy and processing times.



#### Materials
LIDAR data from a previous project I worked on will be used. This data consists of terrestrial LIDAR from a van and point clouds generated using photogrammetry techniques from airborne photos taken using a drone.



#### Expected results
The ability to generated a KML/GeoJSON file consisting of 3D data points consisting of the road edge. If time permits the detection of the road centre will also be attempted.



#### Risk assessment
Footpaths and hardshoulders on or beside the road may lead to issues in the correct determination of the road edge. A plan to correct this may be to find road markings in the road and to use this information to compute the road edge.



#### Acknowledgments
PMS Pavement Management Services Ltd, collected LIDAR data.
