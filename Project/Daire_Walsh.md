# Daire Walsh

#### Project Name
Auto Road Edge Extraction

#### Author
Daire Walsh


#### Introduction
A Mobile Mapping System(MMS) is usually mounted on a van or car. Data is recorded by the MMS as it moves along a road. An MMS will have one or more laser scanners, GPS system, orientation system and possibly one or more cameras. LiDAR is collected by MMS's. LIDAR, which stands for Light Detection and Ranging, is a surveying method that uses light to measure distances to a target. This process generates millions of 3D points which allows for a highly accurate 3D model of a section of road to be generated. This project proposes to use machine learning techniques to extract road edges from a lidar data source.



##### Methods
I propose to try and implement a neural network, support vector machine and random forests to process the data using scikit-learn or tensor flow. The three algorithms will be used to compare accuracy and processing times. Preprocessing will be applied to the LiDAR datasets. This will consist of noise elimination as well as cropping to make data sizes more manageable. There is a possibility of OpenCV being used with the Aerial data as this contains RGB data. Accuracy will be assessed by generating manual road edges from the datasets and comparing them to the machine learning produced results. LAStools will be used for processing the LIDAR datasets.


#### Materials
LIDAR data from a previous project I worked on will be used. This data consists of terrestrial LIDAR from a van and point clouds generated using photogrammetry techniques from airborne photos taken using a drone.



#### Expected results
The ability to generate a KML/GeoJSON file consisting of 3D data points detailing the road edge. If time permits the detection of the road centre will also be attempted.



#### Risk assessment
Footpaths and hard shoulders on or beside the road may lead to issues in the correct determination of the road edge. A plan to correct this may be to detect road markings on the road and to use this information to compute the road edge.



#### Acknowledgments
PMS Pavement Management Services Ltd, collected LIDAR data. Sean Mannion and Fearghus Foyle of Yamsu Technologies for the aerial imagery. I would also like to thank Tim McCarthy, Conor Cahalane and Aidan Magee with help in processing the aerial imagery.  



#### Statement
I will keep all my source code in Gitlab/Github and commit and push often and arrange for you to have access to the repository.
