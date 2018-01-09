import json
import subprocess
import os.path
import os
from pyproj import Proj, transform


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
           print(newPoint)
           if count % 10 == 0:
               inProj = Proj(init=self.src_epsg)
               outProj = Proj(init=self.dst_epsg)
               x1,y1 = newPoint[0], newPoint[1]
               x2,y2 = transform(inProj,outProj,x1,y1)
               polygon += str(x2) + " " + str(y2) + ","

           count += 1

       inProj = Proj(init=self.src_epsg)
       outProj = Proj(init=self.dst_epsg)

       x1,y1 = self.points[0][0][0], self.points[0][0][1]
       x2,y2 = transform(inProj,outProj,x1,y1)

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


class Data_Preprocessor(object):


    def __init__(self, src_epsg, dst_epsg, src_path):
        self.points = points
        self.src_epsg = src_epsg
        self.dst_epsg = dst_epsg
        self.src_path = src_path

    def las2txt(self, src_path, dst_path=None):

        if dst_path is None:
            dst_path = src_path[:-3] + "txt"


        cmd = ("las2txt -i" +
                src_path +
                " --parse xyzainrM  " +
                "--delimiter \",\" " +
                "--labels " +
                "--header " +
                "-o " +
                dst_path)

        if not os.path.exists(output_path):
            subprocess.call(cmd.split(" "))
        else:
            print("File Already Excists")


    def split_data(self):





if __name__ == "__main__":
    with open('assets/road_edge.geojson') as data_file:
        data = json.load(data_file)

    road_edge_coords = data["features"][0]["geometry"]["coordinates"]

    road_points = "road.las"
    bank_points = "bank.las"

    my_segmenter = Segmenter(road_edge_coords,
                        "epsg:3857",
                        "epsg:2157",
                        "Project/scratch/segmented_road.las")

    my_segmenter.segment_in(road_points)
    my_segmenter.segment_out(bank_points)
