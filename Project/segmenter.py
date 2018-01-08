import json
import subprocess
import os.path
import os
from pyproj import Proj, transform


class Segmenter:
    def __init__(points, src_epsg, dst_epsg, scr_path):
        self.points = points
        self.src_epsg = src_epsg
        self.dst_epsg = dst_epsg
        slef.scr_path = scr_path


   def create_polygon(self):
       polygon = "POLYGON(("
       count = 0
       for newPoint in slef.points[0]:
           if count % 10 == 0:
               inProj = Proj(init=self.src_epsg)
               outProj = Proj(init=self.dst_epsg)
               x1,y1 = newPoint[0], newPoint[1]
               x2,y2 = transform(inProj,outProj,x1,y1)
               polygon += str(x2) + " " + str(y2) + ","

               count += 1

     inProj = Proj(init=self.src_epsg)
     outProj = Proj(init=self.dst_epsg)
     x1,y1 = points[0][0][0], points[0][0][1]
     x2,y2 = transform(inProj,outProj,x1,y1)
     polygon += str(x2) + " " + str(y2) + ","

     polygon = polygon[:-1]

     polygon += "))"

   def displayEmployee(self):
      print "Name : ", self.name,  ", Salary: ", self.salary

def create_pdal_command(points, src_epsg, dst_epsg, src_path):

    polygon = "POLYGON(("
    count = 0
    for newPoint in points[0]:
        if count % 10 == 0:
            inProj = Proj(init='epsg:3857')
            outProj = Proj(init='epsg:2157')
            x1,y1 = newPoint[0], newPoint[1]
            x2,y2 = transform(inProj,outProj,x1,y1)
            polygon += str(x2) + " " + str(y2) + ","

        count += 1

    inProj = Proj(init='epsg:3857')
    outProj = Proj(init='epsg:2157')
    x1,y1 = points[0][0][0], points[0][0][1]
    x2,y2 = transform(inProj,outProj,x1,y1)
    polygon += str(x2) + " " + str(y2) + ","

    polygon = polygon[:-1]

    polygon += "))"

    pdal_Command = ("docker run -v /home/daire/Code/CS401_Projects/Project:/data " +
                    "pdal/pdal:1.5 pdal pipeline " +
                    "data/assets/pipelines/pipeline.json " +
                    "--readers.las.filename=data/scratch/segmented_road.las " +
                    "--filters.crop.polygon=\"" + polygon + "\" " +
                    "--writers.las.filename=data/scratch/roadfdfdd.las")

    return pdal_Command


if __name__ == "__main__":
    with open('assets/road_edge.geojson') as data_file:
        data = json.load(data_file)

    road_points = data["features"][0]["geometry"]["coordinates"]

    seg_cmd = create_pdal_command(road_points, "epsg:3857", "epsg:2157")

    #subprocess.call(seg_cmd.split(" "))
    os.system(seg_cmd)
