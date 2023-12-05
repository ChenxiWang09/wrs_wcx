import numpy as np
class part_grasp_pos:
    def __init__(self,partname,coordinates):
        self.partname=partname
        self.coordinates=coordinates

    def grasp_pos_compute(self):
        self.grasp_pos=np.zeros((1,3), dtype=float)
        if(self.partname=='1.pgm'):  # first part grasp point is the middle of the 1st and 2nd point's middle point
            self.grasp_pos[0][0] = (self.coordinates[0][0] + self.coordinates[1][0]) / 2
            self.grasp_pos[0][1] = (self.coordinates[0][1] + self.coordinates[1][1]) / 2
            self.grasp_pos[0][2] = (self.coordinates[0][2] + self.coordinates[1][2]) / 2

        elif(self.partname=='2.pgm'): # second part grasp point is the middle of the 1st and 2nd point's middle point
            self.grasp_pos[0][0] = (self.coordinates[0][0] + self.coordinates[1][0]) / 2
            self.grasp_pos[0][1] = (self.coordinates[0][1] + self.coordinates[1][1]) / 2
            self.grasp_pos[0][2] = (self.coordinates[0][2] + self.coordinates[1][2]) / 2

        elif(self.partname=='3.pgm'): # second part grasp point is the middle of the 1st and 2nd point's middle point
            self.grasp_pos[0][0] = (self.coordinates[0][0] + self.coordinates[1][0]) / 2
            self.grasp_pos[0][1] = (self.coordinates[0][1] + self.coordinates[1][1]) / 2
            self.grasp_pos[0][2] = (self.coordinates[0][2] + self.coordinates[1][2]) / 2