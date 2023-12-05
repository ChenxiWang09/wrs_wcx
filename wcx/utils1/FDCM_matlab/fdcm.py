import matlab.engine as engine

class FastDirectionalChamferMatching():

    def __init__(self):
        self.eng = engine.start_matlab()

    def matching(self, queryaddress, tempaddress, size_min_x, size_min_y, size_max_x, size_max_y,times):
        part_num, cost = self.eng.chamfermatching_nopic_forpython(queryaddress, tempaddress, size_min_x, size_min_y, size_max_x, size_max_y,times , nargout=2)
        return cost

    def no_pos_matching(self, queryaddress: object, tempaddress: object) -> object:
        cost = self.eng.chamfermatching_no_pos_no_num(queryaddress, tempaddress)
        return cost


if __name__ == '__main__':

    info = []
    queryaddress = '/home/hlabbunri/Desktop/chenxi wang/wrs-wcx/wcx/grasptask/data/part_edge.pgm'
    tempaddress = '/home/hlabbunri/Desktop/chenxi wang/wrs-wcx/wcx/grasptask/data/template_edge.pgm'
    size_min_x = 209
    size_min_y = 25
    size_max_x = 536
    size_max_y = 303
    times = 5

    matching(queryaddress, tempaddress, size_min_x, size_min_y, size_max_x, size_max_y,times)
