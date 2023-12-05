import numpy

def planefitting_mutiply(pointcloud,chosenpos):
    n=len(pointcloud)
    m=len(pointcloud[0])
    plane=numpy.zeros(n,m)
    # initial matrix
    searchingsize=2
    pointsnum=1
    while(judge==1):
        if(pointsnum==1):