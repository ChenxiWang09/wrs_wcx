import cv2
def cannyedge(trigger=False):
    template_path = '/wcx/Assemble_task/data/image/p4_a.png'
    rgb = cv2.imread(template_path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    n = len(gray)
    m = len(gray[0])
    for i in range(n):
        for j in range(m):
            if gray[i][j]>120:
                gray[i][j] = 0
            else:
                gray[i][j] = 255
    while trigger:
        cv2.imshow('edge',gray)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite('/wcx/Assemble_task/data/image/p4_a.pgm', gray)
            print('success!')
if __name__ == '__main__':
        cannyedge(trigger=True)