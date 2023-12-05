import cv2
import numpy as np
import pylab as pl
import sys, os
    # 鼠标事件
def mouse1(event, x, y, flags, param):
    global flag, horizontal, vertical, flag_hor, flag_ver, dx, dy, sx, sy, dst, x1, y1, x2, y2, x3, y3, f1, f2
    global zoom, scroll_har, scroll_var, img_w, img_h, img, dst1, win_w, win_h, show_w, show_h
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        pts_2d_1.append([x, y])
        if flag == 0:
            if horizontal and 0 < x < win_w and win_h - scroll_w < y < win_h:
                flag_hor = 1  # 鼠标在水平滚动条上
            elif vertical and win_w - scroll_w < x < win_w and 0 < y < win_h:
                flag_ver = 1  # 鼠标在垂直滚动条上
            if flag_hor or flag_ver:
                flag = 1  # 进行滚动条垂直
                x1, y1, x2, y2, x3, y3 = x, y, dx, dy, sx, sy  # 使鼠标移动距离都是相对于初始滚动条点击位置，而不是相对于上一位置
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        if flag == 1:
            if flag_hor:
                w = (x - x1) / 2  # 移动宽度
                dx = x2 + w * f1  # 原图x
                if dx < 0:  # 位置矫正
                    dx = 0
                elif dx > img_w - show_w:
                    dx = img_w - show_w
                sx = x3 + w  # 滚动条x
                if sx < 0:  # 位置矫正
                    sx = 0
                elif sx > win_w - scroll_har:
                    sx = win_w - scroll_har
            if flag_ver:
                h = y - y1  # 移动高度
                dy = y2 + h * f2  # 原图y
                if dy < 0:  # 位置矫正
                    dy = 0
                elif dy > img_h - show_h:
                    dy = img_h - show_h
                sy = y3 + h  # 滚动条y
                if sy < 0:  # 位置矫正
                    sy = 0
                elif sy > win_h - scroll_var:
                    sy = win_h - scroll_var
            dx, dy = int(dx), int(dy)
            img1 = img[dy:dy + show_h, dx:dx + show_w]  # 截取显示图片
            print(dy, dy + show_h, dx, dx + show_w)
            dst = img1.copy()
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        flag, flag_hor, flag_ver = 0, 0, 0
        x1, y1, x2, y2, x3, y3 = 0, 0, 0, 0, 0, 0
    elif event == cv2.EVENT_MOUSEWHEEL:  # 滚轮
        if flags > 0:  # 滚轮上移
            zoom += wheel_step
            if zoom > 1 + wheel_step * 20:  # 缩放倍数调整
                zoom = 1 + wheel_step * 20
        else:  # 滚轮下移
            zoom -= wheel_step
            if zoom < wheel_step:  # 缩放倍数调整
                zoom = wheel_step
        zoom = round(zoom, 2)  # 取2位有效数字
        img_w, img_h = int(img_original_w * zoom), int(img_original_h * zoom)  # 缩放都是相对原图，而非迭代
        img_zoom = cv2.resize(img_original, (img_w, img_h), interpolation=cv2.INTER_AREA)
        horizontal, vertical = 0, 0
        if img_h <= win_h and img_w <= win_w:
            dst1 = img_zoom
            cv2.resizeWindow("img", img_w, img_h)
            scroll_har, scroll_var = 0, 0
            f1, f2 = 0, 0
        else:
            if img_w > win_w and img_h > win_h:
                horizontal, vertical = 1, 1
                scroll_har, scroll_var = win_w * show_w / img_w, win_h * show_h / img_h
                f1, f2 = (img_w - show_w) / (win_w - scroll_har), (img_h - show_h) / (win_h - scroll_var)
            elif img_w > win_w and img_h <= win_h:
                show_h = img_h
                win_h = show_h + scroll_w
                scroll_har, scroll_var = win_w * show_w / img_w, 0
                f1, f2 = (img_w - show_w) / (win_w - scroll_har), 0
            elif img_w <= win_w and img_h > win_h:
                show_w = img_w
                win_w = show_w + scroll_w
                scroll_har, scroll_var = 0, win_h * show_h / img_h
                f1, f2 = 0, (img_h - show_h) / (win_h - scroll_var)
            dx, dy = dx * zoom, dy * zoom  # 缩放后显示图片相对缩放图片的坐标
            sx, sy = dx / img_w * (win_w - scroll_har), dy / img_h * (win_h - scroll_var)
            img = img_zoom.copy()  # 令缩放图片为原图
            dx, dy = int(dx), int(dy)
            img1 = img[dy:dy + show_h, dx:dx + show_w]
            dst = img1.copy()

    if horizontal and vertical:
        sx, sy = int(sx), int(sy)
        # 对dst1画图而非dst，避免鼠标事件不断刷新使显示图片不断进行填充
        dst1 = cv2.copyMakeBorder(dst, 0, scroll_w, 0, scroll_w, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.rectangle(dst1, (sx, show_h), (int(sx + scroll_har), win_h), (181, 181, 181), -1)  # 画水平滚动条
        cv2.rectangle(dst1, (show_w, sy), (win_w, int(sy + scroll_var)), (181, 181, 181), -1)  # 画垂直滚动条
    elif horizontal == 0 and vertical:
        sx, sy = int(sx), int(sy)
        dst1 = cv2.copyMakeBorder(dst, 0, 0, 0, scroll_w, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.rectangle(dst1, (show_w, sy), (win_w, int(sy + scroll_var)), (181, 181, 181), -1)  # 画垂直滚动条
    elif horizontal and vertical == 0:
        sx, sy = int(sx), int(sy)
        dst1 = cv2.copyMakeBorder(dst, 0, scroll_w, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.rectangle(dst1, (sx, show_h), (int(sx + scroll_har), win_h), (181, 181, 181), -1)  # 画水平滚动条
    cv2.imshow("img", dst1)
    cv2.waitKey(27)

def mouse2(event, x, y, flags, param):
    global flag, horizontal, vertical, flag_hor, flag_ver, dx, dy, sx, sy, dst, x1, y1, x2, y2, x3, y3, f1, f2
    global zoom, scroll_har, scroll_var, img_w, img_h, img, dst1, win_w, win_h, show_w, show_h
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        pts_2d_2.append([x, y])
        if flag == 0:
            if horizontal and 0 < x < win_w and win_h - scroll_w < y < win_h:
                flag_hor = 1  # 鼠标在水平滚动条上
            elif vertical and win_w - scroll_w < x < win_w and 0 < y < win_h:
                flag_ver = 1  # 鼠标在垂直滚动条上
            if flag_hor or flag_ver:
                flag = 1  # 进行滚动条垂直
                x1, y1, x2, y2, x3, y3 = x, y, dx, dy, sx, sy  # 使鼠标移动距离都是相对于初始滚动条点击位置，而不是相对于上一位置
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        if flag == 1:
            if flag_hor:
                w = (x - x1) / 2  # 移动宽度
                dx = x2 + w * f1  # 原图x
                if dx < 0:  # 位置矫正
                    dx = 0
                elif dx > img_w - show_w:
                    dx = img_w - show_w
                sx = x3 + w  # 滚动条x
                if sx < 0:  # 位置矫正
                    sx = 0
                elif sx > win_w - scroll_har:
                    sx = win_w - scroll_har
            if flag_ver:
                h = y - y1  # 移动高度
                dy = y2 + h * f2  # 原图y
                if dy < 0:  # 位置矫正
                    dy = 0
                elif dy > img_h - show_h:
                    dy = img_h - show_h
                sy = y3 + h  # 滚动条y
                if sy < 0:  # 位置矫正
                    sy = 0
                elif sy > win_h - scroll_var:
                    sy = win_h - scroll_var
            dx, dy = int(dx), int(dy)
            img1 = img[dy:dy + show_h, dx:dx + show_w]  # 截取显示图片
            print(dy, dy + show_h, dx, dx + show_w)
            dst = img1.copy()
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        flag, flag_hor, flag_ver = 0, 0, 0
        x1, y1, x2, y2, x3, y3 = 0, 0, 0, 0, 0, 0
    elif event == cv2.EVENT_MOUSEWHEEL:  # 滚轮
        if flags > 0:  # 滚轮上移
            zoom += wheel_step
            if zoom > 1 + wheel_step * 20:  # 缩放倍数调整
                zoom = 1 + wheel_step * 20
        else:  # 滚轮下移
            zoom -= wheel_step
            if zoom < wheel_step:  # 缩放倍数调整
                zoom = wheel_step
        zoom = round(zoom, 2)  # 取2位有效数字
        img_w, img_h = int(img_original_w * zoom), int(img_original_h * zoom)  # 缩放都是相对原图，而非迭代
        img_zoom = cv2.resize(img_original, (img_w, img_h), interpolation=cv2.INTER_AREA)
        horizontal, vertical = 0, 0
        if img_h <= win_h and img_w <= win_w:
            dst1 = img_zoom
            cv2.resizeWindow("img", img_w, img_h)
            scroll_har, scroll_var = 0, 0
            f1, f2 = 0, 0
        else:
            if img_w > win_w and img_h > win_h:
                horizontal, vertical = 1, 1
                scroll_har, scroll_var = win_w * show_w / img_w, win_h * show_h / img_h
                f1, f2 = (img_w - show_w) / (win_w - scroll_har), (img_h - show_h) / (win_h - scroll_var)
            elif img_w > win_w and img_h <= win_h:
                show_h = img_h
                win_h = show_h + scroll_w
                scroll_har, scroll_var = win_w * show_w / img_w, 0
                f1, f2 = (img_w - show_w) / (win_w - scroll_har), 0
            elif img_w <= win_w and img_h > win_h:
                show_w = img_w
                win_w = show_w + scroll_w
                scroll_har, scroll_var = 0, win_h * show_h / img_h
                f1, f2 = 0, (img_h - show_h) / (win_h - scroll_var)
            dx, dy = dx * zoom, dy * zoom  # 缩放后显示图片相对缩放图片的坐标
            sx, sy = dx / img_w * (win_w - scroll_har), dy / img_h * (win_h - scroll_var)
            img = img_zoom.copy()  # 令缩放图片为原图
            dx, dy = int(dx), int(dy)
            img1 = img[dy:dy + show_h, dx:dx + show_w]
            dst = img1.copy()

    if horizontal and vertical:
        sx, sy = int(sx), int(sy)
        # 对dst1画图而非dst，避免鼠标事件不断刷新使显示图片不断进行填充
        dst1 = cv2.copyMakeBorder(dst, 0, scroll_w, 0, scroll_w, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.rectangle(dst1, (sx, show_h), (int(sx + scroll_har), win_h), (181, 181, 181), -1)  # 画水平滚动条
        cv2.rectangle(dst1, (show_w, sy), (win_w, int(sy + scroll_var)), (181, 181, 181), -1)  # 画垂直滚动条
    elif horizontal == 0 and vertical:
        sx, sy = int(sx), int(sy)
        dst1 = cv2.copyMakeBorder(dst, 0, 0, 0, scroll_w, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.rectangle(dst1, (show_w, sy), (win_w, int(sy + scroll_var)), (181, 181, 181), -1)  # 画垂直滚动条
    elif horizontal and vertical == 0:
        sx, sy = int(sx), int(sy)
        dst1 = cv2.copyMakeBorder(dst, 0, scroll_w, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.rectangle(dst1, (sx, show_h), (int(sx + scroll_har), win_h), (181, 181, 181), -1)  # 画水平滚动条
    cv2.imshow("img", dst1)
    cv2.waitKey(27)

pts_2d_1=[]
pts_2d_2=[]

img_original = cv2.imread("3.jpg")  # 此处需换成大于img_w * img_h的图片
img_original_h, img_original_w = img_original.shape[0:2]  # 原图宽高
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.moveWindow("img", 300, 100)
img = img_original.copy()
img_h, img_w = img.shape[0:2]  # 原图宽高
show_h, show_w = 600, 800  # 显示图片宽高
horizontal, vertical = 0, 0  # 原图是否超出显示图片
dx, dy = 0, 0  # 显示图片相对于原图的坐标
scroll_w = 16  # 滚动条宽度
sx, sy = 0, 0  # 滚动块相对于滚动条的坐标
flag, flag_hor, flag_ver = 0, 0, 0  # 鼠标操作类型，鼠标是否在水平滚动条上，鼠标是否在垂直滚动条上
x1, y1, x2, y2, x3, y3 = 0, 0, 0, 0, 0, 0  # 中间变量
win_w, win_h = show_w + scroll_w, show_h + scroll_w  # 窗口宽高
scroll_har, scroll_var = win_w * show_w / img_w, win_h * show_h / img_h  # 滚动条水平垂直长度
wheel_step, zoom = 0.05, 1  # 缩放系数， 缩放值
zoom_w, zoom_h = img_w, img_h  # 缩放图宽高
f1, f2 = (img_w - show_w) / (win_w - scroll_har), (img_h - show_h) / (win_h - scroll_var)  # 原图可移动部分占滚动条可移动部分的比例

if img_h <= show_h and img_w <= show_w:
    cv2.imshow("img", img)
else:
    if img_w > show_w:
        horizontal = 1
    if img_h > show_h:
        vertical = 1
    i = img[dy:dy + show_h, dx:dx + show_w]
    dst = i.copy()
cv2.resizeWindow("img", win_w, win_h)
cv2.setMouseCallback('img', mouse1)

cv2.waitKey()
cv2.destroyAllWindows()

img_original = cv2.imread("4.jpg")  # 此处需换成大于img_w * img_h的图片
img_original_h, img_original_w = img_original.shape[0:2]  # 原图宽高
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.moveWindow("img", 300, 100)
img = img_original.copy()
img_h, img_w = img.shape[0:2]  # 原图宽高
show_h, show_w = 600, 800  # 显示图片宽高
horizontal, vertical = 0, 0  # 原图是否超出显示图片
dx, dy = 0, 0  # 显示图片相对于原图的坐标
scroll_w = 16  # 滚动条宽度
sx, sy = 0, 0  # 滚动块相对于滚动条的坐标
flag, flag_hor, flag_ver = 0, 0, 0  # 鼠标操作类型，鼠标是否在水平滚动条上，鼠标是否在垂直滚动条上
x1, y1, x2, y2, x3, y3 = 0, 0, 0, 0, 0, 0  # 中间变量
win_w, win_h = show_w + scroll_w, show_h + scroll_w  # 窗口宽高
scroll_har, scroll_var = win_w * show_w / img_w, win_h * show_h / img_h  # 滚动条水平垂直长度
wheel_step, zoom = 0.05, 1  # 缩放系数， 缩放值
zoom_w, zoom_h = img_w, img_h  # 缩放图宽高
f1, f2 = (img_w - show_w) / (win_w - scroll_har), (img_h - show_h) / (win_h - scroll_var)  # 原图可移动部分占滚动条可移动部分的比例

if img_h <= show_h and img_w <= show_w:
    cv2.imshow("img", img)
else:
    if img_w > show_w:
        horizontal = 1
    if img_h > show_h:
        vertical = 1
    i = img[dy:dy + show_h, dx:dx + show_w]
    dst = i.copy()
cv2.resizeWindow("img", win_w, win_h)
cv2.setMouseCallback('img', mouse2)

cv2.waitKey()
cv2.destroyAllWindows()
# Read source image.
im_src = cv2.imread('3.jpg')
# Four corners of the book in source image
pts_src = np.array([pts_2d_1[0], pts_2d_1[1], pts_2d_1[2], pts_2d_1[3]])

# Read destination image.
im_dst = cv2.imread('4.jpg')
# Four corners of the book in destination image.
pts_dst = np.array([pts_2d_2[0], pts_2d_2[1], pts_2d_2[2], pts_2d_2[3]])

# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)

# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

pl.figure(), pl.imshow(im_src[:, :, ::-1]), pl.title('src'),
pl.figure(), pl.imshow(im_dst[:, :, ::-1]), pl.title('dst')
pl.figure(), pl.imshow(im_out[:, :, ::-1]), pl.title('out'), pl.show()  # show dst