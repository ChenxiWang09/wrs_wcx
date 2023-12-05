import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import wcx.utils1.FDCM_matlab.fdcm as fdcm_matlab
import numpy as np
import os
# 继承 FileSystemEventHandler 类，用于定义文件系统事件处理器

file = ''
class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        # 当文件夹中创建文件时触发此方法
        global file
        if not event.is_directory:  # 确保事件不是文件夹的创建事件
            # 获取新创建的文件名
            filename = os.path.basename(event.src_path)
            # print(filename)
            if filename.startswith("screenshot_"):
                if(file != event.src_path):
                    file = event.src_path
                    name = filename[11]
                    name = int(name)

                    cost = 0
                    filename = event.src_path
                    print('read file:', filename)
                    template_path = '/home/hlabbunri/Desktop/chenxi wang/wrs-wcx/wcx/Assemble_task/data/image/p4_a.pgm'
                    fdcm = fdcm_matlab.FastDirectionalChamferMatching()
                    cost = fdcm.no_pos_matching(filename, template_path)
                    print('number:', name, ' cost:', cost)
                    save_path = '/home/hlabbunri/Desktop/chenxi wang/wrs-wcx/wcx/Assemble_task/data/watch_npy/'+str(name)+'.npy'
                    # print('Successfully save ', name)
                    name += 1
                    # print('save file:',save_path)
                    np.save(save_path, cost)
                    if(name == 3):
                        name = 0


# 创建一个观察者对象并设置观察的文件夹路径
folder_path = '/home/hlabbunri/Desktop/chenxi wang/wrs-wcx/wcx/Assemble_task/data/watch_image'
event_handler = MyHandler()
observer = Observer()
observer.schedule(event_handler, folder_path, recursive=False)

# 启动观察者
observer.start()

# 运行一个无限循环，持续监视文件夹
try:
    while True:
        time.sleep(0.25)
except KeyboardInterrupt:
    observer.stop()

observer.join()


