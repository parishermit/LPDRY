"""
    对视频的相关操作
"""
import os
from pathlib import Path

import cv2 as cv
import plate_Location
from split import Split_Char
import cnn_predict


# 从视频中截取帧
# 参数：传入一个视频路径
# 返回值，截取到的帧，图片集

def get_video_seperated(video_path):

    video_ = cv.VideoCapture(str(video_path))
    video_images = []
    times = []
    # 判断是否正常打开
    if video_.isOpened():
        # 读帧
        success, frame = video_.read()
    else:
        success = False
    i = 0
    # 设置固定帧率
    timeF = 35
    # 循环读取视频帧
    while success:
        if i % timeF == 0:
            video_images.append(frame)
            milliseconds = video_.get(cv.CAP_PROP_POS_MSEC)

            seconds = milliseconds // 1000
            milliseconds = milliseconds % 1000
            minutes = 0
            hours = 0
            if seconds >= 60:
                minutes = seconds // 60
                seconds = seconds % 60

            if minutes >= 60:
                hours = minutes // 60
                minutes = minutes % 60
            # print(int(hours), ":", int(minutes), ":", int(seconds), ":", int(milliseconds))
            time = "{}h,{}min,{}second,{}milliseconds".format(int(hours), int(minutes), int(seconds), int(milliseconds))
            times.append(time)
        success, frame = video_.read()
        i = i + 1
    video_.release()

    return video_images, times


def identity_function(image):
    plates = plate_Location.plate_locate1(image)

    car_id = '识别出错'
    is_checked = False

    for plate in plates:
        split_char = Split_Char(plate)
        char_images, color = split_char.split_and_save_imagesAndColor()
        if (color == 'green' and len(char_images) < 8) or (color != 'green' and len(char_images) < 7):
            # 如果达不到条件,继续
            continue
        else:
            # 达到条件了就正常输出
            is_checked = True
            break

    if is_checked == True:
        string0 = cnn_predict.getstring(color)
        # print(string0)
        car_id = string0
    else:
        print(car_id)

    return car_id


def video_identity(path):
# if __name__ == '__main__':
    # video_path = "vedio/plate_vedio.mp4"
    video_path = Path(path)
    video_name = os.path.basename(video_path)

    images, times = get_video_seperated(video_path)

    #plate_identify_list = [[video_name], [plate_name],[time]]
    plate_identify_list = []

    for i in range(len(images)):
        img = images[i]
        time = times[i]
        plate = identity_function(img)

        if plate != '识别出错':
            #str = {"video_name":video_name, "plate_name:":plate, "time":time}
            str = [[video_name], [plate], [time]]
            plate_identify_list.append(str)
            #print(str)
    print(plate_identify_list)
    return plate_identify_list

def find_plate(video_path,plate_name):
    plate_list=video_identity(video_path)
    result=[]
    is_in_video_=False
    for info in plate_list:
        #print(info[1])
        if info[1] == plate_name:
           #print(info[1])
           result.append(info)
           is_in_video_=True

    if is_in_video_:
        return result
    return "未找到车牌ww"


# path = "video_/plate_video_.mp4"
# print(find_plate(path, ["晋A9900H"]))
