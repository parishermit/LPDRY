from flask import Flask, render_template, request
import json

from pathlib import Path
from video import video_identity
from flask import Flask, url_for
from flask import send_from_directory
from werkzeug.utils import secure_filename
import cv2 as cv
from split import Split_Char
import cnn_predict
import os
from identity import identity_function

app = Flask(__name__)
# 进行相关的配置文件
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = os.getcwd()+'\\static\\upload\\'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'mp4'])
# MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# 定义验证后缀的脚本文件
def allowed_file(filename):
    print(filename.rsplit('.', 1)[1])
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def car_recognize():  # put application's code here
    return render_template("car_recognize.html")

@app.route('/image/search', methods=['GET', 'POST'])
def image_search():  # put application's code here
    #初始化车牌
    car_id = '000'
    filename=''
    msg=''
    if request.method == 'POST':
        file = request.files['uploadfile']
        print(file.filename)
        if file and allowed_file(file.filename):
            # 将初始的给保存一下
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # 然后对图片进行大小的处理
            imagePath = app.config['UPLOAD_FOLDER'] + filename
            imagePath = Path(imagePath)
            car_id=identity_function(str(imagePath))
        # file_url = url_for('image_search', filename=filename)
        return render_template("image_search.html", car_id1=car_id, imagename=filename)
    return render_template("image_search.html", car_id1=car_id, imagename="porsche-911-922-carrera-cabriolet-5120x2880-2020-cars-5k-22016.jpeg")
    #         # 这一块是调用一些接口，然后进行相关的操作,输入的参数是imagePath
    #         # 将方法的返回值赋给car_id
    #         # car_id = 666
    #         # plates = plate_Location.plateLocate(str(imagePath))
    #         # print(1111)
    #         # print(len(plates))
    #         # for i in range(0, len(plates)):
    #         #     cv.imwrite("/debugger/plate"+str(i)+".jpg", plates[i])
    #         #     print(plates[i])
    #
    #         is_checked = False
    #         plates = plate_Location.plateLocate(str(imagePath))
    #         for plate in plates:
    #             split_char = Split_Char(plate)
    #             char_images, color = split_char.split_and_save_imagesAndColor()
    #             if (color == 'green' and len(char_images) < 8) or (color != 'green' and len(char_images) < 7):
    #                 # 如果达不到条件,继续
    #                 continue
    #             else:
    #                 # 达到条件了就正常输出
    #                 is_checked = True
    #                 break
    #
    #         if is_checked == True:
    #             string0 = cnn_predict.getstring(color)
    #             print(string0)
    #             car_id = string0
    #             return render_template('image_search.html', imagename=filename, car_id=car_id)
    #
    #         file_url = url_for('image_search', filename=filename)
    #         return render_template('image_search.html', imagename=filename, car_id=car_id)
    #     else:
    #         msg = '错误，上传文件类型错误'
    #         return render_template('image_search.html.html', msg=msg)
    #
    # return render_template('image_search.html',
    #                        imagename="porsche-911-922-carrera-cabriolet-5120x2880-2020-cars-5k-22016.jpeg",
    #                        car_id=car_id)

@app.route('/group/introduction')
def group_introduction():  # put application's code here
    # src1="/static/video/first~1.mp4"
    return render_template("group_introduction.html")

@app.route('/video/search', methods=['GET', 'POST'])
def vedio_search():  # put application's code here
    #初始化车牌
    car_id_list = []
    #初始化文件名
    filename=''
    if request.method == 'POST':
        file = request.files['uploadfile']
        print(file.filename)
        if file and allowed_file(file.filename):
            # 将初始的给保存一下
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            videoPath = app.config['UPLOAD_FOLDER'] + filename
            videoPath = Path(videoPath)

            car_id_list=video_identity(videoPath)
            print("car_id_list=", len(car_id_list))
            print("car_id_list.name=", car_id_list[0][1])
        return render_template("video_search.html", videoname=filename, car_id_list=car_id_list, len=len(car_id_list))
    return render_template("video_search.html", videoname="first~1.mp4")


@app.route('/mysql')
def my_sql():  # put application's code here
    # dataInfo = [
    #         {"video_name":"视频1", "car_num":"A12345", "video_time": "21:21"},
    #     {"video_name": "视频2", "car_num": "A12346", "video_time": "21:22"},
    #     {"video_name": "视频3", "car_num": "A12347", "video_time": "21:23"},
    #     {"video_name":"视频4", "car_num":"A12348", "video_time": "21:24"}
    #     ]
    return render_template("mysql.html")

@app.route('/take/poto')
def po():  # put application's code here
    return render_template("take_poto.html")

if __name__ == '__main__':
    app.run()
