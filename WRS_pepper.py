#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

import zbar
import MySQLdb
import sys
import cv2
import Image
import socket
from math import radians, degrees, sqrt
import time
import argparse
import numpy as np
from threading import Thread
import thread
# 只有一个register函数，用于程序退出时的回调函数
import atexit
#ROS
import rospy
import actionlib
from actionlib_msgs.msg import GoalID
from visualization_msgs.msg import Marker
import sensor_msgs.point_cloud2 as pc2
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, Twist, Pose, PoseStamped, Vector3
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo
from nav_msgs.msg import Odometry, OccupancyGrid
from tf import TransformListener, Transformer, transformations
from std_srvs.srv import Empty
from std_msgs.msg import Int32, String, ColorRGBA
# pexpect
import pexpect
import dynamic_reconfigure.client
# naoqi
import qi
#人脸识别
from face_train_use_keras import Model


class pepper_test():
    def __init__(self, params):
        # 初始化ROS节点
        rospy.init_node("pepper_test")
        # 程序退出时的回调函数
        atexit.register(self.__del__)

        self.ip = params['ip']
        self.port = params['port']
        self.session = qi.Session()

        try:
            self.session.connect("tcp://" + self.ip + ":" + str(self.port))
        except RuntimeError:
            print ("[Kamerider W] : connection Error!!")
            sys.exit(1)

        # 订阅需要的服务
        self.VideoDev = self.session.service("ALVideoDevice")
        self.FaceCha = self.session.service("ALFaceCharacteristics")
        self.Memory = self.session.service("ALMemory")
        self.Dialog = self.session.service("ALDialog")
        self.AnimatedSpe = self.session.service("ALAnimatedSpeech")
        self.AudioDev = self.session.service("ALAudioDevice")
        self.BasicAwa = self.session.service("ALBasicAwareness")
        self.AutonomousLife = self.session.service("ALAutonomousLife")
        self.TabletSer = self.session.service("ALTabletService")
        self.TextToSpe = self.session.service("ALTextToSpeech")
        self.Motion = self.session.service("ALMotion")
        self.RobotPos = self.session.service("ALRobotPosture")
        self.Tracker = self.session.service("ALTracker")

        # 关闭 basic_awareness
        self.BasicAwa.setEnabled(False)

        # 设置tracker模式为头追踪
        self.Tracker.setMode("Head")

        # 初始化平板
        self.TabletSer.cleanWebview()
        # 初始化位置点
        self.PepperPosition = PoseStamped()
        # 推荐的商品的位置
        self.Point_A = MoveBaseGoal()
        self.Point_A.target_pose.header.frame_id = '/map'
        self.Point_A.target_pose.header.stamp = self.PepperPosition.header.stamp
        self.Point_A.target_pose.header.seq = self.PepperPosition.header.seq
        self.Point_A.target_pose.pose.position.x = 1.57589215035
        self.Point_A.target_pose.pose.position.y =-4.95822132707
        self.Point_A.target_pose.pose.position.z = .0
        self.Point_A.target_pose.pose.orientation.x = .0
        self.Point_A.target_pose.pose.orientation.y = .0
        self.Point_A.target_pose.pose.orientation.z =-0.996909852153
        self.Point_A.target_pose.pose.orientation.w =0.0785541003461
        # 起始位置
        self.Point_B = MoveBaseGoal()
        self.Point_B.target_pose.header.frame_id = '/map'
        self.Point_B.target_pose.header.stamp = self.PepperPosition.header.stamp
        self.Point_B.target_pose.header.seq = self.PepperPosition.header.seq
        self.Point_B.target_pose.pose.position.x =3.48176515207
        self.Point_B.target_pose.pose.position.y =-4.7906732889
        self.Point_B.target_pose.pose.position.z = .0
        self.Point_B.target_pose.pose.orientation.x = .0
        self.Point_B.target_pose.pose.orientation.y = .0
        self.Point_B.target_pose.pose.orientation.z =-0.99958660438
        self.Point_B.target_pose.pose.orientation.w =0.0287510059817

        # 设置输出音量
        # 相机
        video_subs_list = ['rgb_t_0', 'rgb_b_0', 'dep_0']

        for sub_name in video_subs_list:
            print sub_name, self.VideoDev.unsubscribe(sub_name)
        self.rgb_top = self.VideoDev.subscribeCamera('rgb_t', 0, 2, 11, 40)
        self.depth = self.VideoDev.subscribeCamera('dep', 2, 1, 17, 20)

        # 语言
        self.Dialog.setLanguage("English")

        # 载入topic
        self.Topic_path = '/home/nao/top/WRS_pepper_enu.top'
        self.Topic_path = self.Topic_path.decode('utf-8')

        # topic回调函数
        self.go_to_pointA_sub = self.Memory.subscriber("go_to_pointA")
        self.go_to_pointA_sub.signal.connect(self.callback_pointA)
        self.Bar_code_sub = self.Memory.subscriber("Bar_code")
        self.Bar_code_sub.signal.connect(self.callback_Bar_code)
        self.change_img_language_sub = self.Memory.subscriber("Info_Chinese")
        self.change_img_language_sub.signal.connect(self.callback_img_lan_CHN)
        self.record_item_sub = self.Memory.subscriber("record")
        self.record_item_sub.signal.connect(self.callback_record_item)
        self.pay_bill_sub = self.Memory.subscriber("pay_bill")
        self.pay_bill_sub.signal.connect(self.callback_pay_bill)
        self.finish_sub = self.Memory.subscriber("finish")
        self.finish_sub.signal.connect(self.callback_finish)
        self.recommend_sub = self.Memory.subscriber("recommend_start")
        self.recommend_sub.signal.connect(self.callback_recommend)
        self.hide_image_sub = self.Memory.subscriber("hide_image")
        self.hide_image_sub.signal.connect(self.callback_hide_image)
        self.half_price_sub = self.Memory.subscriber("half_price")
        self.half_price_sub.signal.connect(self.callback_half_price)
        self.face_start_sub = self.Memory.subscriber("start_face")
        self.face_start_sub.signal.connect(self.start_face_recog)
        self.japan_sub = self.Memory.subscriber("japan")
        self.japan_sub.signal.connect(self.callback_japan)
        # 自主说话模式
        self.configure = {"bodyLanguageMode": "contextual"}

        # ROS节点
        self.nav_as = actionlib.SimpleActionClient("/move_base", MoveBaseAction)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.goal_cancel_pub = rospy.Publisher('/move_base/cancel', GoalID, queue_size=1)
        self.nav_as.wait_for_server()

        self.transform = TransformListener()
        self.transformer = Transformer(True, rospy.Duration(10))
        self.goal_pub = rospy.Publisher('/move_base/current_goal', PoseStamped, queue_size=1)
        # 清除costmap
        self.map_clear_srv = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        self.map_clear_srv()

        # 连接数据库
        self.db = MySQLdb.connect("localhost", "root", "'", "WRS_PEPPER")

        # 商品条形码对应的价格
        self.price = {"jam": 8,
                      "tea": 3,
                      "bread": 5,
                      "juice": 6,
                      "potato chips": 7,
                      "beef": 10,
                      "instant noodles": 5}
        self.price_sum = 0

        # 变量值
        self.filename = "/home/jiashi/Desktop/1538483080.png"

        # 当前记录的单号
        self.order_id = 6
        self.month = 10

        # 记录是否停止对话
        self.if_stop_dia = False
        self.if_start_dia = False
        self.Pause_dia = False

        # 初始化头的位置
        self.Motion.setStiffnesses("Head", 1.0)
        self.Motion.setAngles("Head", [0., -0.25], .05)

        # 要执行的javascript
        self.java_script = ""

        # 设置说话速度
        self.TextToSpe.setParameter("speed", 80.0)

        # pepper头的角度
        self.angle = -.33

        # 第一个商品
        self.first_item = 1
        self.state = True
        self.start_head = True

        # 调用成员函数
        self.set_volume(1)
        self.cal_user_similarity()
        self.close_auto_life()
        self.Topic_name = self.Dialog.loadTopic(self.Topic_path.encode('utf-8'))
        self.Dialog.activateTopic(self.Topic_name)
        self.show_image("welcome.jpg")
        self.start_head_fix()
        self.start_dialog()
        self.activate_keyboard_control()

    def __del__(self):
        print "\033[0;32m stop audio recording... \033[0m"
        try:
            self.Dialog.unsubscribe("WRS_pepper")
        except RuntimeError:
            print "[Kamerider W ] : the event \'WRS_pepper\' hasn't been subscribed"
        list_act = self.Dialog.getActivatedTopics()
        print "[Kamerider] : getActivatedTopics", list_act
        list_load = self.Dialog.getLoadedTopics('english')
        print "[Kamerider] : getLoadedTopics", list_load
        for i in list_act:
            self.Dialog.deactivateTopic(str(i))
        for i in list_load:
            self.Dialog.unloadTopic(str(i))
        self.Dialog.stopDialog()
        self.TabletSer.hideWebview()
        self.TabletSer.hideImage()
        self.cancel_plan()
        self.db.close()

    def set_volume(self, volume):
        self.TextToSpe.setVolume(volume)

    def close_auto_life(self):
        if self.AutonomousLife.getState() != "disabled":
            self.AutonomousLife.setState("disabled")
        self.RobotPos.goToPosture("Stand", .5)

    def head_fix_thread(self, arg):
        while self.start_head:
            self.Motion.setStiffnesses("Head", 1.0)
            self.Motion.setAngles("Head", [0., self.angle], .05)
            time.sleep(2)

    def stop_head_fix(self):
        self.start_head = False

    def start_head_fix(self):
        arg = tuple([1])
        self.state = True
        thread.start_new_thread(self.head_fix_thread, arg)
        self.start_head = True

    def cal_user_similarity(self):
        cur = self.db.cursor()
        cur.execute("select * from history where month='%d'"%(self.month))
        cust_infos = cur.fetchall()
        usr_info = dict()
        usr_list = dict()
        for r in cust_infos:
            # print(r[0],int(r[1]),r[2],int(r[3]),int(r[4]))
            # r[0]是人名 r[1]是这个人购买的某一项商品 r[2]是当前月份 r[3]是该笔交易中购买该商品的数目 r[4]是该笔交易的单号
            if r[2] not in usr_info:
                usr_info[r[2]] = set()
            usr_info[r[2]].add(r[0])
            if r[0] not in usr_list:
                usr_list[r[0]] = 0
        C = dict()
        N = dict()
        for i, users in usr_info.items():
            for u in users:
                N[u] = 0
                C[u] = dict()

        for i, users in usr_info.items():
            for u in users:
                N[u] += 1
                for v in users:
                    if u == v:
                        continue
                    else:
                        try:
                            C[u].update({v: C[u][v] + 1})
                        except:
                            C[u].update({v: 1})

        W = dict()
        for i, users in usr_info.items():
            for u in users:
                W[u] = dict()
        for u, related_users in C.items():
            for v, cuv in related_users.items():
                W[u][v] = cuv / sqrt(N[u] * N[v])
        self.usr_sim = W
        print "===================================================================="
        print W
        print "===================================================================="

    def get_img(self):
        msg = self.VideoDev.getImageRemote(self.rgb_top)
        w = msg[0]
        h = msg[1]
        c = msg[2]
        data = msg[6]
        ba = str(bytearray(data))
        nparr = np.fromstring(ba, np.uint8)
        img_np = nparr.reshape((h, w, c))
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_np

    def start_face_recog(self, msg):
        self.stop_dialog()
        model = Model()
        model.load_model(file_path='./face_recog/model/face.model.h5')
        color = (0, 255, 0)
        cascade_path = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml"
        num = 0
        pos_person = {'jiashi' : 0,
                      'yushizhuo' : 0}
        while num <= 2:
            frame = self.get_img()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(cascade_path)
            faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            if len(faceRects) > 0:
                if num == 0:
                    self.show_image("recognising111.gif")
                    sentence = "Please do not move. I am trying to find out who you \\pau=30\\ are"
                    self.say(sentence)
                num += 1
                for faceRect in faceRects:
                    x, y, w, h = faceRect
                    image = frame[y - 10 : y + h + 10, x - 10: x + w + 10]
                    faceID = model.face_predict(image)
                    if faceID == 0:
                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)

                        # 文字提示是谁
                        cv2.putText(frame, 'jiashi',
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)  # 字的线宽
                        pos_person['jiashi'] += 1
                    elif faceID == 1:
                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)

                        # 文字提示是谁
                        cv2.putText(frame, 'yushizhuo',
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)  # 字的线宽
                        pos_person['yushizhuo'] += 1
                    else:
                        pass
            cv2.namedWindow("face_pre", cv2.WINDOW_NORMAL)
            cv2.imshow("face_pre", frame)
            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        # 经过3次取值，获取的最有可能的人
        print pos_person
        self.current_person = max(pos_person, key=pos_person.get)
        this_per = self.get_history_item(self.current_person)
        print "========================"
        item_list = list(this_per[self.current_person])
        print item_list
        image_name = self.current_person + '.png'
        # 展示人的身份卡
        self.show_image(image_name)
        sentence = "I see, you must be " + self.current_person + "\\pau=100\\, do you like the " \
                   + item_list[len(item_list) - 1] + " you bought last time ?"
        self.say(sentence)
        self.start_dialog()
        print ('\033[0;32m [Kamerider I] Your name is ' + self.current_person + "\033[0m")

    def say(self, text):
        print '\033[0;32m [Kamerider] say:' + text + "\033[0m"
        time.sleep(0.15)

        self.AnimatedSpe.say(text, self.configure)
        time.sleep(0.15)

    def get_history_item(self, person_name):
        cur = self.db.cursor()
        cur.execute("select * from history where name ='%s' and "
                    "month='%d'" % (person_name, self.month))
        item = cur.fetchall()
        usr_item = dict()
        usr_item[person_name] = set()
        for r in item:
            usr_item[person_name].add(r[2])
        # 这个人以前买过的商品
        return usr_item

    def callback_recommend(self, msg):
        self.stop_dialog()
        this_per = self.get_history_item(self.current_person)
        temp = self.usr_sim[self.current_person].items()
        tt = dict()
        for i in range(len(temp)):
            tt.update({temp[i][1]:temp[i][0]})
        for v, wuv in sorted(tt.items(), reverse=True):
            sim_usr_item = self.get_history_item(wuv)

        temp1 = this_per.values()
        print temp1,"++++++++++++++++++++"
        temp2 = sim_usr_item.values()
        print temp2,"______________________"
        if temp2[0].difference(temp1[0]) != None:
            print temp2[0].difference(temp1[0])
            A = list(temp2[0].difference(temp1[0]))
            A.append("temp")
            # A[0]就是推荐的商品
            self.recommend_item_name = A[0]
            image_name = A[0] + '.png'
            print A[0]
            # 展示图片
            self.show_image(image_name)

        sentence = "According to your consumption record, I think you might like " + \
                   A[0] + ", do you want to try it?"
        self.say(sentence)
        self.start_dialog()

    def show_image(self, item_name):
        print "11111"
        self.TabletSer.hideImage()
        print "1111"
        # 展示推荐的物品对应的图片
        self.TabletSer.showImage("http://198.18.0.1/apps/boot-config/" + item_name)
        # self.TabletSer.showImage("http://198.18.0.1/img/help_charger.png")

    def go_to_waypoint(self, Point):
        self.angle = .1
        self.nav_as.send_goal(Point)
        self.map_clear_srv()
        while self.nav_as.get_state() != 3:
            time.sleep(5)
            self.map_clear_srv()
        self.angle = -.33
        return

    def scan_bar_code(self):
        self.stop_dialog()
        if_succe = False
        # video = cv2.VideoCapture(0)
        while not if_succe:
            # success, frame1 = video.read()
            frame = self.get_img()
            # frame = frame1.copy()
            cv2.imwrite(self.filename, frame)
            scanner = zbar.ImageScanner()

            # configure the reader
            scanner.parse_config('enable')

            # obtain image data
            pil = Image.open(self.filename).convert('L')
            width, height = pil.size
            raw = pil.tobytes()

            # wrap image data
            image = zbar.Image(width, height, 'Y800', raw)

            # scan the image for barcodes
            scanner.scan(image)

            # extract results
            for symbol in image:
                # do something useful with results
                print symbol.data
                if_succe = True
                # 扫出来哪个展示哪一张图片
                self.current_item = symbol.data

        item_name = self.current_item + '.png'
        print "======================================="
        print item_name
        print "======================================="
        self.show_image(item_name)
        if self.first_item == 1:
            self.show_image("half_price1.png")
            self.first_item += 1
            sentence = "Hey, Why don't you take two? \\pau=200\\ The sencod is half price!!"
            self.say(sentence)
        else:
            sentence = "I have recorded this item"
            self.say(sentence)
        self.start_dialog()

    def callback_Bar_code(self, msg):
        self.stop_dialog()
        if_succe = False
        # video = cv2.VideoCapture("rtsp://pepper:pepper@192.168.3.53:6554/MPEG-4/ch1/main/av_stream_0")
        # video = cv2.VideoCapture(0)
        num = 1
        while not if_succe:
            frame = self.get_img()
            # success, frame1 = video.read()
            # frame = frame1.copy()
            cv2.imwrite(self.filename, frame)
            scanner = zbar.ImageScanner()
            # configure the reader
            scanner.parse_config('enable')
            # obtain image data
            pil = Image.open(self.filename).convert('L')
            width, height = pil.size
            raw = pil.tobytes()
            # wrap image data
            image = zbar.Image(width, height, 'Y800', raw)
            # scan the image for barcodes
            scanner.scan(image)
            # extract results
            for symbol in image:
                # do something useful with results
                print symbol.data
                if_succe = True
                # 扫出来哪个展示哪一张图片
                self.current_item = symbol.data
        name = self.current_item + '.png'
        self.show_image(name)
        sentence = "This is White peach jam. It's price is eight yuan."
        self.say(sentence)
        sentence = "I think it's a bit sweet than Blueberry jam."
        self.say(sentence)
        sentence = "And People often apply jam to bread slices for breakfest."
        self.say(sentence)
        sentence = "So I think it is a good idea to buy a bag of bread as well."
        self.say(sentence)
        # sentence = "the infomation is here, you can choose your language"
        # self.say(sentence)
        self.start_dialog()
        # 接下来是选择语言的对话

    def callback_hide_image(self, msg):
        self.TabletSer.hideImage()
        # 展示推荐的物品对应的图片
        self.TabletSer.showImage("http://198.18.0.1/apps/boot-config/welcome.jpg")

    def callback_japan(self, msg):
        self.stop_dialog()
        self.TextToSpe.setLanguage("Japanese")
        sentence = "こんにちは、"
        self.say(sentence)
        sentence = "コンビニエンスストアへようこそ"
        self.say(sentence)
        self.TextToSpe.setLanguage("English")
        self.start_dialog()

    def callback_pointA(self, msg):

        self.if_stop_dia = True
        self.angle = .1
        self.stop_dialog()
        self.go_to_waypoint(self.Point_A)
        # 抬起头看人
        self.angle = -.33
        self.say("here is the " + self.recommend_item_name + " area , I think you may like it")
        self.start_dialog()
        # 接下来问pepper不认识的东西

    def callback_img_lan_CHN(self, msg):
        show_img = self.current_item + "_CHN.jpg"
        self.show_image(show_img)

    def callback_finish(self, msg):
        self.stop_dialog()
        sentence = "Thank you very much! I am looking forward to seeing you next time"
        self.say(sentence)
        self.TabletSer.cleanWebview()
        self.show_image("welcome.jpg")
        self.go_to_waypoint(self.Point_B)
        self.stop_head_fix()

    def callback_record_item(self, msg):
        self.stop_dialog()
        sentence = "please show me the QR code"
        self.say(sentence)
        self.scan_bar_code()
        cur = self.db.cursor()
        cur.execute("insert into history (name,month,item,num,order_id) values('%s','%d','%s','%d','%d')" % (self.current_person, self.month, self.current_item, 1, self.order_id))
        self.db.commit()
        self.price_sum += self.price[self.current_item]
        self.start_dialog()

    def callback_pay_bill(self, msg):
        self.stop_dialog()
        sentence = "The total for your goods is " + str(self.price_sum)
        self.say(sentence)
        self.show_image("QR_code.jpg")
        sentence = "you can scan this QR code to pay the bill"
        self.say(sentence)
        self.start_dialog()
        self.price_sum = 0

    def callback_half_price(self, msg):
        self.price_sum += self.price[self.current_item] / 2

    def stop_dialog(self):
        try:
            self.Dialog.unsubscribe("WRS_pepper")
        except RuntimeError:
            print "[Kamerider W ] : the event \'my_subscribe_test\' hasn't been subscribed"

    def start_dialog(self):
        self.Dialog.subscribe("WRS_pepper")

    def cancel_plan(self):
        self.goal_cancel_pub.publish(GoalID())

    def set_velocity(self, x, y, theta):
        tt = Twist()
        tt.linear.x = x
        tt.linear.y = y
        tt.angular.z = theta
        self.cmd_vel_pub.publish(tt)

    def stop(self):
        self.cancel_plan()
        self.set_velocity(0, 0, 0)

    def activate_keyboard_control(self):
        print ("==========================keyboard control==========================")
        command = ''
        while command != 'c':
            command = raw_input('next command:')
            if command == 'sfr':
                self.start_face_recog()
            elif command == 'sbc':
                self.callback_Bar_code("111")
            elif command == 'w':
                self.set_velocity(.25, 0, 0)
            elif command == 's':
                self.stop()
            elif command == 'x':
                self.set_velocity(-0.25, 0, 0)
            elif command == 'a':
                self.set_velocity(0, 0.25, 0)
            elif command == 'd':
                self.set_velocity(0, -0.25, 0)
            elif command == 'q':
                self.set_velocity(0, 0, 0.35)
            elif command == 'e':
                self.set_velocity(0, 0, -0.35)

def main():
    params = {
        'ip' : "10.3.100.39",
        'port' : 9559,
        'rgb_topic' : 'pepper_robot/camera/front/image_raw'
    }
    pio = pepper_test(params)

if __name__ == "__main__":
    main()
'''
import dlib
import cv2
import time

import zbar
import MySQLdb
import sys
import cv2
import Image
import socket
from math import radians, degrees, sqrt
import time
import argparse
import numpy as np
from threading import Thread
import thread
# 只有一个register函数，用于程序退出时的回调函数
import atexit
#ROS
import rospy
import actionlib
from actionlib_msgs.msg import GoalID
from visualization_msgs.msg import Marker
import sensor_msgs.point_cloud2 as pc2
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, Twist, Pose, PoseStamped, Vector3
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo
from nav_msgs.msg import Odometry, OccupancyGrid
from tf import TransformListener, Transformer, transformations
from std_srvs.srv import Empty
from std_msgs.msg import Int32, String, ColorRGBA
# pexpect
import pexpect
import dynamic_reconfigure.client
import qi


def read_capture(detector):
    num = 100
    success = True
    session = qi.Session()
    session.connect("tcp://192.168.3.59:9559")
    VideoDev = session.service("ALVideoDevice")
    rgb_top = VideoDev.subscribeCamera('rgb_t', 0, 2, 11, 40)

    while success:

        msg = VideoDev.getImageRemote(rgb_top)
        w = msg[0]
        h = msg[1]
        c = msg[2]
        data = msg[6]
        ba = str(bytearray(data))
        nparr = np.fromstring(ba, np.uint8)
        img_np = nparr.reshape((h, w, c))
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # dlib的人脸检测器只能检测80x80和更大的人脸，如果需要检测比它小的人脸，需要对图像上采样，一次上采样图像尺寸放大一倍
        # rects = detector(img,1) #1次上采样
        rects = detector(img_np, 0)
        print rects
        if len(rects) != 0:
            img_name = '%s%d.jpg'%('/home/jiashi/Pictures/keras_dataset/zhangjiashi/', num)
            num += 1

        for rect in rects:  # rect.left(),rect.top(),rect.right(),rect.bottom()
            if len(rects) != 0:
                img2save = img_np[rect.top() - 10: rect.bottom() + 10, rect.left() - 10: rect.right() + 10]
                cv2.imwrite(img_name, img2save)
                print "=================="
                print rect.left()
                print rect.right()
                print rect.top()
                print rect.bottom()
            cv2.rectangle(img_np, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 0, 255), 2, 8)
        cv2.imshow('capture face detection', img_np)
        if cv2.waitKey(1) >= 0:
            break
    cv2.destroyAllWindows()
    video.release()



if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    read_capture(detector)
'''