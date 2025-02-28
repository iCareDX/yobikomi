'''
bot_face_track_recognizer.py

カメラ映像を取得し、顔を検出して認識する顔追跡システムのボット用スクリプトです
カメラで顔を検出し、顔の特徴を抽出して辞書と比較し、顔認識を行います
また、顔の中心を捉えてカメラのパンとチルトを制御し、顔の追跡も行います
'''

import cv2
import numpy as np
import time
import subprocess
from pathlib import Path
from collections import Counter
from picamera2 import Picamera2
import json
# from bot_motor_controller import pan_tilt, neopixels_all, neopixels_off

def get_screen_resolution():
    output = subprocess.check_output("xrandr | grep '*' | awk '{print $1}'", shell=True)
    resolution = output.decode('utf-8').strip().split('x')
    width = int(resolution[0])
    height = int(resolution[1])
    return width, height

class Camera():
    def __init__(self):
        # self.cap = cv2.VideoCapture(0)
        # self.cap.set(3, 640)
        # self.cap.set(4, 480)
        self.picam2 = Picamera2()
        # rawを設定することでカメラの最高解像度を利用し、画角を広げます。（指定しなければデジタルズームされた狭い画角になる）
        fullReso = self.picam2.camera_properties['PixelArraySize']
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}, raw={"size": fullReso}))
        self.picam2.start()

    def get_frame(self):
        #ret, frame = self.cap.read()
        frame = self.picam2.capture_array()
        if frame is not None:
            return frame
        else:
            print("🖥️ SYSTEM: カメラからのフレーム取得に失敗しました。")
            return None

    def release_camera(self):
        self.picam2.stop()
        self.picam2.close()

def face_recognize():
    # モデルの読み込み
    face_detector_weights = str(Path("dnn_models/face_detection_yunet_2023mar.onnx").resolve())
    #face_detector_weights = str(Path("dnn_models/yunet_s_640_640.onnx").resolve())  # 顔検出用のweights
    face_detector = cv2.FaceDetectorYN_create(face_detector_weights, "", (0, 0))

    # 顔識別モデルを読み込む
    face_recognizer_weights = str(Path("dnn_models/face_recognizer_fast.onnx").resolve())  # 顔認識用のweights
    face_recognizer = cv2.FaceRecognizerSF_create(face_recognizer_weights, "")

    COSINE_THRESHOLD = 0.363
    #NORML2_THRESHOLD = 1.128

    # 特徴を読み込み特徴量辞書をつくる
    dictionary = []
    files = Path("face_dataset").glob("*.npy")
    for file in files:
        feature = np.load(file)
        user_id = Path(file).stem
        dictionary.append((user_id, feature))

    # 特徴を辞書と比較してマッチしたユーザーとスコアを返す関数
    def match(recognizer, feature1, data_directory):
        for element in data_directory:
            user_id, feature2 = element
            score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
            if score > COSINE_THRESHOLD:
                return True, (user_id, score)
        return False, ("", 0.0)
    
    recognized_ids =[]
    
    # カメラのデフォルトのパン/チルト (度単位)。コードを開始するときに、おおよその顔の位置を指すように設定しました
    # カメラの範囲は 0 ～ 180 です。以下の値を変更して、パンとチルトの開始点を決定します。
    cam_pan = 90
    cam_tilt = 60
    # カメラを開始位置に向けます (pan() 関数とtilt() 関数が期待するデータは -90 度から 90 度までの任意の数値です)
    # pan_tilt(cam_pan-90,cam_tilt-90)

    cam = Camera()  # カメラオブジェクトを作成
    # neopixels_all(50, 50, 50)

    # time_start = time.perf_counter()
    # time_end = 0

    # 各顔のIDごとの前のフレームのバウンディングボックスの面積
    previous_areas = {}
    i = 0
    frame_count = 0
    frame_skip = 5
    display_message = ""
    before_most_common_id = ""
    screen_width, screen_height = get_screen_resolution()
    frame_width = int(screen_width * 0.7)
    frame_height = int(screen_height * 0.7)
    with open("data/user_data.json", "r", encoding="utf-8") as f:
        user_data = json.load(f)
    start_time = time.time()

    while True:
        frame = cam.get_frame()  # カメラからフレームを取得
        frame = cv2.flip(frame, -1)  # カメラ画像の上下を入れ替える

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame_count += 1

        # 入力サイズを指定する
        height, width, _ = frame.shape
        face_detector.setInputSize((width, height))

        # 顔を検出する
        _, faces = face_detector.detect(frame)
        faces = faces if faces is not None else []

        # 検出した顔のバウンディングボックスとランドマークを描画する
        frame_output = frame.copy()

        for face in faces:
            # 顔を切り抜き特徴を抽出する
            aligned_face = face_recognizer.alignCrop(frame, face)
            feature = face_recognizer.feature(aligned_face)

            # 辞書とマッチングする
            result, user = match(face_recognizer, feature, dictionary)

            # マッチングしたらボックスとテキストの色を変える
            if result is True:
                color = (0,255,0)
                # neopixels_all(0, 50, 0)
            else:
                color = (255,255,255)
                # neopixels_all(50, 50, 50)

            # バウンディングボックス
            x, y, w, h = list(map(int, face[:4]))
            thickness = 2
            cv2.rectangle(frame_output, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)

            # 処理速度向上のため、ランドマークの描画をコメントアウト
            # # ランドマーク（右目、左目、鼻、右口角、左口角）
            # landmarks = list(map(int, face[4:len(face)-1]))
            # landmarks = np.array_split(landmarks, len(landmarks) / 2)
            # for landmark in landmarks:
            #     radius = 3
            #     thickness = -1
            #     cv2.circle(frame_output, landmark, radius, color, thickness, cv2.LINE_AA)
            
            # 認識の結果を描画する
            id, score = user if result else ("unknown", 0.0)
            text = "{0} ({1:.2f})".format(id, score)
            position = (x, y - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 1
            cv2.putText(frame_output, text, position, font, scale, color, thickness, cv2.LINE_AA)

            # マッチングしたらIDを一度だけ追加する
            if result:
                recognized_ids.append(id)
                #print(recognized_ids)

            i += 1
            current_area = w * h
            position = (x, y - 30)
            if i % 1 == 0:
                print(current_area)
                color = (255, 255, 255)
                if id in previous_areas:
                    previous_area = previous_areas[id]
                    diff_range = previous_area / 30
                    if current_area > previous_area + diff_range:
                        print(id + "は近づいています")
                        text = "approaching"
                        cv2.putText(frame_output, text, position, font, scale, color, thickness, cv2.LINE_AA)
                    elif current_area < previous_area - diff_range:
                        print(id + "は遠ざかっています")
                        text = "moving away"
                        cv2.putText(frame_output, text, position, font, scale, color, thickness, cv2.LINE_AA)
                    else:
                        print(id + "は静止しています")
                        text = "not moving"
                        cv2.putText(frame_output, text, position, font, scale, color, thickness, cv2.LINE_AA)
                else:
                    print(id)

            previous_areas[id] = current_area

            # 顔の中心を捉える
            x = x + (w/2)
            y = y + (h/2)

            # 画像の中心を基準として補正
            turn_x  = float(x - (width / 2))
            turn_y  = float(y - (height / 2))

            # オフセット・パーセンテージに変換
            turn_x  /= float(width / 2)
            turn_y  /= float(height / 2)

            # Sスケールオフセットを度数に変換
            #（下の2.5の値はPIDの比例係数のような働きをします）
            turn_x   *= 2.5 # VFOV
            turn_y   *= 2.5 # HFOV
            cam_pan  += -turn_x
            cam_tilt += turn_y

            #print(cam_pan-90, cam_tilt-90)

            # パン/チルト0～180度 に固定
            cam_pan = max(0,min(180,cam_pan))
            cam_tilt = max(0,min(180,cam_tilt))

            # サーボの更新
            # pan_tilt(int(cam_pan-90),int(cam_tilt-90))
        
        if frame is not None:
            frame_output = cv2.resize(frame_output, (frame_width, frame_height))
            cv2.putText(frame_output, display_message, (0, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            cv2.imshow("face detection", frame_output)

        elapsed_time = time.time() - start_time
        if elapsed_time >= 5:
            if recognized_ids:
                most_common_id = Counter(recognized_ids).most_common(1)[0][0]
                if most_common_id != before_most_common_id:
                    name = user_data.get(most_common_id, {}).get("name", "unknown")
                    send_message = f"前にいるのは{name}さんです。"
                    display_message = f"Hello, {most_common_id}!"
                    speak_message = f"こんにちは、{name}さん！"
                    subprocess.Popen(['./jvoice.sh', speak_message])
                    before_most_common_id = most_common_id
            else:
                display_message = ""
            recognized_ids.clear()
            start_time = time.time()

        # time_end = time.perf_counter() - time_start
        # if time_end > 5:
        #     break

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cam.release_camera()  # カメラを解放
    cv2.destroyAllWindows()
    time.sleep(0.5)
    # pan_tilt(0,0)
    # neopixels_off()
    if recognized_ids:
        return [item[0] for item in Counter(recognized_ids).most_common(7)]
    else:
        return []
    
if __name__ == '__main__':
    # print(get_screen_resolution())
    recognized_id = face_recognize()
    print(recognized_id)
