'''
bot_face_data_creator.py

ボイスアシスタントの顔認証と年齢性別データを作成するためのスクリプトです
カメラからユーザーの顔を検出し、顔画像を収集して特徴量を抽出し、性別と年齢を推論してJSONファイルに保存します
顔認証システムのトレーニングやデータ収集に使用します
'''

import cv2
import numpy as np
import time
import json
import subprocess
from pathlib import Path
from picamera2 import Picamera2
# from bot_motor_controller import pan_tilt, neopixels_all, neopixels_off
import requests
import sounddevice as sd
import struct

def vvox_tts(text):
    # エンジン起動時に表示されているIP、portを指定
 #   host = "iCareDXnoMac-mini.local" # nakamura mac
 #   host = "192.168.11.144"           # MacM2
    host = "iCareDXnoMac-mini.local" #"iCareDXMacM2.local"
    port = 50021
    # 音声化する文言と話者を指定(3で標準ずんだもんになる)
    params = (
        ('text', text),
        ('speaker', 3),
    )
    # 音声合成用のクエリ作成
    query = requests.post(
        f'http://{host}:{port}/audio_query',
        params=params
    )
    # 音声クエリのJSONデータを取得し、話速(speedScale)を指定
    query_data = query.json()
    query_data['speedScale'] = 0.8

    # 音声合成を実施
    synthesis = requests.post(
        f'http://{host}:{port}/synthesis',
        headers = {"Content-Type": "application/json"},
        params = params,
        data = json.dumps(query_data)
 #        data = json.dumps(query.json())     # 話速指定のために外しました
    )
    # 音声データを取得
    content_len = len(synthesis.content)
    # print(content_len)
    
    if content_len > 44:
        riff = struct.unpack_from("<4s",synthesis.content,0)
        if riff[0] == b'RIFF':
            riff_size, riff_type = struct.unpack_from("<I4s",synthesis.content,4)
            chunk_offset = 12
            # print(riff_size, riff_type, chunk_offset)
            sample_per_sec = 0
            data_block_offset = 0
            data_block_size = 0
            
            while True:
                chunk_type, chunk_size = struct.unpack_from("<4sI",synthesis.content,chunk_offset)
                chunk_offset += 8
                # print(chunk_type, chunk_size, chunk_offset)
                if chunk_type == b'fmt ':
                    format_id, channels, sample_per_sec, bytes_per_sec, block_size, bits_per_sample = struct.unpack_from("<HHIIHH",synthesis.content,chunk_offset)
                    # print(format_id, channels, sample_per_sec, bytes_per_sec, block_size, bits_per_sample)
                elif chunk_type == b'data':
                    data_block_offset = chunk_offset
                    data_block_size = chunk_size
                    break
                else:
                    # pass
                    break
                
                chunk_offset += chunk_size
                
            # print(data_block_offset, data_block_size)
            if (data_block_offset == 44 and
                (data_block_offset + data_block_size) <= content_len):
                voice = synthesis.content[data_block_offset:(data_block_offset + data_block_size)]
                # 音声データをNumPy配列に変換（16ビットPCMに対応）
                voice_data = np.frombuffer(voice, dtype=np.int16)
                # サンプリングレートが24000以外だとずんだもんが高音になったり低音になったりする
                sample_rate = 24000
                # 再生処理 (sounddeviceを使用)
                sd.play(voice_data, samplerate=sample_rate)
                sd.wait()  # 再生が終わるまで待機
            


# カメラ映像を表示するウィンドウサイズの設定のためにスクリーンの解像度を取得する
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

# jsonファイルを作成する関数
def save_json(id, name, image, feature, gender, age, category, interested):
    user = {id:{
        "id": id,
        "name": name,
        "image": image,
        "feature": feature,
        "gender": gender,
        "age": age,
        "category": category,
        "interested": interested
    }
    }

    isempty = Path("data/user_data.json").stat().st_size == 0

    if isempty is True:
        with open(Path("data/user_data.json"), "w") as file:
            json.dump(user, file, ensure_ascii=False, indent=4)
    else:
        with open(Path("data/user_data.json")) as file:
            load_user = json.load(file)

        save_user = dict(load_user, **user)
        
        with open(Path("data/user_data.json"), 'w') as file:
            json.dump(save_user, file, ensure_ascii=False, indent=4)

def face_date_create():
    # 顔認識モデルの読み込み
    face_detector_weights = str(Path("dnn_models/face_detection_yunet_2023mar.onnx").resolve()) #元のモデルのリンクが消失しているため、代わりにこちらを使用
    #face_detector_weights = str(Path("dnn_models/yunet_s_640_640.onnx").resolve())
    face_detector = cv2.FaceDetectorYN_create(face_detector_weights, "", (0, 0))

    # 顔識別モデルを読み込む
    face_recognizer_weights = str(Path("dnn_models/face_recognizer_fast.onnx").resolve())
    face_recognizer = cv2.FaceRecognizerSF_create(face_recognizer_weights, "")

    # 年齢識別モデルを読み込む
    ageProto = str(Path("dnn_models/age_deploy.prototxt").resolve())
    ageModel = str(Path("dnn_models/age_net.caffemodel").resolve())

    # 性別識別モデルを読み込む
    genderProto = str(Path("dnn_models/gender_deploy.prototxt").resolve())
    genderModel = str(Path("dnn_models/gender_net.caffemodel").resolve())

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['male', 'female']

    # DNNネットワークに接続
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    # CPU使用を指定
    ageNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    genderNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

    # ユーザーデータの初期化
    user_id = ""
    user_name = ""
    user_image = ""
    user_feature = ""
    user_gender = ""
    user_age = ""
    user_category = ""
    user_interested = ""

    # # ユーザー名、ユーザーIDの入力
    # print("🖥️ SYSTEM: ユーザー名をひらがな（またはカタカナ）で入力してEterキーを押してください")
    # user_name = input("> ")
    # print("🖥️ SYSTEM: ユーザーIDをアルファベット（正規表現）で入力してEnterキーを押してください")
    # user_id = input("> ")
    # print("🖥️ SYSTEM: 興味のあることをひとつ入力してEnterキーを押してください")
    # user_interested = input("> ")
    # print("🖥️ SYSTEM: 画像データを 撮影します\n撮影はSキーを押してください\n終了はQキーを押してください")

    # カメラのデフォルトのパン/チルト (度単位)。コードを開始するときに、おおよその顔の位置を指すように設定しました
    # カメラの範囲は 0 ～ 180 です。以下の値を変更して、パンとチルトの開始点を決定します。
    cam_pan = 90
    cam_tilt = 60
    # カメラを開始位置に向けます (pan() 関数とtilt() 関数が期待するデータは -90 度から 90 度までの任意の数値です)
    # pan_tilt(cam_pan-90,cam_tilt-90)

    cam = Camera()  # カメラオブジェクトを作成
    # neopixels_all(50, 50, 50)

    # カメラ映像を表示するウィンドウのサイズを設定用
    screen_width, screen_height = get_screen_resolution()
    frame_width = int(screen_width * 0.7)
    frame_height = int(screen_height * 0.7)

    while(True):
        frame = cam.get_frame()  # カメラからフレームを取得
        frame = cv2.flip(frame, 0)  # カメラ画像の上下を入れ替える

        # 入力サイズを指定する
        height, width, _ = frame.shape
        face_detector.setInputSize((width, height))

        # 顔を検出する
        _, faces = face_detector.detect(frame)
        faces = faces if faces is not None else []

        # 検出した顔のバウンディングボックスとランドマークを描画する
        frame_output = frame.copy() 

        for face in faces:
            # バウンディングボックス
            x, y, w, h = list(map(int, face[:4]))
            color = (255, 255, 255)
            thickness = 1
            cv2.rectangle(frame_output, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)

            # ランドマーク（右目、左目、鼻、右口角、左口角）
            landmarks = list(map(int, face[4:len(face)-1]))
            landmarks = np.array_split(landmarks, len(landmarks) / 2)
            for landmark in landmarks:
                radius = 3
                thickness = -1
                cv2.circle(frame_output, landmark, radius, color, thickness, cv2.LINE_AA)
            
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

            # パン/チルト0～180度 に固定
            cam_pan = max(0,min(180,cam_pan))
            cam_tilt = max(0,min(180,cam_tilt))

            # サーボの更新
            # pan_tilt(int(cam_pan-90),int(cam_tilt-90))

            break

        # 画像を表示する
        frame_output = cv2.resize(frame_output, (frame_width, frame_height))
        cv2.imshow("face data create", frame_output)
        key = cv2.waitKey(10)

        if key == ord('s'):
            # 検出された顔を切り抜く
            aligned_faces = []
            if faces is not None:
                for face in faces:
                    aligned_face = face_recognizer.alignCrop(frame, face)
                    aligned_faces.append(aligned_face)

            # 画像を保存する
            for i, aligned_face in enumerate(aligned_faces):
                user_image = user_id + ".jpg"
                cv2.imwrite((str(Path("face_dataset/" + user_image))), aligned_face)
                cv2.imshow("aligned_face", aligned_face)
            
                # 特徴を抽出する
                aligned_face_img = cv2.imread(str(Path("face_dataset/" + user_image)))
                face_feature = face_recognizer.feature(aligned_face_img)

                # 特徴を保存する
                user_feature = user_id + ".npy"
                dictionary = Path("face_dataset/" + user_feature)
                np.save(dictionary , face_feature)

            # 性別を推論する
            blob = cv2.dnn.blobFromImage(frame_output, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            user_gender = genderList[genderPreds[0].argmax()]
            print("🖥️ SYSTEM: 性別 : {}, conf = {:.3f}".format(user_gender, genderPreds[0].max()))

            # 年齢を推論する
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            user_age = ageList[agePreds[0].argmax()]
            print("🖥️ SYSTEM: 年齢 : {}, conf = {:.3f}".format(user_age, agePreds[0].max()))

            # カテゴリー分類
            if user_age in ageList[:4]:
                if user_gender == "male":
                    user_category = "boy"
                else:
                    user_category = "girl"
            elif user_age in ageList[4: 8]:
                if user_gender == "male":
                    user_category = "man"
                else:
                    user_category = "woman"

            # jsonファイルを保存する
            # save_json(user_id, user_name, user_image, user_feature, user_gender, user_age, user_category, user_interested)
            print("🖥️ SYSTEM: ユーザーデータ\n"
                  f"ID: {user_id} \n"
                  f"名前: {user_name} \n"
                  f"写真: {user_image} \n"
                  f"特徴量: {user_feature} \n"
                  f"性別: {user_gender} \n"
                  f"年齢: {user_age} \n"
                  f"分類: {user_category} \n"
                  f"興味: {user_interested} \n"
                  "を保存しました")
            vvox_tts("こんにちわ")

        if key == ord('q'):
            print("🖥️ SYSTEM: 撮影を終了します")
            break

    cam.release_camera()  # カメラを解放
    cv2.destroyAllWindows()
    time.sleep(0.5)
    # pan_tilt(0,0)
    time.sleep(0.5)
    # neopixels_off()

if __name__ == '__main__':
    face_date_create()
