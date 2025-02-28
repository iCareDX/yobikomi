import asyncio
from datetime import date, datetime
import re
import websockets
from gpiozero import Button, LED

import cv2
import numpy as np
import time
import json
import subprocess
from pathlib import Path
from picamera2 import Picamera2

from logging import getLogger, FileHandler, Formatter, INFO
logger = getLogger(__name__)
logger.setLevel(INFO)
f_handler = FileHandler("./app.log")
log_format = '%(asctime)s,%(msecs)d | %(levelname)s | %(name)s - %(message)s'
formatter = Formatter(fmt=log_format, datefmt='%Y-%m-%d %H:%M:%S')
f_handler.setFormatter(formatter)
logger.addHandler(f_handler)

import sounddevice as sd
import wave

from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

API_KEY = "EMPTY"
API_BASE = "http://iCareDXnoMac-Mini.local:8000/v1"
# API_BASE = "http://192.168.2.1:8000/v1"
MODEL = "llama-3"

model = OpenAI(
    model_name=MODEL, 
    openai_api_key=API_KEY, 
    openai_api_base=API_BASE,
    temperature=1.0
)
chat_model = ChatOpenAI(
    model_name=MODEL, 
    openai_api_key=API_KEY, 
    openai_api_base=API_BASE
    # streaming=True, 
    # callbacks=[StreamingStdOutCallbackHandler()] ,
    # temperature=0
)

is_speaking = False
is_generating = False
is_facedetecting = False
waiting_is_speaking = False

led = LED(25)
button = Button(23)

template = """\
あなたは、株式会社ワイヤレスコミュニケーション研究所の社員です。
株式会社ワイヤレスコミュニケーション研究所は、東京ビックサイトで行われている展示会でブースを構えて次の3点の商品を紹介しています。
1.「エーアイインソール」:新製品です。靴にインソール型センサーを敷いて歩くだけで足圧が一目でわかります。私の後ろでデモをやってるからぜひ見てください。
2.「対話するワンコ」:しゃべりかけると天気を教えてくれたり好きな食べものを教えてくれます。
3.「エーアイスリープ」:ベットの上に敷いて寝るだけで心拍、呼吸や睡眠状態、起き上がったことがわかります。
"""
user_input = """\
展示会の来場者をブースに呼び込むため、来場者に声を掛けてください。
例:「こんにちは、弊社のブースでは次の商品を紹介しております。気になるところや知りたいことがございましたら近くにいる社員にお尋ねください」
"""

JSON_PATH = "./tenki.json"
here_location = {}
today_weather = {}
tomorrow_weather = {}

def read_weather_data(file_path):
# JSONファイルから天気データを読み込み
    global here_location, today_weather, tomorrow_weather
    with open(file_path, 'r', encoding='utf-8') as file:    
        weather_data = json.load(file)
        here_location = weather_data['location']
        today_weather = weather_data['today']['forecasts'][0]
        tomorrow_weather = weather_data['tomorrow']['forecasts'][0]

def play_sign_wave_data():
    wf = wave.open("se_30101.wav")
    fs = wf.getframerate()
    sign_wave_data = wf.readframes(wf.getnframes())   
    sign_wave_data = np.frombuffer(sign_wave_data, dtype='int16')
    sd.play(sign_wave_data, fs)



def generate_context():
    global here_location, today_weather, tomorrow_weather
    d = date.today()
    date_str = d.strftime("%Y年%m月%d日")
    dt = datetime.now()
    datetime_str = dt.strftime("%H時%M分")
    w_list = ['月曜日', '火曜日', '水曜日', '木曜日', '金曜日', '土曜日', '日曜日']
    context = f"""\
    ###
    今日の日付：{date_str}
    今日の曜日：{w_list[dt.weekday()]}
    現在の時間：{datetime_str}
    今日の{here_location}の天気：{today_weather['weather']}
    今日の最高気温：{today_weather['high_temp']}
    今日の最低気温：{today_weather['low_temp']}
    今日の降水確率：{today_weather['rain_probability'].items()}
    明日の天気：{tomorrow_weather['weather']}
    """
    return context

def llm_main(user_input):
    context = generate_context()
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=template+context),
            HumanMessagePromptTemplate.from_template("{user_input}"),
           # HumanMessage(content=user_input),
        ]
    )

    parser = StrOutputParser()

    chain = prompt | chat_model | parser

    logger.info("Send LLM")
    response = chain.invoke({"user_input":user_input})
    print(f"LLM Response: {response}")
    logger.info(f"LLM Response: {response}")
    return response



async def connect_with_retry(uri, max_retries=5, backoff_factor=1.5):
    retries = 0
    while retries < max_retries:
        try:
            websocket = await websockets.connect(uri)
            return websocket
        except (websockets.exceptions.ConnectionClosedError, asyncio.TimeoutError) as e:
            wait_time = backoff_factor ** retries
            print(f"Connection failed with error {e}. Retrying in {wait_time} seconds...")
            logging.error(f"Connection failed with error {e}. Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            retries += 1
    raise Exception(f"Failed to connect to {uri} after {max_retries} attempts")

async def run_test():
    # global is_speaking
    async with websockets.connect('ws://localhost:8767') as websocket_facedetect:
        
        async def process_llm_response(llm_response):
            global is_generating, is_speaking, is_facedetecting, waiting_is_speaking

            async with websockets.connect('ws://localhost:8766') as websocket_tts:
                llm_response = llm_response.strip('「」')
                sentences = re.split(r'(?<=[,.，。？！])\s*|\n', llm_response)
                # print(f"Split sentences: {sentences}")
                # print("in tts")
                waiting_is_speaking = False
                for sentence in sentences:
                    if sentence and is_facedetecting:
                        is_speaking = True
                        led.off()
                        text = sentence.strip()  # Remove any leading/trailing whitespace and newlines
                        print(f"TTS: {text}")
                        logger.info(f"TTS send data: {text}")
                        await websocket_tts.send(text)
                        message = await websocket_tts.recv()  # Wait for TTS completion signal
                        # print(message)
                        logger.info(message)
                        
                        # print("TTS completed")
                    else:
                        break
                is_speaking = False
                led.on()
        async def send_facedetect():
            while True:
                await asyncio.sleep(5)
                await websocket_facedetect.send("Ping")

        async def receive_face_data():
            global is_generating, is_speaking, is_facedetecting, waiting_is_speaking

            while True:
                message = await websocket_facedetect.recv()
                # print(message)

                faces = ["1",]
                message_json = json.loads(message)
                if message_json["faces"] == "1":
                    is_facedetecting = True
                else:
                    is_facedetecting = False
                if is_facedetecting and not is_generating and not is_speaking and not waiting_is_speaking:
                    # print("llm_generating")
                    # print(faces)
                    # print(is_generating)
                     
                    is_generating = True
                    # play_sign_wave_data()
                    led.blink()
                    llm_response_text = llm_main(user_input)
                    task1 = asyncio.create_task(process_llm_response(llm_response_text))
                    # await task1
                    is_generating = False
                    waiting_is_speaking = True
                else:
                    continue

        try:
            # await asyncio.gather(send_facedetect(), receive_face_data())
            tasks = []
            task_s = asyncio.create_task(send_facedetect())
            tasks.append(task_s)
            task_r = asyncio.create_task(receive_face_data())
            tasks.append(task_r)
            await asyncio.gather(*tasks)

        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Retrying facedetect connection due to error: {e}")
            logger.error(f"Retrying facedetect connection due to error: {e}")
            websocket_tts = await connect_with_retry('ws://localhost:8767')
            # await receive_face_date()

async def main():
    read_weather_data(JSON_PATH)
    await run_test()

if __name__ == "__main__":    
    asyncio.run(main())
'''
    llm_response = llm_main(user_input)
    llm_response = llm_response.strip('「」')
    sentences = re.split(r'(?<=[,.，。？！])\s*|\n', llm_response)
    # print(f"Split sentences: {sentences}")
    for sentence in sentences:
        if sentence: 
            vvox_tts(sentence)
'''
