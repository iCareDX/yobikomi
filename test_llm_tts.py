import asyncio
from datetime import date, datetime
import re
import websockets
from gpiozero import Button, LED

from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

API_KEY = "EMPTY"
API_BASE = "http://iCareDXnoMac-Mini.local:8000/v1" #  nakamura mac
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

led = LED(25)
button = Button(23)

template = """\
あなたは、株式会社ワイヤレスコミュニケーション研究所の社員です。
株式会社ワイヤレスコミュニケーション研究所は、東京ビックサイトで行われている展示会でブースを構えて次の3点の商品を紹介しています。
1.「Ai step」:新製品です。靴にインソール型センサーを敷いて歩くだけで足圧が一目でわかります。私の後ろでデモをやってるからぜひ見てください。
2.「対話するワンコ」:しゃべりかけると天気を教えてくれたり好きな食べものを教えてくれます。
3.「AiSleep」:ベットの上に敷いて寝るだけで心拍、呼吸や睡眠状態、起き上がったことがわかります。
"""
user_input = """\
展示会の来場者をブースに呼び込むため、来場者に声を掛けてください。
例:「こんにちは、弊社のブースでは次の商品を紹介しております。気になるところや知りたいことがございましたら近くにいる社員にお尋ねください」
"""



def generate_context():
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

    response = chain.invoke({"user_input":user_input})
    print(response)
    return response

import numpy as np
import json
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
    query_data['speedScale'] = 1.2

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
async def send_to_tts(websocket_tts, text):
    """Send text to TTS server and wait for completion signal."""
    global is_speaking
    is_speaking = True
    led.off()

    text = text.strip()  # Remove any leading/trailing whitespace and newlines
    print(f"TTS send data: {text}")
    await websocket_tts.send(text)
    await websocket_tts.recv()  # Wait for TTS completion signal
    is_speaking = False
    led.on()
    # print("TTS completed")

async def process_llm_response(websocket_tts, llm_response):
    # led.off()
    
    llm_response = llm_response.strip('「」')
    sentences = re.split(r'(?<=[,.，。？！])\s*|\n', llm_response)
    # print(f"Split sentences: {sentences}")
    for sentence in sentences:
        if sentence:
            await send_to_tts(websocket_tts, sentence)


async def run_test():
    # global is_speaking
    async with websockets.connect('ws://localhost:8766') as websocket_tts:
        # while True:

        llm_response_text = llm_main(user_input)
        await process_llm_response(websocket_tts, llm_response_text)
    # led.off())

async def main():

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
