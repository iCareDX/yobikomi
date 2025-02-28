import re
import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path

#JSON_PATH = "/home/icaredx/VOSKServer/tenki.json"
JSON_PATH = "./tenki.json"
# URL = 'https://tenki.jp/forecast/3/16/4410/13206/' #府中市の気象情報URL
URL = 'https://tenki.jp/forecast/3/16/4410/13108/' #江東区の気象情報URL


def main(url):
    # bs4でパース
    s = soup(url)

    dict = {}

    # 予測地点
    l_pattern = r"(.+)の今日明日の天気"
    l_src = s.title.text
    dict['location'] = re.findall(l_pattern, l_src)[0]

    soup_tdy = s.select('.today-weather')[0]
    soup_tmr = s.select('.tomorrow-weather')[0]

    dict["today"] = forecast2dict(soup_tdy)
    dict["tomorrow"] = forecast2dict(soup_tmr)
 
#    print(json.dumps(dict, ensure_ascii=False))  # JSON形式で出力
      
    # './tenki.json' にデータを書き込む　上書きモード
    with open(JSON_PATH, 'w', encoding='utf-8') as wfile:
       json.dump(dict, wfile, ensure_ascii=False, indent=4)
       
#    wfile=open("/Users/icaredx11/GIT/VOSKServer/weather_data/demofile2.txt", 'a', encoding='utf-8')
#    wfile.write(dict['location'] + "の天気データがtenki.jsonに保存されました")
#    wfile.close()

    print(dict['location'] + "の天気データがtenki.jsonに保存されました") 


def soup(url):
    r = requests.get(url)
    html = r.text.encode(r.encoding)
    return BeautifulSoup(html, 'html.parser')

def forecast2dict(soup):
    data = {}

    # 日付処理
    d_pattern = r"(\d+)月(\d+)日\(([土日月火水木金])+\)"
    d_src = soup.select('.left-style')
    date = re.findall(d_pattern, d_src[0].text)[0]
    data["date"] = "%s-%s(%s)" % (date[0], date[1], date[2])
#    print("=====" + data["date"] + "=====")

    # ## 取得
    weather           = soup.select('.weather-telop')[0]
    high_temp         = soup.select("[class='high-temp temp']")[0]
    high_temp_diff    = soup.select("[class='high-temp tempdiff']")[0]
    low_temp          = soup.select("[class='low-temp temp']")[0]
    low_temp_diff     = soup.select("[class='low-temp tempdiff']")[0]
    rain_probability  = soup.select('.rain-probability > td')
    wind_wave         = soup.select('.wind-wave > td')[0]

    # ## 格納
    data["forecasts"] = []
    forecast = {}
    forecast["weather"] = weather.text.strip()
    forecast["high_temp"] = high_temp.text.strip()
    forecast["high_temp_diff"] = high_temp_diff.text.strip()
    forecast["low_temp"] = low_temp.text.strip()
    forecast["low_temp_diff"] = low_temp_diff.text.strip()
    every_6h = {}
#    for i in range(4):
#        time_from = 0+6*i
#        time_to   = 6+6*i
#        itr       = '{:02}-{:02}'.format(time_from,time_to)
#        every_6h[itr] = rain_probability[i].text.strip()

    # 時間帯を音声で読みやすいようにindex名を指定
    every_6h["0時〜6時"] = rain_probability[0].text.strip()
    every_6h["6時〜12時"] = rain_probability[1].text.strip()
    every_6h["12時〜18時"] = rain_probability[2].text.strip()
    every_6h["18時〜24時"] = rain_probability[3].text.strip()

    forecast["rain_probability"] = every_6h
    forecast["wind_wave"] = wind_wave.text.strip()

    data["forecasts"].append(forecast)

#    print(
#        "天気              ： " + forecast["weather"] + "\n"
#        "最高気温(C)       ： " + forecast["high_temp"] + "\n"
#        "最高気温差(C)     ： " + forecast["high_temp_diff"] + "\n"
#        "最低気温(C)       ： " + forecast["low_temp"] + "\n"
#        "最低気温差(C)     ： " + forecast["low_temp_diff"] + "\n"
#        "降水確率[00-06]   ： " + forecast["rain_probability"]['00-06'] + "\n"
#        "降水確率[06-12]   ： " + forecast["rain_probability"]['06-12'] + "\n"
#        "降水確率[12-18]   ： " + forecast["rain_probability"]['12-18'] + "\n"
#        "降水確率[18-24]   ： " + forecast["rain_probability"]['18-24'] + "\n"
#        "風向              ： " + forecast["wind_wave"] + "\n"
#    )

    return data

if __name__ == '__main__':
    main(URL)

