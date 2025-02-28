from datetime import date, datetime
import json

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
        print( type(today_weather))
        print( today_weather['weather'])
        


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

if __name__ == "__main__": 
    read_weather_data(JSON_PATH)
    print(generate_context())