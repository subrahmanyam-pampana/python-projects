import requests
import sys

API_KEY = "dc9194b88e3b680b6915713352729bab"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

city = sys.argv[1]
request_url = f"{BASE_URL}?appid={API_KEY}&q={city}"
response = requests.get(request_url)

if response.status_code == 200:
    data = response.json()
    weather = data["weather"][0]["description"]
    print("weather: ", weather)
    temperature = round(data["main"]["temp"] - 273.15, 2)
    print("temperature: ", temperature, "C")
    # print(data)
else:
    print("An Error Occured!")
