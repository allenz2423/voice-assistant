import json
import urllib.request
import urllib.parse
from typing import Optional

async def get_weather_report(location: Optional[str] = "") -> str:
    """Fetches real-time weather and forecast via wttr.in JSON API."""
    loc_clean = (location or "").strip()
    encoded_loc = urllib.parse.quote(loc_clean) if loc_clean and loc_clean.lower() not in ["local", "here", "current"] else ""
    url = f"https://wttr.in/{encoded_loc}?format=j1"
    headers = {"User-Agent": "curl/8.0"}

    try:
        req = urllib.request.Request(url, headers=headers)
        # wttr.in is fast and returns standard JSON with format=j1
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))

        current = data.get("current_condition", [{}])[0]
        area = data.get("nearest_area", [{}])[0]
        area_name = area.get("areaName", [{}])[0].get("value", "your area")
        region = area.get("region", [{}])[0].get("value", "")

        temp_f = current.get("temp_F", "")
        temp_c = current.get("temp_C", "")
        feels_f = current.get("FeelsLikeF", "")
        feels_c = current.get("FeelsLikeC", "")
        desc = current.get("weatherDesc", [{}])[0].get("value", "")
        humidity = current.get("humidity", "")
        wind_mph = current.get("windspeedMiles", "")

        today_forecast = data.get("weather", [{}])[0]
        max_f = today_forecast.get("maxtempF", "")
        min_f = today_forecast.get("mintempF", "")
        max_c = today_forecast.get("maxtempC", "")
        min_c = today_forecast.get("mintempC", "")

        loc_str = f"in {area_name}" if not region else f"in {area_name}, {region}"
        if loc_clean and loc_clean.lower() not in ["local", "here", "current"]:
            loc_str = f"in {loc_clean.title()}"

        report = (
            f"Currently {loc_str}, it is {temp_f} degrees Fahrenheit ({temp_c} Celsius) and {desc.lower()}. "
            f"It feels like {feels_f} degrees with {humidity} percent humidity and winds at {wind_mph} miles per hour. "
            f"Today's forecast has a high of {max_f} degrees ({max_c} Celsius) and a low of {min_f} degrees ({min_c} Celsius)."
        )
        return report
    except Exception as e:
        return f"Unable to retrieve weather data at this moment: {e}"
