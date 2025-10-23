import os
import requests
import pandas as pd
from datetime import datetime
import time
import sched


# -------------------------READ SENSOR DATA--------------------------------------
def get_sensor_readings(station_id):
    url = f"https://tie.digitraffic.fi/api/weather/v1/stations/{station_id}/data"
    try:
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"Weather API failed for station {station_id}")
            return []

        data = resp.json().get('sensorValues', [])
        readings = []
        for sensor in data:
            name = sensor.get('name')
            value = float(sensor.get('value', -1))
            time_str = sensor.get('measuredTime')
            if not time_str:
                continue
            time_val = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            readings.append({"name": name, "value": value, "time": time_val})
        return readings
    except Exception as e:
        print("Sensor fetch error:", e)
        return []


# -----------------------CLASSIFY THE TIMESTAMPS---------------------------------
def classify_timestamps(readings):
    dry_times, wet_times = [], []
    for reading in readings:
        if reading["name"] != "SADE":
            continue
        sade = reading["value"]
        t = reading["time"]

        v = next((r for r in readings if r["name"] == "VEDEN_MÄÄRÄ1" and abs((r["time"] - t).total_seconds()) <= 60),
                 None)
        k = next((r for r in readings if r["name"] == "KITKA1_LUKU" and abs((r["time"] - t).total_seconds()) <= 60),
                 None)

        if not v or not k:
            continue

        if sade == 0.0 and v["value"] <= 0.1 and k["value"] >= 0.72:
            dry_times.append(t)
        else:
            wet_times.append(t)
    return dry_times, wet_times


# ----------------------DOWNLOAD IMAGE-------------------------------------------
def download_image(url):
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.content
    except Exception as e:
        print(f"Image download failed: {e}")
    return None


# ----------------------SAVE IMAGE LOCALLY---------------------------------------
def save_image(local_dir, filename, img_bytes):
    try:
        os.makedirs(local_dir, exist_ok=True)
        filepath = os.path.join(local_dir, filename)
        with open(filepath, "wb") as f:
            f.write(img_bytes)
        print(f"Saved: {filepath}")
    except Exception as e:
        print(f"Save failed: {e}")


# ----------------------COMPARE TIMESTAMPS--------------------------------------
def find_matching_time(image_time, timestamps):
    for t in timestamps:
        if abs((image_time - t).total_seconds()) <= 540:
            return True
    return False


# ------------------------------------MAIN-----------------------------------------------------
def main():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), " - Script started...")
    # get camera and weather station IDs
    df = pd.read_csv("camera_and_weatherstation_id_reduced.csv")
    df.columns = df.columns.str.strip()

    for _, row in df.iterrows():
        cam_id = row["camera_id"].strip()
        topdown_base = str(row["topdown_id"]).strip()
        station_id = str(row["station_id"]).strip()

        time.sleep(1)
        print(f"\nProcessing camera {cam_id} + station {station_id}")
        readings = get_sensor_readings(station_id)
        dry_timestamps, wet_timestamps = classify_timestamps(readings)

        # for each of the three camera suffixes
        for suffix in ("01", "02", "09"):
            full_topdown_id = f"{topdown_base}{suffix}"

            # for storing images: 01/02/09/dataset-dry-images and dataset-wet-images
            dry_dir = os.path.join(suffix, "dataset-dry-images")
            wet_dir = os.path.join(suffix, "dataset-wet-images")
            os.makedirs(dry_dir, exist_ok=True)
            os.makedirs(wet_dir, exist_ok=True)

            cam_url = f"https://tie.digitraffic.fi/api/weathercam/v1/stations/{full_topdown_id}/history"
            resp = requests.get(cam_url)
            if resp.status_code != 200:
                print(f"Camera API failed for {full_topdown_id}")
                continue

            data = resp.json()
            for preset in data.get("presets", []):
                for image_data in preset.get("history", []):
                    url = image_data.get("imageUrl")
                    timestamp = image_data.get("lastModified")
                    if not url or not timestamp:
                        continue

                    img_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    is_dry = find_matching_time(img_time, dry_timestamps)
                    is_wet = find_matching_time(img_time, wet_timestamps)

                    if not is_dry and not is_wet:
                        continue

                    save_dir = dry_dir if is_dry else wet_dir
                    filename = f"{cam_id}_{img_time.isoformat().replace(':', '-')}.jpg"

                    img_bytes = download_image(url)
                    if img_bytes:
                        save_image(save_dir, filename, img_bytes)
                    else:
                        print(f" Failed to save {filename}")

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), " - Script finished.")
    scheduler.enter(3600, 1, main)


# ---------------------------------INITIALIZATION-------------------------------
scheduler = sched.scheduler(time.time, time.sleep)
if __name__ == "__main__":
    main()
    # scheduler.enter(3600, 1, main)
    # scheduler.run()

