import re
import random
import shutil
import datetime
from pathlib import Path
from collections import defaultdict, deque

IMG_DIR = Path(r"snow_covered_cleaned")
OUT_DIR = Path(r"../picture/snow_covered")
SAMPLE_SIZE = 100
CITY_TZ = "Europe/Helsinki"
random.seed(42)
# ===========================================

# format1: C01507_2025-10-04T00-51-18+00-00.jpg
pattern1 = re.compile(
    r"^(?P<cam>C\d+?)_(?P<date>\d{4}-\d{2}-\d{2})T"
    r"(?P<h>\d{2})-(?P<m>\d{2})-(?P<s>\d{2})\+(?P<tzh>\d{2})-(?P<tzm>\d{2})\.(?P<ext>jpg|jpeg|png|bmp)$",
    re.IGNORECASE,
)

# format2: cam1_4-10_15-30.jpg
pattern2 = re.compile(
    r"^(?P<cam>cam\d+?)_(?P<d>\d{1,2})-(?P<m>\d{1,2})_(?P<h>\d{1,2})-(?P<mi>\d{1,2})\.(?P<ext>jpg|jpeg|png|bmp)$",
    re.IGNORECASE,
)


def parse_name(name: str):
    m = pattern1.match(name)
    if m:
        cam = m.group("cam")
        date_str = m.group("date")
        h = int(m.group("h"))
        mi = int(m.group("m"))
        s = int(m.group("s"))
        dt_utc = datetime.datetime(
            *[int(x) for x in date_str.split("-")],
            h, mi, s,
            tzinfo=datetime.timezone.utc,
        )
        return cam, date_str, dt_utc

    m = pattern2.match(name)
    if m:
        cam = m.group("cam")
        day = int(m.group("d"))
        month = int(m.group("m"))
        hour = int(m.group("h"))
        minute = int(m.group("mi"))

        year = 2025
        date_str = f"{year}-{month:02d}-{day:02d}"

        dt_utc = datetime.datetime(
            year, month, day, hour, minute, 0,
            tzinfo=datetime.timezone.utc,
        )
        return cam, date_str, dt_utc

    return None


def is_day(dt_utc: datetime.datetime) -> str:
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(CITY_TZ)
    except Exception:
        tz = datetime.timezone(datetime.timedelta(hours=2))
    dt_local = dt_utc.astimezone(tz)

    try:
        from astral import LocationInfo
        from astral.sun import sun
        helsinki = LocationInfo("Helsinki", "Finland", CITY_TZ, 60.1699, 24.9384)
        s = sun(helsinki.observer, date=dt_local.date(), tzinfo=tz)
        return "day" if s["sunrise"] <= dt_local <= s["sunset"] else "night"
    except Exception:
        hour = dt_local.hour
        return "day" if 9 <= hour <= 15 else "night"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cams = defaultdict(lambda: defaultdict(list))

    for p in IMG_DIR.iterdir():
        if not p.is_file():
            continue
        info = parse_name(p.name)
        if not info:
            continue
        cam, date_str, dt_utc = info
        day_flag = is_day(dt_utc)
        time_str = dt_utc.strftime("%H-%M-%S")
        key = (date_str, day_flag, time_str)
        cams[cam][key].append(p)

    if not cams:
        print("No valid images.")
        return

    cam_list = list(cams.keys())
    quota_per_cam = max(1, SAMPLE_SIZE // len(cam_list))

    sampled = []

    for cam in cam_list:
        layers = []
        for key, files in cams[cam].items():
            fs = list(files)
            random.shuffle(fs)
            layers.append((key, deque(fs)))
        random.shuffle(layers)

        picked = 0
        while layers and picked < quota_per_cam:
            new_layers = []
            for key, dq in layers:
                if picked >= quota_per_cam:
                    break
                if dq:
                    sampled.append(dq.popleft())
                    picked += 1
                if dq:
                    new_layers.append((key, dq))
            layers = new_layers

    if len(sampled) < SAMPLE_SIZE:
        all_files = []
        for cam in cams.values():
            for files in cam.values():
                all_files.extend(files)
        remain = [f for f in all_files if f not in sampled]
        random.shuffle(remain)
        need = SAMPLE_SIZE - len(sampled)
        sampled.extend(remain[:need])

    sampled = sampled[:SAMPLE_SIZE]

    for src in sampled:
        dst = OUT_DIR / src.name
        shutil.copy2(src, dst)

    print(f"cams: {len(cams)}")
    print(f"sampled: {len(sampled)}")
    print(f"saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
