# -*- coding: utf-8 -*-
"""
GFW FourWings 数据拉取脚本：官方锚地版本（33 CFR 110.214，2025-09 修订）
F = Commercial Anchorage F (outside of Long Beach Breakwater)
G = Commercial Anchorage G (outside of the Middle Breakwater)
用法:
  /opt/anaconda3/bin/python pull_gfw_anchorages.py ais 2024-01-01 2025-01-01
  /opt/anaconda3/bin/python pull_gfw_anchorages.py sar 2024-01-01 2025-01-01
按月 × 按锚地分块查询，结果增量追加到 data/ais_anchorage_raw.csv / data/sar_anchorage_raw.csv。
坐标来源: https://www.ecfr.gov/current/title-33/chapter-I/subchapter-J/part-110/subpart-B/section-110.214
"""
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

WORK = Path(__file__).parent
TOKEN_FILE = WORK / ".gfw_token"
NB_COPY = WORK / "AIS-SAR_10.26test_副本.ipynb"
NB_ORIG = WORK / "AIS-SAR_10.26test.ipynb"


def load_token():
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    nb_path = NB_COPY if NB_COPY.exists() else NB_ORIG
    src = json.loads(nb_path.read_text())
    for cell in src["cells"]:
        text = "".join(cell.get("source", []))
        m = re.search(r'access_token="([^"]+)"', text)
        if m:
            return m.group(1)
    raise RuntimeError("未找到 GFW token")


def dms(deg, minutes, seconds, sign=1):
    return sign * (deg + minutes / 60 + seconds / 3600)


# 33 CFR 110.214(b)(6)(i) Commercial Anchorage F (outside of Long Beach Breakwater)
ANCHORAGE_F = [
    (dms(33, 43, 5.1), -dms(118, 8, 4.0)),
    (dms(33, 43, 5.0), -dms(118, 10, 32.5)),
    (dms(33, 42, 13.3), -dms(118, 9, 54.8)),
    (dms(33, 40, 51.3), -dms(118, 9, 32.2)),
    (dms(33, 38, 36.2), -dms(118, 7, 43.8)),
    (dms(33, 40, 44.4), -dms(118, 6, 51.4)),
]

# 33 CFR 110.214(b)(7)(i) Commercial Anchorage G (outside of the Middle Breakwater)
ANCHORAGE_G = [
    (dms(33, 43, 5.4), -dms(118, 11, 17.9)),
    (dms(33, 43, 5.4), -dms(118, 12, 18.6)),
    (dms(33, 42, 25.8), -dms(118, 14, 19.2)),
    (dms(33, 40, 50.4), -dms(118, 13, 1.2)),
    (dms(33, 41, 2.9), -dms(118, 12, 19.0)),
    (dms(33, 42, 10.8), -dms(118, 11, 36.0)),
]

ANCHORAGES = {"F": ANCHORAGE_F, "G": ANCHORAGE_G}


def poly(coords):
    # coords 为 (lat, lon) 元组列表，GeoJSON 要求 [lon, lat]
    ring = [[lo, la] for la, lo in coords]
    return {"type": "Polygon", "coordinates": [ring + [ring[0]]]}


def month_chunks(start: str, end: str):
    chunks = []
    cur = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    while cur < stop:
        nxt = min(cur + pd.offsets.MonthBegin(1), stop)
        chunks.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt
    return chunks


async def query_chunk(client, kind, start, end, geojson, retries=3):
    fn = (client.fourwings.create_ais_presence_report if kind == "ais"
          else client.fourwings.create_sar_presence_report)
    kwargs = dict(
        spatial_resolution="HIGH",
        temporal_resolution="DAILY",
        start_date=start,
        end_date=end,
        spatial_aggregation=True,
        geojson=geojson,
    )
    if kind == "ais":
        kwargs["group_by"] = "VESSEL_ID"
    last_err = None
    for attempt in range(retries):
        try:
            result = await fn(**kwargs)
            return result.df()
        except Exception as e:
            last_err = e
            wait = 10 * (attempt + 1)
            print(f"     attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:200]}; retry in {wait}s", flush=True)
            await asyncio.sleep(wait)
    raise last_err


async def main(kind, start, end):
    import gfwapiclient as gfw

    client = gfw.Client(access_token=load_token())
    out_path = WORK / "data" / f"{kind}_anchorage_raw.csv"
    out_path.parent.mkdir(exist_ok=True)

    done = set()
    if out_path.exists():
        existing = pd.read_csv(out_path, usecols=["anchorage", "query_start", "query_end"])
        done = set(zip(existing["anchorage"], existing["query_start"], existing["query_end"]))

    chunks = month_chunks(start, end)
    total = len(chunks) * len(ANCHORAGES)
    print(f"{kind}: {total} chunks ({len(ANCHORAGES)} anchorages x {len(chunks)} months)", flush=True)
    i = 0
    for name, coords in ANCHORAGES.items():
        gj = poly(coords)
        for cs, ce in chunks:
            i += 1
            if (name, cs, ce) in done:
                print(f"  [{i}/{total}] skip anchorage {name} {cs}..{ce}", flush=True)
                continue
            t0 = time.time()
            df = await query_chunk(client, kind, cs, ce, gj)
            df = df.copy()
            df["anchorage"] = name
            df["query_start"] = cs
            df["query_end"] = ce
            header = not out_path.exists()
            df.to_csv(out_path, mode="a", index=False, header=header)
            print(f"  [{i}/{total}] anchorage {name} {cs}..{ce}: {df.shape[0]} rows, {time.time() - t0:.1f}s", flush=True)
            time.sleep(1.5)

    print("DONE", kind, flush=True)


if __name__ == "__main__":
    kind_, start_, end_ = sys.argv[1], sys.argv[2], sys.argv[3]
    asyncio.run(main(kind_, start_, end_))
