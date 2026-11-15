# -*- coding: utf-8 -*-
"""
GFW FourWings 数据拉取脚本（分块 + 断点续传）
用法:
  /opt/anaconda3/bin/python pull_gfw.py ais 2024-01-01 2024-07-01
  /opt/anaconda3/bin/python pull_gfw.py sar 2024-01-01 2024-07-01
按月分块查询，结果增量追加到 data/ais_raw.csv 和 data/sar_raw.csv。
token 从 AIS-SAR_10.26test_副本.ipynb 中自动读取，不写进任何新文件。
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


LA_LB = {
    "type": "Polygon",
    "coordinates": [[
        [-118.35, 33.65],
        [-118.15, 33.65],
        [-118.15, 33.75],
        [-118.35, 33.75],
        [-118.35, 33.65],
    ]]
}


def month_chunks(start: str, end: str):
    """把 [start, end) 切成自然月块，返回 (chunk_start, chunk_end) 列表，end 为开区间。"""
    chunks = []
    cur = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    while cur < stop:
        nxt = min(cur + pd.offsets.MonthBegin(1), stop)
        chunks.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt
    return chunks


async def query_chunk(client, kind, start, end, retries=3):
    fn = (client.fourwings.create_ais_presence_report if kind == "ais"
          else client.fourwings.create_sar_presence_report)
    kwargs = dict(
        spatial_resolution="HIGH",
        temporal_resolution="DAILY",
        start_date=start,
        end_date=end,
        spatial_aggregation=True,
        geojson=LA_LB,
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
    out_path = WORK / "data" / f"{kind}_raw.csv"
    out_path.parent.mkdir(exist_ok=True)

    done_chunks = set()
    if out_path.exists():
        existing = pd.read_csv(out_path, usecols=["query_start", "query_end"])
        done_chunks = set(zip(existing["query_start"], existing["query_end"]))

    chunks = month_chunks(start, end)
    print(f"{kind}: {len(chunks)} chunks from {start} to {end}", flush=True)
    for i, (cs, ce) in enumerate(chunks, 1):
        if (cs, ce) in done_chunks:
            print(f"  [{i}/{len(chunks)}] skip {cs}..{ce} (already saved)", flush=True)
            continue
        t0 = time.time()
        df = await query_chunk(client, kind, cs, ce)
        df = df.copy()
        df["query_start"] = cs
        df["query_end"] = ce
        header = not out_path.exists()
        df.to_csv(out_path, mode="a", index=False, header=header)
        print(f"  [{i}/{len(chunks)}] {cs}..{ce}: {df.shape[0]} rows, {time.time() - t0:.1f}s", flush=True)
        time.sleep(2)  # 温和限速

    print("DONE", kind, flush=True)


if __name__ == "__main__":
    kind_, start_, end_ = sys.argv[1], sys.argv[2], sys.argv[3]
    asyncio.run(main(kind_, start_, end_))
