#!/usr/bin/env python3
"""
一个适合入门的 B 站网页端爬虫示例。

它抓取的是网页端“综合热门”页面背后的公开数据接口，并把结果写入 CSV。
这样你可以先学会爬虫最核心的 4 件事：
1. 发送请求
2. 带请求头伪装浏览器
3. 解析返回数据
4. 保存成本地文件

运行示例:
    python bilibili_spider.py
    python bilibili_spider.py --pages 2 --page-size 10
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://api.bilibili.com/x/web-interface/popular"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/",
}


def fetch_hot_videos(page: int, page_size: int) -> list[dict[str, Any]]:
    """抓取一页热门视频数据。"""
    params = {"pn": page, "ps": page_size}
    request_url = f"{API_URL}?{urlencode(params)}"
    request = Request(request_url, headers=DEFAULT_HEADERS)

    try:
        with urlopen(request, timeout=15) as response:
            raw_text = response.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(f"HTTP 错误: {exc.code} {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"网络错误: {exc.reason}") from exc

    data = json.loads(raw_text)
    if data.get("code") != 0:
        raise RuntimeError(
            f"接口返回失败: code={data.get('code')} message={data.get('message')}"
        )

    return data.get("data", {}).get("list", [])


def clean_text(value: str) -> str:
    """把标题里的换行和多余空格清掉，方便写 CSV。"""
    return " ".join((value or "").split())


def normalize_video(video: dict[str, Any]) -> dict[str, Any]:
    """把原始 JSON 整理成更适合初学者理解的字段。"""
    owner = video.get("owner") or {}
    stat = video.get("stat") or {}
    bvid = video.get("bvid") or ""

    return {
        "标题": clean_text(video.get("title", "")),
        "BV号": bvid,
        "UP主": owner.get("name", ""),
        "播放量": stat.get("view", 0),
        "弹幕量": stat.get("danmaku", 0),
        "点赞数": stat.get("like", 0),
        "评论数": stat.get("reply", 0),
        "收藏数": stat.get("favorite", 0),
        "分区": (video.get("tname") or ""),
        "视频链接": video.get("short_link_v2")
        or video.get("short_link")
        or (f"https://www.bilibili.com/video/{bvid}" if bvid else ""),
    }


def save_to_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """把结果保存为 CSV，utf-8-sig 方便 Windows Excel 打开。"""
    if not rows:
        raise RuntimeError("没有可写入的数据。")

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抓取 B 站网页端综合热门视频")
    parser.add_argument("--pages", type=int, default=1, help="抓取多少页，默认 1 页")
    parser.add_argument(
        "--page-size",
        type=int,
        default=20,
        help="每页抓几条，默认 20 条",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="每页请求之间休眠几秒，默认 1 秒",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bilibili_hot_videos.csv"),
        help="输出文件名，默认 bilibili_hot_videos.csv",
    )
    args = parser.parse_args()

    if args.pages < 1:
        parser.error("--pages 不能小于 1")
    if not 1 <= args.page_size <= 50:
        parser.error("--page-size 建议在 1 到 50 之间")
    if args.sleep < 0:
        parser.error("--sleep 不能小于 0")

    return args


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, Any]] = []

    print("开始抓取 B 站综合热门...")
    for page in range(1, args.pages + 1):
        print(f"正在抓第 {page} 页")
        videos = fetch_hot_videos(page=page, page_size=args.page_size)

        if not videos:
            print("这一页没有数据，提前结束。")
            break

        all_rows.extend(normalize_video(video) for video in videos)
        time.sleep(args.sleep)

    save_to_csv(all_rows, args.output)
    print(f"抓取完成，共 {len(all_rows)} 条，已保存到: {args.output.resolve()}")


if __name__ == "__main__":
    main()
