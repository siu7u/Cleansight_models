#!/usr/bin/env python3
"""
从 Label Studio 服务器下载导出 JSON 引用的原始视频到 raw/videos/。

JSON 只存路径引用(data.video),视频本体在服务器。已存在且非空的文件会 [skip]。
下载后做完整性抽查:优先 ffprobe 读时长/帧数;没有 ffprobe 时退化为"大小 > 0"。

用法(在 yolo_pipeline/ 下执行):
    export LS_HOST=http://<LS地址>:8080
    export LS_TOKEN=<AccessToken>     # LS 页面 Account & Settings -> Access Token
    python3 01_pull_data.py
"""
import os
import shutil
import subprocess
import sys
import urllib.request

from utils.common import load_config
from utils import lsexport

LS_HOST = os.environ.get("LS_HOST", "").rstrip("/")
LS_TOKEN = os.environ.get("LS_TOKEN", "")


def probe(path) -> str:
    """返回一行完整性信息;损坏返回以 'BAD' 开头的串。"""
    if shutil.which("ffprobe"):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration:stream=nb_read_packets", "-of", "default=nw=1",
                 "-select_streams", "v:0", "-count_packets", str(path)],
                capture_output=True, text=True, timeout=120,
            )
            info = out.stdout.replace("\n", " ").strip()
            return info or "BAD 无视频流"
        except Exception as e:  # noqa: BLE001
            return f"BAD ffprobe 失败: {e}"
    size = path.stat().st_size
    return f"size={size/1e6:.1f}MB" if size > 0 else "BAD 0 字节"


def main():
    if not LS_HOST or not LS_TOKEN:
        sys.exit("请先设置环境变量 LS_HOST 和 LS_TOKEN(见脚本头部说明)")

    load_config()  # 目前不需要具体项,仅确认配置可读
    json_path = lsexport.latest_export()
    lsexport.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    tasks = lsexport.load_tasks(json_path)
    print(f"导出: {json_path.name}  共 {len(tasks)} 个 task -> {lsexport.VIDEO_DIR}")

    ok, skip, fail, bad = 0, 0, 0, []
    for t in tasks:
        rel = t.get("data", {}).get("video")
        if not rel:
            continue
        name = os.path.basename(rel)
        out = lsexport.VIDEO_DIR / name
        if out.exists() and out.stat().st_size > 0:
            print(f"  [skip] {name} 已存在")
            skip += 1
            continue
        url = f"{LS_HOST}{rel}"
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Token {LS_TOKEN}"})
            with urllib.request.urlopen(req, timeout=180) as r, open(out, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            info = probe(out)
            flag = "  ⚠️损坏" if info.startswith("BAD") else ""
            if info.startswith("BAD"):
                bad.append(name)
            print(f"  [ok]   {name}  {info}{flag}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [fail] {url}\n         {e}")
            fail += 1

    print(f"\n完成: 成功 {ok} / 跳过 {skip} / 失败 {fail}")
    if bad:
        print(f"⚠️ 疑似损坏 {len(bad)} 个,建议删掉重下: {', '.join(bad)}")
    if fail:
        print("失败多半是 LS_HOST/LS_TOKEN 不对,或视频接的是云存储(去云端原始位置取)")
    print("下一步:python3 00_status.py 看增量待办")


if __name__ == "__main__":
    main()
