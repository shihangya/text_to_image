import os
import sys
import json
import base64
import logging
import time
from typing import Optional, Literal, List
from pathlib import Path

import requests

# ==================== 配置区域 ====================
# 优先从环境变量读取，也可以在这里直接填写你的 API Key
SENSENOVA_API_KEY = os.environ.get("SENSENOVA_API_KEY", "YOUR_API_KEY_HERE")

BASE_URL = "https://token.sensenova.cn/v1/images"
MODEL_ID = "sensenova-u1.5-lite"

# 建议分辨率常量
RESOLUTIONS = {
    "2k_1:1": "2048x2048",
    "2k_16:9": "2720x1536",
    "2k_9:16": "1536x2720",
    "2k_2:3": "1664x2496",
    "2k_3:2": "2496x1664",
    "4k_1:1": "4096x4096",
}
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class SenseNovaImageClient:
    """SenseNova U1.5 Lite 图片生成与编辑客户端"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or SENSENOVA_API_KEY
        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            raise ValueError(
                "请设置 API Key！\n"
                "方式1: export SENSENOVA_API_KEY='your-key'\n"
                "方式2: 在初始化时传入 api_key 参数"
            )
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def _post(self, endpoint: str, payload: dict) -> dict:
        """统一请求方法，含错误处理"""
        url = f"{BASE_URL}/{endpoint}"
        logger.info(f"请求接口: {url}")
        logger.info(f"请求参数: {json.dumps(payload, ensure_ascii=False, indent=2)}")

        resp = self.session.post(url, json=payload, timeout=120)

        if resp.status_code != 200:
            logger.error(f"请求失败 [{resp.status_code}]: {resp.text}")
            resp.raise_for_status()

        result = resp.json()
        logger.info(f"请求成功，返回 {len(result.get('data', []))} 张图片")
        return result

    @staticmethod
    def _save_image(data_item: dict, output_dir: str = "./output", prefix: str = "img") -> str:
        """
        保存单张图片到本地
        支持 b64_json 和 url 两种返回格式
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if "b64_json" in data_item:
            img_bytes = base64.b64decode(data_item["b64_json"])
            ext = "png"  # b64_json 默认按 png 保存，可根据需要调整
        elif "url" in data_item:
            img_url = data_item["url"]
            logger.info(f"正在从 URL 下载图片: {img_url[:80]}...")
            img_resp = requests.get(img_url, timeout=60)
            img_resp.raise_for_status()
            img_bytes = img_resp.content
            # 从 URL 推断扩展名
            ext = img_url.split(".")[-1].split("?")[0] if "." in img_url else "png"
        else:
            raise ValueError(f"未知的返回格式: {data_item}")

        filename = f"{prefix}_{int(time.time())}.{ext}"
        filepath = out_path / filename
        filepath.write_bytes(img_bytes)
        logger.info(f"✅ 图片已保存: {filepath.absolute()} ({len(img_bytes) / 1024:.1f} KB)")
        return str(filepath.absolute())

    def generate(
        self,
        prompt: str,
        size: str = "2720x1536",
        output_format: Literal["png", "jpeg", "webp"] = "png",
        response_format: Literal["b64_json", "url"] = "b64_json",
        watermark: bool = False,
        prompt_extend: bool = True,
        save_dir: str = "./output",
    ) -> List[str]:
        """
        文生图接口 /v1/images/generations

        Args:
            prompt: 图像描述文本
            size: 图像尺寸，参考 RESOLUTIONS 常量或直接传 "WxH"
            output_format: 图片文件格式 (png/jpeg/webp)
            response_format: 返回方式 (b64_json/url)
            watermark: 是否添加水印（False=无水印，公测期免费）
            prompt_extend: 是否开启提示词自动润色
            save_dir: 图片保存目录

        Returns:
            保存后的本地文件路径列表
        """
        payload = {
            "model": MODEL_ID,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "output_format": output_format,
            "response_format": response_format,
            "watermark": watermark,
            "prompt_extend": prompt_extend,
        }

        result = self._post("generations", payload)
        saved_paths = []
        for item in result.get("data", []):
            path = self._save_image(item, save_dir, prefix="gen")
            saved_paths.append(path)
        return saved_paths

    def edit(
        self,
        image_url: str,
        prompt: str,
        size: str = "auto",
        response_format: Literal["b64_json", "url"] = "b64_json",
        watermark: bool = False,
        prompt_extend: bool = True,
        save_dir: str = "./output",
    ) -> List[str]:
        """
        图片编辑接口 /v1/images/edits

        Args:
            image_url: 公网图片URL 或 Base64 Data-URL (data:image/png;base64,...)
                       ⚠️ 不支持纯无前缀 Base64 字符串
            prompt: 编辑指令
            size: 输出尺寸，"auto" 自动适配主图
            response_format: 返回方式
            watermark: 是否添加水印
            prompt_extend: 是否开启提示词自动润色
            save_dir: 图片保存目录

        Returns:
            保存后的本地文件路径列表
        """
        payload = {
            "model": MODEL_ID,
            "images": [{"image_url": image_url}],
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": response_format,
            "watermark": watermark,
            "prompt_extend": prompt_extend,
        }

        result = self._post("edits", payload)
        saved_paths = []
        for item in result.get("data", []):
            path = self._save_image(item, save_dir, prefix="edit")
            saved_paths.append(path)
        return saved_paths


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 方式1: 设置环境变量 SENSENOVA_API_KEY
    # 方式2: client = SenseNovaImageClient(api_key="sk-xxxxxxxx")
    client = SenseNovaImageClient()

    # --- 示例1: 文生图 ---
    print("\n🎨 === 文生图示例 ===")
    gen_paths = client.generate(
        prompt="一个裹着塑料布的女性模特躺在沙发上，柔和晨光，写实摄影风格",
        size=RESOLUTIONS["2k_1:1"],
        output_format="png",
        response_format="b64_json",
        watermark=False,       # 公测期免费去水印
        prompt_extend=True,
    )
    print(f"生成结果: {gen_paths}")

    # --- 示例2: 图片编辑（使用公网URL）---
    # print("\n✏️ === 图片编辑示例 ===")
    # edit_paths = client.edit(
    #     image_url="https://example.com/source.png",
    #     prompt="把背景改成雪山，人物保持不变",
    #     size="auto",
    #     response_format="url",
    #     watermark=False,
    # )
    # print(f"编辑结果: {edit_paths}")

    # --- 示例3: 图片编辑（使用本地文件转 Base64 Data-URL）---
    # import base64 as b64
    # local_img = Path("D:\H3C_demo\H3C\小脚本\output\gen_1789010192.png").read_bytes()
    # data_url = f"data:image/png;base64,{b64.b64encode(local_img).decode()}"
    # edit_paths = client.edit(
    #     image_url=data_url,
    #     prompt="将背景替换为纯白色",
    #     watermark=False,
    # )
    # print(f"编辑结果: {edit_paths}")