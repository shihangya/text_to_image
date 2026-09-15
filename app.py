"""
SenseNova U1.5 Lite 图像生成 Web 应用后端

基于 SenseNova_U1.5_Lite_SDK.py 的 API 接口封装，
提供文生图与图片编辑的 Web API 服务。

v3.0.0 更新:
  - 异步 HTTP (httpx) 支持并发请求
  - UUID 历史记录 ID (防碰撞)
  - 原子文件写入 (防损坏)
  - 缩略图生成 (Pillow, 节省浏览器内存)
  - 批量生成 (n=1~4)
  - 负面提示词
  - 路径安全校验
  - 结构化日志
"""

import base64
import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from PIL import Image

# ==================== 日志 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ==================== 配置常量（与 SDK 一致） ====================
BASE_URL = "https://token.sensenova.cn/v1/images"
MODEL_ID = "sensenova-u1.5-lite"

RESOLUTIONS = {
    "2048x2048": "2K 1:1",
    "2720x1536": "2K 16:9",
    "1536x2720": "2K 9:16",
    "1664x2496": "2K 2:3",
    "2496x1664": "2K 3:2",
    "4096x4096": "4K 1:1",
}

TIMEOUT = 120

# 配置文件路径（与 app.py 同目录）
CONFIG_FILE = Path(__file__).parent / ".config.json"
# 默认图片保存目录（与 app.py 同目录的 output/）
DEFAULT_SAVE_DIR = str(Path(__file__).parent / "output")
# 手绘风格提示词库（低风险接入：仅做数据读取与提示词拼装）
STYLE_LIBRARY_PATH = Path(__file__).parent / "data" / "styles.json"
STYLE_IMAGE_ROOT = Path(__file__).parent / "handraw-style" / "images" / "individual"
STYLE_IMAGE_BUCKETS = {
    "001-200": Path("001-200"),
    "201-400": Path("201-400"),
}

# 图片格式 → MIME 类型
FORMAT_MIME = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}

# ==================== 数据模型 ====================

class GenerateRequest(BaseModel):
    api_key: Optional[str] = Field(None, description="SenseNova API Key（为空时使用已保存的 Key）")
    prompt: str = Field(..., min_length=1, description="图像描述文本")
    negative_prompt: Optional[str] = Field(None, max_length=2000, description="负面提示词")
    size: str = Field("2720x1536", description="图像尺寸 WxH")
    output_format: str = Field("png", description="图片格式 png/jpeg/webp")
    n: int = Field(1, ge=1, le=4, description="生成图片数量 (1-4)")
    watermark: bool = Field(False, description="是否添加水印")
    prompt_extend: bool = Field(True, description="是否开启提示词自动润色")


class EditRequest(BaseModel):
    api_key: Optional[str] = Field(None, description="SenseNova API Key（为空时使用已保存的 Key）")
    image_url: str = Field(..., description="公网URL 或 Base64 Data-URL")
    prompt: str = Field(..., min_length=1, description="编辑指令")
    negative_prompt: Optional[str] = Field(None, max_length=2000, description="负面提示词")
    size: str = Field("auto", description="输出尺寸，auto 自动适配")
    watermark: bool = Field(False, description="是否添加水印")
    prompt_extend: bool = Field(True, description="是否开启提示词自动润色")


class ConfigRequest(BaseModel):
    api_key: Optional[str] = None
    save_dir: Optional[str] = None


class StylePromptRequest(BaseModel):
    style_number: str = Field(..., min_length=3, max_length=3, description="风格编号，例如 041")
    theme: str = Field(..., min_length=1, max_length=2000, description="画面主题")
    ratio: Optional[str] = Field(None, max_length=100, description="可选画幅")
    subject: Optional[str] = Field(None, max_length=2000, description="可选主体限制")
    text: Optional[str] = Field(None, max_length=2000, description="可选文字要求")


# ==================== 异步 HTTP 客户端 ====================
async_client: Optional[httpx.AsyncClient] = None


# ==================== 工具函数 ====================

def _build_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


async def _extract_images(result: dict) -> list:
    """从 API 响应中提取图片数据，统一转为 base64 字符串"""
    images = []
    for item in result.get("data", []):
        if "b64_json" in item:
            images.append(item["b64_json"])
        elif "url" in item:
            try:
                resp = await async_client.get(item["url"], timeout=60)
                resp.raise_for_status()
                images.append(base64.b64encode(resp.content).decode())
            except Exception as e:
                logger.warning("Failed to download image from URL: %s", e)
                images.append(item["url"])
    return images


def _format_api_error(resp: httpx.Response) -> str:
    """格式化 API 错误信息"""
    try:
        body = resp.json()
        if isinstance(body, dict):
            msg = body.get("error", {}).get("message", "") or body.get("message", "") or str(body)
            return f"[{resp.status_code}] {msg}"
    except Exception:
        pass
    return f"[{resp.status_code}] {resp.text[:500]}"


# ==================== 手绘风格提示词库 ====================

STYLE_LIBRARY_CACHE: Optional[List[dict]] = None


def _load_style_library() -> List[dict]:
    """读取手绘风格风格库并做轻量缓存"""
    global STYLE_LIBRARY_CACHE
    if STYLE_LIBRARY_CACHE is not None:
        return STYLE_LIBRARY_CACHE

    try:
        data = json.loads(STYLE_LIBRARY_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load style library: %s", e)
        raise HTTPException(status_code=500, detail="风格库加载失败")

    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail="风格库格式不正确")

    STYLE_LIBRARY_CACHE = data
    return data


def _normalize_style_number(value: str) -> str:
    """规范风格编号为 001 格式"""
    text = str(value).strip()
    if not text.isdigit():
        raise HTTPException(status_code=400, detail="风格编号必须是 001-261")
    num = int(text)
    if not 1 <= num <= 261:
        raise HTTPException(status_code=400, detail="风格编号必须在 001-261 范围内")
    return f"{num:03}"


def _style_bucket_for_number(number: str) -> str:
    """根据风格编号选择参考图桶目录"""
    num = int(number)
    if 1 <= num <= 200:
        return "001-200"
    return "201-400"


def _style_image_path(number: str) -> Optional[Path]:
    """返回风格编号对应的参考图文件路径；不存在时返回 None"""
    bucket = _style_bucket_for_number(number)
    base = STYLE_IMAGE_ROOT
    if not base.exists():
        return None
    candidate = base / bucket / f"{number}.png"
    return candidate if candidate.exists() else None


def _filter_styles(
    styles: List[dict],
    q: Optional[str] = None,
    group: Optional[str] = None,
    limit: int = 300,
) -> List[dict]:
    """按关键词 / 分组 / 编号筛选风格"""
    if q:
        q_lower = q.strip().lower()
    else:
        q_lower = ""

    results = []
    for style in styles:
        if group and style.get("group") != group:
            continue

        if q_lower:
            haystack = " ".join([
                style.get("number", ""),
                style.get("group", ""),
                style.get("reference", ""),
                style.get("generation_name", ""),
                style.get("traits", ""),
            ]).lower()
            if q_lower not in haystack:
                continue

        results.append(style)
        if len(results) >= limit:
            break
    return results


def _build_style_prompt(style: dict, theme: str, ratio: Optional[str], subject: Optional[str], text: Optional[str]) -> dict:
    """按风格库模板生成双语提示词"""
    number = style["number"]
    generation_name = style["generation_name"]
    reference = style["reference"]
    theme = theme.strip()

    extra_zh_parts = []
    extra_en_parts = []
    if ratio:
        extra_zh_parts.append(f"画幅：{ratio}")
        extra_en_parts.append(f"aspect ratio: {ratio}")
    if subject:
        extra_zh_parts.append(f"主体限制：{subject}")
        extra_en_parts.append(f"subject constraints: {subject}")
    if text:
        extra_zh_parts.append(f"文字要求：{text}")
        extra_en_parts.append(f"text requirement: {text}")

    extra_zh = "；".join(extra_zh_parts)
    extra_en = "; ".join(extra_en_parts)

    zh_prompt = f"风格名称：#{number} · {generation_name}。参考作者/风格名称：{reference}。主题：{theme}。"
    if extra_zh:
        zh_prompt += f"；{extra_zh}。"

    en_prompt = f"Style name: #{number} · {generation_name}. Reference author/style name: {reference}. Theme: {theme}."
    if extra_en:
        en_prompt += f" {extra_en}."

    return {
        "number": number,
        "group": style.get("group", ""),
        "reference": reference,
        "generation_name": generation_name,
        "traits": style.get("traits", ""),
        "prompt_zh": zh_prompt,
        "prompt_en": en_prompt,
        "prompt": f"{zh_prompt}\n{en_prompt}",
    }


# ==================== 配置管理 ====================

def _load_config() -> dict:
    """从 .config.json 读取配置"""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to load config: %s", e)
    return {}


def _save_config(config: dict):
    """将配置写入 .config.json (原子写入)"""
    tmp = CONFIG_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(CONFIG_FILE))


def _get_api_key(req_key: Optional[str] = None) -> str:
    """获取 API Key：优先使用请求中传入的，否则使用已保存的"""
    if req_key and req_key.strip():
        return req_key.strip()
    config = _load_config()
    key = config.get("api_key", "").strip()
    if not key:
        raise HTTPException(
            status_code=400,
            detail="未提供 API Key，且未保存 API Key。请先在页面上保存 API Key。"
        )
    return key


def _get_save_dir() -> str:
    """获取图片保存目录"""
    config = _load_config()
    return config.get("save_dir", DEFAULT_SAVE_DIR)


def _ensure_save_dir() -> str:
    """确保保存目录存在，返回绝对路径字符串"""
    save_dir = _get_save_dir()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    return str(Path(save_dir).resolve())


def _safe_filepath(save_dir: str, filename: str) -> Path:
    """安全地构建文件路径，防止目录穿越"""
    filepath = (Path(save_dir) / filename).resolve()
    base = Path(save_dir).resolve()
    if not str(filepath).startswith(str(base)):
        raise HTTPException(status_code=403, detail="非法文件名")
    return filepath


# ==================== 历史管理 ====================

def _get_history_file() -> Path:
    """获取历史文件路径（与图片保存在同一目录）"""
    save_dir = _ensure_save_dir()
    return Path(save_dir) / "history.json"


def _load_history() -> list:
    """读取历史记录"""
    hf = _get_history_file()
    if hf.exists():
        try:
            data = json.loads(hf.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning("Failed to load history: %s", e)
    return []


def _save_history(history: list):
    """写入历史记录 (原子写入)"""
    hf = _get_history_file()
    tmp = hf.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(hf))


def _add_to_history(entry: dict):
    """添加一条历史记录（新记录插到最前面）"""
    history = _load_history()
    history.insert(0, entry)
    # 限制最多保留 200 条
    _save_history(history[:200])


def _delete_from_history(hid: str) -> dict:
    """删除一条历史记录，返回被删除的条目"""
    history = _load_history()
    new_history = []
    deleted = None
    for item in history:
        if item.get("id") == hid:
            deleted = item
        else:
            new_history.append(item)
    _save_history(new_history)
    return deleted or {}


# ==================== 图片保存与缩略图 ====================

def _save_image_to_disk(b64_data: str, prefix: str, fmt: str, custom_suffix: str = "") -> str:
    """
    将 base64 图片保存到磁盘，返回文件名。
    b64_data 为纯 base64 字符串（不含 data URL 前缀）。
    使用原子写入防止文件损坏。
    """
    save_dir = _ensure_save_dir()
    img_bytes = base64.b64decode(b64_data)

    if custom_suffix:
        filename = f"{prefix}_{custom_suffix}.{fmt}"
    else:
        timestamp = int(time.time())
        unique = uuid.uuid4().hex[:8]
        filename = f"{prefix}_{timestamp}_{unique}.{fmt}"

    filepath = Path(save_dir) / filename
    tmp_filepath = filepath.with_name(filepath.name + ".tmp")
    tmp_filepath.write_bytes(img_bytes)
    os.replace(str(tmp_filepath), str(filepath))
    return filename


def _generate_thumbnail(b64_data: str, original_filename: str) -> str:
    """
    为保存的图片生成 200x200 缩略图（JPEG），
    用于历史记录栏预览，大幅节省带宽和浏览器内存。
    """
    img_bytes = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(img_bytes))
    img.thumbnail((200, 200))

    thumb_dir = Path(_ensure_save_dir()) / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    stem = original_filename.rsplit(".", 1)[0]
    thumb_filename = f"thumb_{stem}.jpg"
    thumb_path = thumb_dir / thumb_filename

    # 合成白色背景以处理 RGBA/LA 透明通道
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        alpha = img.split()[-1]
        bg.paste(img.convert("RGB"), mask=alpha)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    img.save(thumb_path, format="JPEG", quality=80)
    return thumb_filename


def _thumbnail_url(original_filename: str) -> str:
    """构建缩略图 URL"""
    stem = original_filename.rsplit(".", 1)[0]
    return f"/thumbnails/thumb_{stem}.jpg"


def _delete_thumbnail(save_dir: str, filename: str):
    """删除对应缩略图"""
    stem = filename.rsplit(".", 1)[0]
    thumb_fp = Path(save_dir) / "thumbnails" / f"thumb_{stem}.jpg"
    if thumb_fp.exists():
        thumb_fp.unlink()


# ==================== FastAPI 应用 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global async_client
    async_client = httpx.AsyncClient(
        timeout=httpx.Timeout(TIMEOUT, connect=10.0),
        follow_redirects=True,
    )
    logger.info("HTTP client initialized")
    yield
    if async_client:
        await async_client.aclose()
        logger.info("HTTP client closed")


app = FastAPI(
    title="SenseNova U1.5 Lite 图像生成",
    description="基于 SenseNova U1.5 Lite 的文生图与图片编辑 Web API",
    version="3.0.0",
    lifespan=lifespan,
)


# ==================== 配置接口 ====================

@app.get("/api/resolutions", tags=["配置"])
def get_resolutions():
    """获取可选分辨率列表"""
    return list(RESOLUTIONS.values())


@app.get("/api/config", tags=["配置"])
def get_config():
    """获取已保存的配置（API Key + 保存目录）"""
    config = _load_config()
    key = config.get("api_key", "")
    save_dir = config.get("save_dir", DEFAULT_SAVE_DIR)
    result = {"save_dir": save_dir}
    if key:
        masked = key[:4] + "****" + key[-4:] if len(key) > 8 else "****"
        result.update({"saved": True, "masked_key": masked, "api_key": key})
    else:
        result["saved"] = False
    return result


@app.post("/api/config", tags=["配置"])
def save_config(req: ConfigRequest):
    """保存配置（API Key / 保存目录）"""
    config = _load_config()
    changed = False

    if req.api_key is not None:
        key = req.api_key.strip()
        if key:
            config["api_key"] = key
            changed = True

    if req.save_dir is not None:
        save_dir = req.save_dir.strip()
        if save_dir:
            try:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                config["save_dir"] = save_dir
                changed = True
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"无法创建目录: {str(e)}")

    if changed:
        _save_config(config)

    # 返回当前状态
    key = config.get("api_key", "")
    save_dir = config.get("save_dir", DEFAULT_SAVE_DIR)
    result = {"save_dir": save_dir}
    if key:
        masked = key[:4] + "****" + key[-4:] if len(key) > 8 else "****"
        result.update({"saved": True, "masked_key": masked, "api_key": key})
    else:
        result["saved"] = False
    return result


@app.delete("/api/config", tags=["配置"])
def delete_config():
    """清除已保存的 API Key（保留保存目录配置）"""
    config = _load_config()
    config.pop("api_key", None)
    _save_config(config)
    return {"saved": False, "message": "API Key 已清除"}


@app.get("/api/styles", tags=["配置"])
def list_styles(
    q: Optional[str] = None,
    group: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
):
    """获取手绘风格列表，支持分页、关键词和分组过滤"""
    if page <= 0:
        page = 1
    if page_size <= 0 or page_size > 100:
        raise HTTPException(status_code=400, detail="page_size 必须在 1-100 之间")
    styles = _load_style_library()
    filtered = _filter_styles(styles, q=q, group=group, limit=len(styles))
    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    items = filtered[start:end]
    for item in items:
        number = item.get("number", "")
        item["image_url"] = f"/style-images/{number}.png" if number else ""
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if total else 0,
    }


@app.post("/api/prompt-style", tags=["配置"])
def prompt_style(req: StylePromptRequest):
    """根据手绘风格编号与主题生成双语提示词"""
    styles = _load_style_library()
    number = _normalize_style_number(req.style_number)
    style = next((item for item in styles if item.get("number") == number), None)
    if not style:
        raise HTTPException(status_code=404, detail="风格编号不存在")
    return _build_style_prompt(
        style=style,
        theme=req.theme,
        ratio=req.ratio,
        subject=req.subject,
        text=req.text,
    )


@app.post("/api/test-key", tags=["配置"])
async def test_api_key(req: ConfigRequest):
    """
    测试 API Key 是否可用。
    注意: SenseNova API 无轻量验证端点，此方法会生成一张小图验证。
    """
    payload = {
        "model": MODEL_ID,
        "prompt": "测试",
        "n": 1,
        "size": "2048x2048",
        "response_format": "b64_json",
        "watermark": False,
        "prompt_extend": False,
    }
    try:
        resp = await async_client.post(
            f"{BASE_URL}/generations",
            json=payload,
            headers=_build_headers(_get_api_key(req.api_key)),
            timeout=TIMEOUT,
        )
    except httpx.RequestException as e:
        raise HTTPException(status_code=502, detail=f"网络请求失败: {str(e)}")

    if resp.status_code == 200:
        return {"valid": True, "message": "API Key 有效"}
    return {"valid": False, "message": _format_api_error(resp)}


# ==================== 生成 / 编辑 ====================

@app.post("/api/generate", tags=["生成"])
async def generate(req: GenerateRequest):
    """
    文生图接口。通过多次调用生成接口实现批量生成 (n=1~4)。
    生成后自动保存到磁盘（原子写入）、生成缩略图并记录历史。
    """
    api_key = _get_api_key(req.api_key)
    base_payload = {
        "model": MODEL_ID,
        "prompt": req.prompt,
        "n": 1,
        "size": req.size,
        "output_format": req.output_format,
        "response_format": "b64_json",
        "watermark": req.watermark,
        "prompt_extend": req.prompt_extend,
    }
    if req.negative_prompt:
        base_payload["negative_prompt"] = req.negative_prompt

    url = f"{BASE_URL}/generations"
    t0 = time.time()

    all_images = []
    filenames = []
    image_urls = []

    for _ in range(req.n):
        try:
            resp = await async_client.post(
                url, json=base_payload,
                headers=_build_headers(api_key),
                timeout=TIMEOUT,
            )
        except httpx.RequestException as e:
            raise HTTPException(status_code=502, detail=f"网络请求失败: {str(e)}")

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=_format_api_error(resp))

        result = resp.json()
        images = await _extract_images(result)
        all_images.extend(images)

        uid = uuid.uuid4().hex[:8]
        timestamp = int(time.time())
        for i, b64_data in enumerate(images):
            is_b64 = not b64_data.startswith("http") and not b64_data.startswith("data:")
            if is_b64:
                idx_suffix = f"_{len(filenames)+1}"
                custom = f"{timestamp}_{uid}{idx_suffix}"
                try:
                    filename = _save_image_to_disk(b64_data, "gen", req.output_format, custom)
                    image_urls.append(f"/images/{filename}")
                    filenames.append(filename)

                    try:
                        _generate_thumbnail(b64_data, filename)
                    except Exception as e:
                        logger.warning("Thumbnail failed for %s: %s", filename, e)

                    history_id = uuid.uuid4().hex[:16]
                    _add_to_history({
                        "id": history_id,
                        "type": "generate",
                        "prompt": req.prompt,
                        "negative_prompt": req.negative_prompt,
                        "size": req.size,
                        "format": req.output_format,
                        "filename": filename,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "elapsed": round(time.time() - t0, 2),
                    })
                except Exception as e:
                    logger.warning("Image save failed (index %d): %s", len(filenames), e)

    elapsed = round(time.time() - t0, 2)
    return {
        "images": all_images,
        "image_urls": image_urls,
        "filenames": filenames,
        "elapsed": elapsed,
        "saved": len(filenames) > 0,
    }


@app.post("/api/edit", tags=["编辑"])
async def edit_image(req: EditRequest):
    """
    图片编辑接口。支持负面提示词。
    编辑后自动保存到磁盘（原子写入）、生成缩略图并记录历史。
    """
    payload = {
        "model": MODEL_ID,
        "images": [{"image_url": req.image_url}],
        "prompt": req.prompt,
        "n": 1,
        "size": req.size,
        "response_format": "b64_json",
        "watermark": req.watermark,
        "prompt_extend": req.prompt_extend,
    }
    if req.negative_prompt:
        payload["negative_prompt"] = req.negative_prompt

    url = f"{BASE_URL}/edits"
    t0 = time.time()

    try:
        resp = await async_client.post(
            url, json=payload,
            headers=_build_headers(_get_api_key(req.api_key)),
            timeout=TIMEOUT,
        )
    except httpx.RequestException as e:
        raise HTTPException(status_code=502, detail=f"网络请求失败: {str(e)}")

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=_format_api_error(resp))

    result = resp.json()
    images = await _extract_images(result)
    elapsed = round(time.time() - t0, 2)

    filenames = []
    image_urls = []

    if images:
        uid = uuid.uuid4().hex[:8]
        timestamp = int(time.time())
        for i, b64_data in enumerate(images):
            is_b64 = not b64_data.startswith("http") and not b64_data.startswith("data:")
            if is_b64:
                idx_suffix = f"_{i+1}" if len(images) > 1 else ""
                custom = f"{timestamp}_{uid}{idx_suffix}"
                try:
                    filename = _save_image_to_disk(b64_data, "edit", "png", custom)
                    image_urls.append(f"/images/{filename}")
                    filenames.append(filename)

                    try:
                        _generate_thumbnail(b64_data, filename)
                    except Exception as e:
                        logger.warning("Thumbnail failed for %s: %s", filename, e)

                    history_id = uuid.uuid4().hex[:16]
                    _add_to_history({
                        "id": history_id,
                        "type": "edit",
                        "prompt": req.prompt,
                        "negative_prompt": req.negative_prompt,
                        "size": req.size,
                        "format": "png",
                        "filename": filename,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "elapsed": elapsed,
                    })
                except Exception as e:
                    logger.warning("Image save failed (index %d): %s", i, e)

    return {
        "images": images,
        "image_urls": image_urls,
        "filenames": filenames,
        "elapsed": elapsed,
        "saved": len(filenames) > 0,
    }


# ==================== 历史记录接口 ====================

@app.get("/api/history", tags=["历史"])
def get_history():
    """获取图片生成历史列表"""
    history = _load_history()
    save_dir = _ensure_save_dir()
    result = []
    for item in history:
        fn = item.get("filename", "")
        if fn and (Path(save_dir) / fn).exists():
            result.append({
                **item,
                "image_url": f"/images/{fn}",
                "thumbnail_url": _thumbnail_url(fn),
            })
    return {"items": result, "total": len(result)}


@app.delete("/api/history", tags=["历史"])
def clear_history():
    """清除全部历史记录（同时删除磁盘图片文件和缩略图）"""
    history = _load_history()
    save_dir = _ensure_save_dir()
    deleted_count = 0
    for item in history:
        fn = item.get("filename", "")
        if fn:
            fp = Path(save_dir) / fn
            if fp.exists():
                fp.unlink()
                deleted_count += 1
            _delete_thumbnail(save_dir, fn)
    _save_history([])
    return {"message": f"已清除 {len(history)} 条记录，删除 {deleted_count} 个文件"}


@app.delete("/api/history/{hid}", tags=["历史"])
def delete_history_item(hid: str):
    """删除单条历史记录（同时删除磁盘文件和缩略图）"""
    deleted = _delete_from_history(hid)
    if not deleted:
        raise HTTPException(status_code=404, detail="记录不存在")
    fn = deleted.get("filename", "")
    if fn:
        save_dir = _ensure_save_dir()
        fp = Path(save_dir) / fn
        if fp.exists():
            fp.unlink()
        _delete_thumbnail(save_dir, fn)
    return {"message": "已删除", "deleted": deleted}


# ==================== 静态图片服务 ====================

@app.get("/images/{filename}", tags=["图片"])
def serve_image(filename: str):
    """返回磁盘上的图片文件"""
    save_dir = _ensure_save_dir()
    filepath = _safe_filepath(save_dir, filename)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"图片不存在: {filename}")
    return FileResponse(str(filepath))


@app.get("/thumbnails/{filename}", tags=["图片"])
def serve_thumbnail(filename: str):
    """返回磁盘上的缩略图文件"""
    save_dir = Path(_ensure_save_dir()) / "thumbnails"
    filepath = (save_dir / filename).resolve()
    base = save_dir.resolve()
    if not str(filepath).startswith(str(base)):
        raise HTTPException(status_code=403, detail="非法文件名")
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"缩略图不存在: {filename}")
    return FileResponse(str(filepath))


@app.get("/style-images/{style_number}.png", tags=["图片"])
def serve_style_image(style_number: str):
    """返回风格库中对应编号的参考图"""
    number = _normalize_style_number(style_number)
    filepath = _style_image_path(number)
    if filepath is None:
        raise HTTPException(status_code=404, detail=f"风格参考图不存在: {number}")
    return FileResponse(str(filepath))


@app.get("/", response_class=HTMLResponse)
def index():
    """返回前端页面"""
    html_path = Path(__file__).parent / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
