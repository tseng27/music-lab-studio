---
title: 音乐实验室工作站
sdk: docker
app_port: 7860
---

# 音乐实验室工作站

在浏览器里使用的多轨音乐工作室，后端带 AI 工具。

**前端**（`frontend/`）是一个单文件网页，不需要安装：

- 多轨时间线：音频轨和 MIDI 乐器轨，片段可以拖动、裁剪、分割、复制，支持淡入淡出和片段增益
- 钢琴卷帘，内置 6 种音色：电钢琴、拨弦、合成铺底、贝斯、主音合成、鼓机
- 混音台：推子、声像、静音/独奏、三段 EQ、压缩、混响和延迟发送、母带限幅
- 分析器：频谱、峰值、响度、调音器
- 歌词对齐：打点同步、LRC 导入导出、卡拉 OK 预览
- 录音、保存/打开工程（`.json`，音频一并存入）、导出 WAV 混音

**后端**（`backend/`）是一个 FastAPI 服务，负责提供上面的网页和下面这些 AI 接口：

| 功能 | 接口 | 用到的模型 |
| --- | --- | --- |
| 分轨分离（人声/伴奏，或四轨） | `POST /api/separate` | [Demucs](https://github.com/adefossez/demucs) `htdemucs` |
| 自动对歌词 | `POST /api/align` | Whisper，经 [stable-ts](https://github.com/jianfch/stable-ts) 做强制对齐（可选择先分离人声） |
| 音频转 MIDI | `POST /api/transcribe` | Spotify [basic-pitch](https://github.com/spotify/basic-pitch) |
| AI 作曲 | `POST /api/generate` | 接上 [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) 后可以生成带人声的歌曲；没接时用 MusicGen 生成纯音乐 |

所有任务都是异步的：提交后返回任务 `id`，再用 `GET /api/jobs/{id}` 查询进度，完成后从 `GET /api/files/{id}/{文件名}` 下载结果。

## 本地运行

```bash
pip install -r backend/requirements.txt
MUSICLAB_MOCK=1 uvicorn backend.app:app --port 7860   # 演示模式：不装模型，AI 功能返回合成的示例结果
# 打开 http://localhost:7860
```

如果要用真正的模型（建议使用 NVIDIA GPU）：

```bash
pip install -r backend/requirements-ai.txt
uvicorn backend.app:app --host 0.0.0.0 --port 7860
```

只装了其中一部分模型也可以运行，AI 工具面板会显示哪些功能可用。

## 部署到 Hugging Face Space

1. 新建一个 Space，SDK 选 **Docker**，硬件选一张 GPU（例如 T4 或 L4）。
2. 把这个仓库推送到 Space（本 README 顶部的 front matter 已经配置好了）。
3. 等它构建完成后打开 Space 网址，工作室和 AI 工具就都能用了。

## 接上 ACE-Step（生成带人声的歌曲）

先按 ACE-Step 1.5 的说明启动它的 API 服务（`python -m acestep.api_server`，默认端口 8001），再给本服务设置环境变量：

```bash
ACESTEP_URL=http://ACE-Step服务的地址:8001
ACESTEP_API_KEY=...      # 只有 ACE-Step 开启了 API 密钥时才需要
```

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MUSICLAB_MOCK` | 空 | 设为 `1` 时进入演示模式 |
| `WHISPER_MODEL` | `medium` | 对歌词时用的 Whisper 模型，中文建议用 `large-v3` |
| `DEMUCS_MODEL` | `htdemucs` | 可以换成 `htdemucs_ft`，质量更好但速度更慢 |
| `MUSICGEN_MODEL` | `facebook/musicgen-small` | 没接 ACE-Step 时用的生成模型 |
| `MUSICLAB_DATA` | `./data` | 上传文件和任务结果的存放位置，过期后会自动清理 |
| `MUSICLAB_JOB_TTL` | `21600` | 任务结果保留的秒数 |
| `MUSICLAB_MAX_UPLOAD_MB` | `200` | 单个上传文件的大小上限 |
| `MUSICLAB_CORS` | `*` | 允许跨域访问的来源 |

## 开发

```bash
pip install pytest
python -m pytest tests            # 在演示模式下测试全部 API
python scripts/build_frontend.py  # 修改 frontend/studio.html 后，重新生成 frontend/index.html
```

`frontend/studio.html` 是网页的源码（不含 `<html>` 外壳，可以直接发布成 claude.ai Artifact）。`frontend/index.html` 由脚本生成，后端提供的就是这个文件。
