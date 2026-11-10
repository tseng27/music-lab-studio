---
title: 音乐实验室工作站
emoji: 🎛️
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
python_version: "3.10"
app_file: app.py
pinned: false
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
- 片段检视：自动侦测 BPM 和调性、一键套用工程速度、拍点对齐格线
- 变调、变速（相位声码器，音高和速度互不影响）
- 音量自动化曲线
- 响度表（LUFS，依 ITU-R BS.1770）和一键母带：导出时调到 -14 或 -9 LUFS，峰值不超过 -1 dB

**后端**（`app.py` + `backend/engines.py`）用 Gradio 的 `Server` 提供上面的网页，同时提供 AI 接口。每个接口都走 Gradio 的排队机制；在 Hugging Face ZeroGPU 上，GPU 只在调用期间分配，用量记在调用者自己的额度上。

| 功能 | 接口 | 用到的模型 |
| --- | --- | --- |
| 分轨分离（人声/伴奏、四轨，或加上吉他和钢琴的六轨） | `/separate` | [Demucs](https://github.com/adefossez/demucs) `htdemucs`／`htdemucs_6s` |
| 自动对歌词 | `/align` | Whisper，经 [stable-ts](https://github.com/jianfch/stable-ts) 做强制对齐（可选择先分离人声） |
| 音频转 MIDI | `/transcribe` | Spotify [basic-pitch](https://github.com/spotify/basic-pitch) |
| AI 作曲 | `/generate` | 接上 [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) 后可以生成带人声的歌曲；没接时用 MusicGen 生成纯音乐 |
| 可用功能检查 | `/health` | |

前端通过 `@gradio/client` 调用这些接口（`frontend/vendor/` 里附带了一份，ISC 许可证）。其他程序也可以用 Python 的 `gradio_client` 调用：

```python
from gradio_client import Client, handle_file
c = Client("tseng27/music-lab-studio")
info, vocals, accomp, *_ = c.predict(handle_file("song.wav"), "vocals", api_name="/separate")
```

## 部署到 Hugging Face Space（免费 ZeroGPU）

1. 新建 Space：SDK 选 **Gradio**，范本选 **Blank**，硬件选 **ZeroGPU**。
2. 在 GitHub 仓库的 Settings → Secrets and variables → Actions 里添加 `HF_TOKEN`，值是 Hugging Face 的 Write 权杖。
3. 之后每次推送到 `main`，`.github/workflows/sync-to-hf.yml` 都会把代码同步到 Space，Space 会自动重新构建。

第一次构建要安装 PyTorch 等依赖，大约需要 10 到 20 分钟。第一次调用每个 AI 功能时还要下载模型，也会比较慢。

ZeroGPU 的免费额度：没有登录的访客共享每天 2 分钟，免费账号每天 5 分钟，PRO 账号每天 40 分钟（[官方说明](https://huggingface.co/docs/hub/spaces-api-endpoints)）。在 Space 页面上登录后使用，就会算在你自己的额度上。

## 本地运行

```bash
pip install -r requirements-dev.txt
MUSICLAB_MOCK=1 python app.py      # 演示模式：不装模型，AI 功能返回合成的示例结果
# 打开 http://localhost:7860
```

如果要用真正的模型（需要 NVIDIA GPU，或者耐心等 CPU 慢慢跑）：`pip install -r requirements.txt`，然后执行 `python app.py`。

## 接上 ACE-Step（生成带人声的歌曲）

ACE-Step 1.5 需要独立的 GPU 机器。先按它的说明启动 API 服务（`python -m acestep.api_server`，默认端口 8001），再给本服务设置环境变量（在 Space 上是 Settings → Variables and secrets）：

```bash
ACESTEP_URL=https://ACE-Step服务的地址
ACESTEP_API_KEY=...      # 只有 ACE-Step 开启了 API 密钥时才需要
```

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MUSICLAB_MOCK` | 空 | 设为 `1` 时进入演示模式 |
| `WHISPER_MODEL` | `medium` | 对歌词时用的 Whisper 模型，中文建议用 `large-v3` |
| `DEMUCS_MODEL` | `htdemucs` | 可以换成 `htdemucs_ft`，质量更好但速度更慢 |
| `MUSICGEN_MODEL` | `facebook/musicgen-small` | 没接 ACE-Step 时用的生成模型 |

## 开发

```bash
pip install -r requirements-dev.txt
python -m pytest tests            # 在演示模式下启动服务，测试全部 API
python scripts/build_frontend.py  # 修改 frontend/studio.html 后，重新生成 frontend/index.html
```

`frontend/studio.html` 是网页的源码（不含 `<html>` 外壳，可以直接发布成 claude.ai Artifact）。`frontend/index.html` 由脚本生成，`app.py` 提供的就是这个文件。
