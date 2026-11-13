# Builds colab/직캠_스테이지_코랩.ipynb from colab/fancam.py (the notebook carries the code itself,
# so it runs without this repository). Run: python colab/build_notebook.py
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
code = open(os.path.join(here, 'fancam.py'), encoding='utf8').read()
server = open(os.path.join(here, 'server.py'), encoding='utf8').read()
body3d = open(os.path.join(here, 'body3d.py'), encoding='utf8').read()


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(True)}


def py(text, form=False):
    cell = {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': text.strip('\n').splitlines(True)}
    if form:
        cell['metadata']['cellView'] = 'form'
    return cell


cells = [
    md("""
# 직캠 스테이지 · 코랩으로 누끼 따기

직캠에서 설윤만 따서, 앱에 바로 넣을 수 있는 클립 파일(`이름.stage-tb.mp4`)을 만들어요.
코랩의 GPU로 돌려서 노트북 PC보다 훨씬 빠르고, 더 큰 모델(SAM 2.1 base+, RVM ResNet-50)을 써서 더 정확해요.

**쓰는 법**
1. 위 메뉴 **런타임 → 런타임 유형 변경**에서 GPU(T4, L4, A100 중 하나)를 골라요.
2. 아래 칸을 위에서부터 차례로 ▶ 눌러 실행해요. (**런타임 → 모두 실행**도 돼요.)
3. 다 되면 내 구글 드라이브의 **직캠 스테이지/코랩** 폴더에 파일이 생겨요.
   PC나 폰에서 받아서 앱의 **파일 가져오기**로 넣으면 끝이에요.

설윤 사진을 드라이브 폴더에 3~5장 넣어 두면 그걸로 찾고, 없으면 영상에서 가장 많이 나온 얼굴(= 직캠 주인공)로 찾아요.
"""),
    md('## 1. GPU 확인과 설치 (처음 한 번 2~3분)'),
    py("""
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "GPU가 없어요: 런타임 → 런타임 유형 변경 → T4 GPU"
import os
os.environ['SAM2_BUILD_CUDA'] = '0'  # skip the optional CUDA extension (slow to build, not needed)
!pip -q install sam2 yt-dlp onnxruntime-gpu ultralytics
"""),
    md('## 2. 구글 드라이브 연결'),
    py("""
from google.colab import drive
drive.mount('/content/drive')
"""),
    md('## 3. 설정\n\n오른쪽 칸을 채우고 ▶를 누르세요.'),
    py("""
VIDEO = 'https://youtu.be/tE33xh6wdEI'  #@param {type:"string"}
#@markdown ↑ 유튜브 링크, 또는 드라이브 안의 영상 경로 (예: `/content/drive/MyDrive/직캠/설윤.mp4`)
NAME = '설윤'  #@param {type:"string"}
PHOTOS_FOLDER = '/content/drive/MyDrive/직캠 스테이지/설윤 사진'  #@param {type:"string"}
#@markdown ↑ 혼자 나온 정면 사진 3~5장이 든 폴더 (없으면 비워 두세요)
START = 0  #@param {type:"number"}
END = 0  #@param {type:"number"}
#@markdown ↑ 초 단위, END가 0이면 끝까지
SHAPE = '누끼'  #@param ["누끼", "원"]
TRACKER = '빠름 (YOLO)'  #@param ["빠름 (YOLO)", "정확 (SAM 2)"]
#@markdown ↑ 원 모드에서만 써요
QUALITY = '고화질 (긴 변 1280)'  #@param ["보통 (긴 변 960)", "고화질 (긴 변 1280)", "FHD (긴 변 1920)"]
FPS = 30  #@param [24, 30, 60] {type:"raw"}
SAM_MODEL = 'auto'  #@param ["auto", "tiny", "small", "base_plus", "large"]
OUT_FOLDER = '/content/drive/MyDrive/직캠 스테이지/코랩'
""", form=True),
    md('## 4. 처리 코드'),
    py('%%writefile /content/fancam.py\n' + code),
    py('%%writefile /content/server.py\n' + server),
    py('%%writefile /content/body3d.py\n' + body3d),
    md("""
## 4-3D. 3D로 만들 때만 (SAM 3D Body)

3D(춤추는 3D 인형)를 만들 때만 실행하세요. 처음 한 번 3~5분 걸려요.

1. [Hugging Face](https://huggingface.co)에서 [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3) 사용 신청을 해서 승인받아요.
2. **Settings → Access Tokens**에서 Read 토큰을 만들어요.
3. 왼쪽 🔑(보안 비밀)에 이름 `HF_TOKEN`으로 넣거나, 이 칸을 실행할 때 나오는 입력창에 붙여 넣어요.
"""),
    py("""
!pip -q install pytorch-lightning pyrender yacs scikit-image einops timm dill pandas rich hydra-core pyrootutils webdataset networkx==3.2.1 roma joblib loguru optree fvcore xtcocotools jsonlines appdirs chump huggingface_hub
!pip -q install git+https://github.com/microsoft/MoGe.git
from huggingface_hub import login
try:
    from google.colab import userdata
    token = userdata.get('HF_TOKEN')
except Exception:
    token = None
if not token:
    from getpass import getpass
    token = getpass('Hugging Face 토큰: ')
login(token=token)
import sys
sys.path.insert(0, '/content')
import body3d
body3d.estimator()  # 코드와 모델을 받아 둬요 (승인이 안 됐으면 여기서 오류가 나요)
print('3D 준비 완료')
"""),
    md("""
## 5-A. 앱과 연결해서 쓰기 (추천)

이 칸을 실행하면 주소가 하나 나와요. 앱의 **만들기 → 코랩 GPU로 처리**에 붙여 넣으면,
앱에서 평소처럼 영상 고르기·멤버 고르기를 하고 **추출 시작**만 누르면 코랩 GPU가 처리해요. (위 3번 설정 칸은 안 써도 돼요.)
**이 칸은 계속 실행 중(⏹)으로 둬야 해요.** 끝나 있으면 코랩이 놀고 있다고 보고 런타임을 끊어요. 진행 상황이 여기 표시돼요.
코랩을 다시 켜면 주소가 바뀌어요.

**2번(구글 드라이브 연결)도 실행해 두면** 완성된 클립이 드라이브의 **직캠 스테이지/코랩** 폴더에도 저장돼요.
앱으로 받는 게 느리면 거기서 받아 **파일 가져오기**로 넣으면 훨씬 빨라요. (PC에 구글 드라이브 앱이 켜져 있으면 저절로 내려와요.)
"""),
    py("""
import sys
sys.path.insert(0, '/content')
import server
server.serve()
server.wait()  # 이 칸은 계속 실행 중으로 둬요 (끝나면 코랩이 런타임을 끊어요)
"""),
    md('## 5-B. 앱 없이 여기서 바로 실행\n\n3번 설정대로 처리해요. 처리하는 동안 이 탭을 열어 두세요.'),
    py("""
import glob, importlib, sys
sys.path.insert(0, '/content')
import fancam
importlib.reload(fancam)

video = VIDEO.strip()
if video.startswith('http'):
    video = fancam.download_youtube(video, '/content/videos')
    print('받은 영상:', video)
photos = sorted(p for p in glob.glob(os.path.join(PHOTOS_FOLDER, '*')) if p.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))) if PHOTOS_FOLDER else []
max_side = {'보통 (긴 변 960)': 960, '고화질 (긴 변 1280)': 1280, 'FHD (긴 변 1920)': 1920}[QUALITY]
out = fancam.make_clip(video, OUT_FOLDER, name=NAME, photos=photos, start=float(START), end=float(END) or None,
                       fps=int(FPS), max_side=max_side, shape='circle' if SHAPE == '원' else 'cutout', sam_model=SAM_MODEL,
                       tracker='yolo' if TRACKER.startswith('빠름') else 'sam')
"""),
    md("""
## 6. 앱에 넣기

드라이브의 **직캠 스테이지/코랩** 폴더에서 `이름.stage-….mp4` 파일을 받아, 앱 보관함의 **파일 가져오기**로 열면 돼요.
이름 끝의 `.stage-tb`/`.stage-sbs`/`.stage-circle`은 앱이 파일을 읽는 데 필요하니 바꾸지 마세요.
"""),
]

nb = {
    'nbformat': 4, 'nbformat_minor': 0,
    # No gpuType here: a fixed one would undo the GPU the user picked (A100, L4…) every time a
    # new copy of the notebook is uploaded. accelerator just asks for some GPU.
    'metadata': {'colab': {'provenance': []}, 'accelerator': 'GPU',
                 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'language_info': {'name': 'python'}},
    'cells': cells,
}
out = os.path.join(here, '직캠_스테이지_코랩.ipynb')
with open(out, 'w', encoding='utf8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('wrote', out)
