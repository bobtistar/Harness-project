# Cloud GPU 실행 가이드 (iNat21 sweep용)

> Aves(CUB-200) 1-domain 실험을 multi-kingdom (Aves/Insecta/Plantae/Fungi/Reptilia)
> 으로 확장하는 데 필요한 클라우드 GPU 셋업·실행·비용 가이드.
> Windows 로컬에는 CUDA 환경이 없으므로 대부분의 사용자는 클라우드를 거치게 됨.

---

## 1) GPU 요구사항 / 데이터 크기 감각

| 작업 | 모델 | 권장 GPU | VRAM | 디스크 |
|---|---|---|---|---|
| BioCLIP2 (ViT-L/14) 인코딩 + 실험 | bioclip2 | RTX 4090 / A10 / A100 | ≥ 16GB | ~20GB |
| OpenCLIP ViT-L/14 (laion-2b) | openclip-vitl14 | 동일 | ≥ 14GB | ~3GB |
| iNat21 val (10K stratified sample) | — | — | — | val 이미지 **~8.4GB** + 캐시 |

**메모리 한계 (코드 측 메모리)**: `cosine_distance_matrix` 가 N²×4B 사용
- N=10,000 → 400MB (안전)
- N=30,000 → 3.6GB (borderline)
- N=100,000 (val 전체) → 40GB (불가) → **반드시 서브샘플**

`run_inat21.sh` 기본값(`MAX_SPECIES=200`, `MAX_PER_SPECIES=10`)이 5 kingdom × 200종 × 10장 = **10,000장**으로 안전 영역.

---

## 2) 추천 클라우드 옵션

| 제공자 | 장점 | GPU 시간당 가격 (대략, USD) | 추천도 |
|---|---|---|---|
| **RunPod** | Spot price, Docker pod, jupyter/ssh | RTX 4090 ~$0.4-0.6 / A100 40GB ~$1.5-2.0 | ★★★★★ |
| **Lambda Labs** | 안정적 (on-demand), 단순 SSH | A10 ~$0.75 / A100 40GB ~$1.3 | ★★★★ |
| **Vast.ai** | 가장 저렴 (개인 호스트), 가변 | RTX 4090 ~$0.3-0.4 | ★★★ (안정성 ↓) |
| **Modal** | serverless, 코드 함수형 | A10G ~$1.1 | ★★★ (스크립트 재작성 필요) |
| **AWS/GCP** | 엔터프라이즈 | A100 ~$3-4 | ★★ (비쌈, IAM 등 진입장벽) |

**처음 한 번이라면 RunPod 추천** — 카드만 등록하면 30초 안에 GPU pod 띄움, 종량제 청구.

---

## 3) 예상 실행 시간과 비용 (RunPod RTX 4090 기준)

본 실험 (N=10K, 2 model sweep, 5 seeds, n_perm=1000):

| 단계 | BioCLIP2 | OpenCLIP ViT-L/14 |
|---|---|---|
| 모델 다운로드 (HF) | ~30s (cache miss 1회) | ~10s |
| 이미지 인코딩 10K | ~40s (AMP) | ~30s |
| Exp1 (5 seeds × 2 cond) | ~3 min | ~3 min |
| Exp2 (rank + latent probe) | ~2 min | ~2 min |
| Exp3 (5 seeds × 6 cond, GPU LCA) | ~4 min | ~4 min |
| **모델당 합계** | **~10 min** | **~9 min** |

**전체 sweep ≈ 20분, RTX 4090 시간당 $0.5 가정 시 ≈ $0.17.**

A100을 쓰면 더 빠르지만 본 워크로드는 GPU-bound가 아니라서 4090이 cost/perf 최적.

---

## 4) RunPod 셋업 (recipe)

### (1) Pod 생성

- **Template**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (또는 동등)
- **GPU**: RTX 4090 1장 (또는 A10 / A100)
- **Container disk**: 30GB
- **Volume**: 50GB persistent (선택, 데이터 보존용)
- **포트**: 22 (ssh) + 8888 (jupyter, 선택)

### (2) Pod 부팅 후 SSH 접속

```bash
ssh -p <PORT> root@<POD_IP> -i <ssh_key>
```

### (3) 코드 + 의존성 설치

```bash
# 1) 리포지토리 클론 (또는 scp 로 직접 업로드)
cd /workspace
git clone <your-repo-url> harness
cd harness/workspace/bioclip2/04_experiments/code

# 2) 의존성
pip install -r requirements.txt
pip install open_clip_torch transformers huggingface_hub pillow tqdm pygbif

# 3) 동작 확인 (mock toy)
python run_experiment.py --mock --toy --out /tmp/mock
```

### (4) iNat21 val 다운로드

iNat21 공식 소스: https://github.com/visipedia/inat_comp/tree/master/2021

```bash
mkdir -p /workspace/inat21 && cd /workspace/inat21

# val 이미지 (~8.4GB tar) + val.json
# (URL은 visipedia 문서 참조; 직접 링크는 변경 가능성 있어 README 확인 권장)
wget -c https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz
wget -c https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz

# 압축 풀기 (~10분)
tar -xzf val.tar.gz   # → val/<dir>/*.jpg
tar -xzf val.json.tar.gz   # → val.json
rm val.tar.gz val.json.tar.gz   # 디스크 정리

ls /workspace/inat21
# 기대: val/  val.json
```

> **iNat21-mini를 train으로 쓰고 싶다면** 추가로 `train_mini.tar.gz`(~42GB) + `train_mini.json.tar.gz`도 받을 수 있음. 단 본 frozen-evaluation 파이프라인은 val만 있으면 충분.

### (5) HuggingFace 토큰 (필요 시)

BioCLIP2 가중치는 `imageomics/bioclip-2` 로 공개라 토큰 불필요하지만, 만약 rate limit이 걸리면:

```bash
huggingface-cli login   # HF 토큰 입력
# 또는
export HF_TOKEN=hf_xxxxx
```

### (6) 실험 sweep 실행

```bash
cd /workspace/harness/workspace/bioclip2/04_experiments/code

INAT_ROOT=/workspace/inat21 \
MODELS="bioclip2,openclip-vitl14" \
BATCH_SIZE=64 \
N_SEEDS=5 \
AMP=1 \
bash run_inat21.sh
```

진행 로그가 stdout으로 흐름. 산출물:
```
../results/inat21_Aves-Insecta-Plantae-Fungi-Reptilia_bioclip2/
    exp1_geometry.json
    exp2_rank_levels.json
    exp3_counterfactuals.json
    img_emb_bioclip2.npz
    run_log.txt

../results/inat21_Aves-Insecta-Plantae-Fungi-Reptilia_openclip-vitl14/
    (동일 구조)
```

### (7) 결과 다운로드

```bash
# 로컬에서
scp -P <PORT> -i <ssh_key> -r \
    root@<POD_IP>:/workspace/harness/workspace/bioclip2/04_experiments/results \
    ./local_results/
```

embedding 캐시(`img_emb_*.npz`)는 ~30MB/모델이라 같이 받아도 부담 없음 → 후속 분석 재실행 시 GPU 없이도 metric 재계산 가능.

### (8) Pod 종료

```bash
# RunPod 대시보드에서 Stop / Terminate.
# Volume을 만들었으면 다음 실행 때 그대로 attach 가능 (데이터 재다운로드 불필요).
```

---

## 5) Lambda Labs / Vast.ai 차이점

**Lambda Labs**:
- Web에서 instance launch → SSH 접속
- Pre-install: PyTorch, conda, jupyter
- 위 (3)-(7) 단계 그대로 적용. apt-get 사용 가능.

**Vast.ai**:
- 호스트마다 image 선택 다름 → `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` 같은 표준 image 권장
- 가격 변동 큼 → "spot" 입찰 후 사용
- "Verified" 호스트만 필터링 권장 (안정성)

---

## 6) 비용 절감 팁

1. **Embedding 캐시 활용**: `img_emb_*.npz` 한 번 만들면 prompt 변경/metric 추가 시 GPU 없이 CPU로 재실행 가능 (코드의 `--cache_emb` 옵션).
2. **이미지 다운로드를 volume에 저장**: pod 매번 끄더라도 volume 살아있으면 8.4GB 재다운로드 불필요.
3. **2 모델 sweep을 1 pod에서 연속 실행**: HF cache가 공유되어 두 번째 모델 로드가 빠름.
4. **A100 대신 RTX 4090**: 본 워크로드는 메모리 16GB로 충분, 4090이 cost/perf 우위.
5. **N_SEEDS=3 로 한 번 점검 후 5로 본 실행**: 첫 sanity check를 작게 → bug 발견 시 비용 절감.

---

## 7) 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `RuntimeError: CUDA out of memory` | `BATCH_SIZE` 를 32 또는 16으로 낮춤. 또는 더 큰 GPU. |
| `cosine_distance_matrix` 에서 MemoryError | N (이미지 수)이 너무 큼 → `MAX_SPECIES` / `MAX_PER_SPECIES` 줄임. |
| `open_clip` 가 hf-hub 모델을 못 찾음 | `pip install -U open_clip_torch` (3.0+ 필요). |
| HuggingFace download stuck | `HF_HUB_DISABLE_PROGRESS_BARS=1` + 재시도. 또는 `huggingface-cli login`. |
| 이미지 경로 mismatch | iNat21 val.json의 `file_name` 이 `val/<dir>/x.jpg` 인데 image_root가 `inat21/` 가 아니라 `inat21/val/` 로 잘못 가리킨 경우. `IMAGE_ROOT=$INAT_ROOT` 로 두고 file_name을 그대로 join. |
| `pygbif` 없다고 에러 | iNat21에는 pygbif 불필요 (메타데이터 내장). CUB만 필요. |
| Windows 에서 `bash run_inat21.sh` 실패 | WSL 또는 클라우드에서 실행. `run_inat21.ps1` 필요하면 별도 작성. |

---

## 8) 다음 단계 (분석)

sweep 끝난 후 `results/` 에 모인 JSON들을 비교 분석하는 노트북 또는 스크립트 짜기:

```python
import json, pathlib
for d in pathlib.Path("../results").glob("inat21_*"):
    exp2 = json.loads((d / "exp2_rank_levels.json").read_text())
    z = exp2["latent_taxonomy_probe"]["z_score"]
    print(f"{d.name}  latent z = {z:.1f}")
```

기대하는 핵심 비교:
- **BioCLIP2의 latent probe z-score** vs **OpenCLIP의 z-score** — BioCLIP2가 *더 강해야* "BioCLIP2가 잠재 구조를 *증폭*" 주장 성립
- **Exp3 silhouette drop pattern (구조 > 순서 > 어휘)** 이 두 모델에서 *모두* 보이는지 — 보이면 모델 일반 성질, BioCLIP2만 보이면 BioCLIP2 특이성
- **kingdom별 (Aves/Insecta/Plantae/Fungi/Reptilia)** 효과 일관성 → RQ4의 random-effects meta-analysis 가능

이 결과들이 워크샵 페이퍼의 Figure 1-3을 채워줍니다.
