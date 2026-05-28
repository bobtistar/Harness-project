# BioCLIP2 Hierarchical Prompt Experiments

이 디렉토리는 이미지 임베딩에 텍스트 프롬프트 정보를 결합했을 때, 계층 분류 정보가 embedding geometry에 어떤 변화를 만드는지 확인하는 실험 코드이다. 메인 실행 파일은 `run_experiment.py`이고, 현재 실험 조건은 `C0`부터 `C4`까지다.

## 실행 순서

`python run_experiment.py ...`를 실행하면 아래 순서로 진행된다.

1. `argparse`로 모델, 데이터, seed, 출력 경로, cache, 시각화 옵션을 읽는다.
2. `_load_env_file()`이 현재 디렉토리와 상위 디렉토리에서 `.env`를 찾고, `_configure_hf_token()`이 `HF_TOKEN` 또는 CLI token을 환경변수에 등록한다.
3. `set_global_seeds()`가 `numpy`, `torch`, CUDA seed와 cuDNN deterministic 옵션을 고정한다.
4. 데이터 입력을 만든다.
   - `--csv`가 있으면 `load_real_dataset()`이 CSV를 읽고 species별 label index, 이미지 경로, taxonomy table을 만든다.
   - `--csv`가 없으면 `get_toy_dataset(samples_per_species=6)`으로 8종 x 6샘플 toy dataset을 만든다.
5. 이미지 임베딩을 만든다.
   - `--cache_emb`가 있고 shape/label이 맞으면 `.npz`에서 `img_emb`를 재사용한다.
   - `--mock`이면 `mock_image_embeddings(labels, dim=128, intra_noise=0.45)`로 class-conditioned embedding을 만든다.
   - 실제 모델이면 `load_model()`로 OpenCLIP/BioCLIP 계열 모델을 로드한 뒤 `encode_images()`로 이미지 또는 toy tensor를 배치 인코딩한다.
6. 실험을 순서대로 실행한다.
   - Exp1: `C0` flat prompt와 `C1` hierarchy prompt 비교
   - Exp2: kingdom부터 species까지 rank별 label로 C0/C1 geometry 비교
   - Exp3: `C0`~`C4` counterfactual prompt ablation
7. 결과를 JSON으로 저장하고, `--visualize`가 켜져 있으면 UMAP 또는 t-SNE PNG를 추가 저장한다.
8. 마지막에 git hash와 주요 라이브러리 버전을 `run_log.txt`에 남긴다.

## 주요 Python 파일

### `run_experiment.py`

실험 전체를 조립하는 진입점이다.

- `condition_embeddings()`는 조건명 `C0`~`C4`를 받아 `generate_prompts()`로 prompt list를 만들고, mock 모드에서는 `mock_text_embeddings()`, 실제 모델 모드에서는 `encode_texts()`로 텍스트 임베딩을 만든다.
- `fuse_image_text()`는 이미지와 텍스트 임베딩을 각각 L2 normalize한 뒤 concat한다. 텍스트가 없는 특수 fallback이 아니면 최종 feature 차원은 image dim + text dim이다.
- `geometric_metrics()`는 하나의 조건 embedding `Z`에 대해 intra variance, inter margin, silhouette, RankMe, uniformity, kNN purity, linear probe accuracy를 계산한다.
- `experiment_1()`은 seed별로 `1e-3` Gaussian noise를 이미지 임베딩에 더한 뒤 C0/C1을 반복 측정한다. 마지막 seed의 embedding은 시각화용으로 `_vis_embeddings`에만 보관하고 JSON 저장 전에 `pop()`한다.
- `experiment_2()`는 `RANKS = [kingdom, ..., species]` 각 rank마다 sample의 taxonomy label을 integer label로 바꾸고 C0/C1 metric을 계산한다. 별도로 species label을 permutation한 silhouette 분포와 실제 silhouette을 비교해 latent taxonomy probe를 저장한다.
- `experiment_3()`은 `C0, C1, C2, C3, C4`를 같은 방식으로 측정하고, `C1 - C0` 대비 `C2/C3/C4 - C0`의 preservation ratio를 bootstrap CI와 함께 계산한다.

### `prompt_variants.py`

프롬프트 조건과 taxonomy record를 정의한다.

- `TaxonomyRecord`는 `kingdom, phylum, cls, order, family, genus, species`를 가진다. Python 예약어 때문에 class rank는 `cls` 필드에 저장한다.
- `C0`: `"a photo of {species}"`
- `C1`: `"a photo of {kingdom} {phylum} {class} {order} {family} {genus} {species}"`
- `C2`: 상위 6개 rank를 `taxK taxP taxC taxO taxF taxG` placeholder로 치환하고 species만 유지한다.
- `C3`: species는 유지하고 상위 rank label은 다른 record에서 rank별로 랜덤 샘플링한다.
- `C4`: 같은 record의 7개 taxonomy label을 무작위 순서로 섞어 word-bag prompt를 만든다.

### `data_loader.py`

실험 입력 데이터를 같은 형태로 맞춘다.

- `get_toy_dataset()`은 Aves, Insecta, Plantae, Fungi에 걸친 8개 `TaxonomyRecord`를 만들고, species index label을 `np.repeat()`로 생성한다.
- `get_toy_images()`는 실제 사진 대신 class별 RGB 평균이 다른 synthetic tensor를 만들어 실제 이미지 인코더 호출을 테스트할 수 있게 한다.
- `load_real_dataset()`은 `file, kingdom, phylum, class, order, family, genus, species` CSV를 읽고 species별 sample 수가 `min_samples_per_species` 이상인 항목만 남긴다. 반환값은 unique records, sample label array, image path list, species-to-taxonomy table이다.

### `extract_embeddings.py`

모델 로드와 embedding 추출을 담당한다.

- `MODEL_REGISTRY`는 `openclip-vitb32`, `openclip-vitl14`, `bioclip`, `bioclip2` 키를 실제 OpenCLIP 또는 HuggingFace checkpoint로 매핑한다.
- `load_model()`은 OpenCLIP 계열이면 `open_clip.create_model_and_transforms()`, BioCLIP 계열이면 `open_clip.create_model_from_pretrained("hf-hub:...")`를 사용한다.
- `encode_images()`는 PIL image list, path list, 또는 `(N,3,H,W)` tensor를 받아 batch 단위로 `model.encode_image()`를 호출한다. path 입력이면 `_PathImageDataset`과 DataLoader를 사용한다.
- `encode_texts()`는 tokenizer로 prompt batch를 token화하고 `model.encode_text()`를 호출한다.
- `mock_image_embeddings()`와 `mock_text_embeddings()`는 모델 없이도 label/prompt 기반 deterministic vector를 만들어 파이프라인을 검증한다.

### `metrics.py`

embedding geometry와 통계 검정을 모은 파일이다.

- geometry: `intra_class_variance`, `inter_class_margin`, `silhouette_cosine`, `rankme`, `uniformity`, `knn_purity_at_k`
- downstream probe: `linear_probe_accuracy()`가 stratified train/test split 후 `LogisticRegression(max_iter=1000)` accuracy를 반환한다.
- hierarchy 보조 지표: `lca_depth_mean`, `hierarchy_distance`, `mutual_information_cluster_rank`
- 통계: `paired_permutation_test()`, `bootstrap_ci()`, `cohens_d_paired()`
- 시각화: `plot_embeddings()`는 조건 하나를 UMAP/t-SNE로 저장하고, `plot_comparison_grid()`는 여러 조건을 subplot grid로 비교한다.

### 데이터 준비/요약 스크립트

- `cub200_build_taxonomy.py`: CUB-200의 `classes.txt`를 읽고 GBIF 조회 및 fallback rule로 `cub_taxonomy.csv`를 만든다.
- `prepare_rare_species.py`: HuggingFace `imageomics/rare-species` dataset을 이미지 파일과 taxonomy CSV로 export한다. 기존 export가 완전하면 재사용한다.
- `summarize_rare_species_results.py`: rare species 결과 폴더에서 Exp1/Exp2/Exp3 JSON을 읽고 모델별 markdown 비교표를 출력한다.
- `models/baseline_flat.py`: C0 flat prompt list를 만드는 얇은 helper다.

## 출력 파일

기본 출력 디렉토리는 `../results`이고 `--out`으로 바꿀 수 있다.

```text
exp1_geometry.json          # C0 vs C1 평균/표준편차, permutation test, 성공 기준
exp2_rank_levels.json       # rank별 C0/C1 metric, latent taxonomy probe
exp3_counterfactuals.json   # C2/C3/C4 preservation ratio와 CI
run_log.txt                 # args, overwrite 경고, git hash, 라이브러리 버전
vis_*.png                   # --visualize 사용 시 조건별/비교 grid 이미지
```

## 빠른 실행

```bash
pip install -r requirements.txt

python run_experiment.py --mock --toy --seed 42 --out ../results/mock

python run_experiment.py \
  --mock --toy --seed 42 \
  --visualize --vis_method tsne \
  --out ../results/mock_vis
```

실제 이미지 CSV를 사용할 때는 다음 column이 필요하다.

```csv
file,kingdom,phylum,class,order,family,genus,species
img_0001.jpg,Animalia,Chordata,Aves,Passeriformes,Passeridae,Passer,Passer domesticus
```

`file`은 `--image_root` 기준 상대 경로로 해석된다.
