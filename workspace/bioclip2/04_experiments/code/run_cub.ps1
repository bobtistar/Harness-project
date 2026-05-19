# CUB-200-2011 데이터셋으로 BioCLIP2 hierarchical prompt 실험 실행 스크립트.
#
# 학부생 친화 사용법:
#   1. CUB_200_2011 디렉토리가 D:/project/harness/CUB_200_2011 에 압축 해제되어 있어야 한다.
#   2. 가상환경에서 requirements.txt 설치:
#         pip install -r requirements.txt
#   3. 이 스크립트를 실행:
#         .\run_cub.ps1
#
# 단계:
#   STEP 1: GBIF로 200종 taxonomy 자동 생성 (인터넷 필요, 약 2~5분, 캐시 후 즉시).
#   STEP 2: OpenCLIP ViT-B/32로 임베딩 추출 + 6 조건 실험 (CPU 1~3시간 / GPU 5~15분).
#   STEP 3: 결과는 ../results/cub200_openclip/ 에 JSON으로 저장.
#
# 빠른 테스트(약 5분, CPU):
#   .\run_cub.ps1 -MaxPerClass 5
#
# GPU 풀 실행:
#   .\run_cub.ps1 -Model openclip-vitb32 -BatchSize 128 -Amp
#
# BioCLIP2 (~6GB 다운로드 필요):
#   .\run_cub.ps1 -Model bioclip2 -BatchSize 32 -Amp

param(
    [string]$CubRoot = "D:/project/harness/CUB_200_2011",
    [string]$OutDir = "../results/cub200_openclip",
    [string]$Model = "openclip-vitb32",
    [int]$BatchSize = 64,
    [int]$NumWorkers = 0,
    [int]$NSeeds = 5,
    [int]$Seed = 42,
    [Nullable[int]]$MaxPerClass = $null,
    [switch]$Amp,
    [string]$Device = $null
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

# ----------------------------------------------------------------------------
# STEP 1 — taxonomy CSV 생성 (이미 있으면 스킵)
# ----------------------------------------------------------------------------
$TaxCsv = Join-Path $CubRoot "cub_taxonomy.csv"
$CacheJson = Join-Path $CubRoot "cub_gbif_cache.json"

if (Test-Path $TaxCsv) {
    Write-Host "[STEP 1] $TaxCsv 이미 존재 → 스킵" -ForegroundColor Green
} else {
    Write-Host "[STEP 1] GBIF로 taxonomy 생성 시작 (약 2~5분 소요)..." -ForegroundColor Cyan
    python cub200_build_taxonomy.py `
        --cub_root $CubRoot `
        --out_csv  $TaxCsv `
        --cache    $CacheJson
    if ($LASTEXITCODE -ne 0) { Write-Error "taxonomy 빌드 실패"; exit 1 }
}

# ----------------------------------------------------------------------------
# STEP 2 — run_experiment.py 호출
# ----------------------------------------------------------------------------
$ImgRoot = Join-Path $CubRoot "images"
$CacheEmb = Join-Path $OutDir ("img_emb_" + $Model.Replace("/", "_") + ".npz")

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$pyArgs = @(
    "run_experiment.py",
    "--csv",         $TaxCsv,
    "--image_root",  $ImgRoot,
    "--model",       $Model,
    "--seed",        $Seed,
    "--n_seeds",     $NSeeds,
    "--batch_size",  $BatchSize,
    "--num_workers", $NumWorkers,
    "--out",         $OutDir,
    "--cache_emb",   $CacheEmb
)
if ($Amp)             { $pyArgs += "--amp" }
if ($Device)          { $pyArgs += @("--device", $Device) }
if ($MaxPerClass)     { $pyArgs += @("--max_per_class", $MaxPerClass) }

Write-Host "[STEP 2] 실험 실행: python $($pyArgs -join ' ')" -ForegroundColor Cyan
python @pyArgs
if ($LASTEXITCODE -ne 0) { Write-Error "실험 실행 실패"; exit 1 }

# ----------------------------------------------------------------------------
# STEP 3 — 요약 출력
# ----------------------------------------------------------------------------
Write-Host ""
Write-Host "[STEP 3] 완료. 산출물:" -ForegroundColor Green
Get-ChildItem $OutDir -File | ForEach-Object { "  " + $_.FullName }
Write-Host ""
Write-Host "주요 JSON:" -ForegroundColor Yellow
Write-Host "  exp1_geometry.json       — RQ1 (hierarchical vs flat)"
Write-Host "  exp2_rank_levels.json    — RQ2 (rank-level + latent taxonomy probe)"
Write-Host "  exp3_counterfactuals.json — RQ3 (6 조건 ablation, C2 vs C3 핵심)"
