#!/bin/bash
#PBS -q DBG
#PBS --group=<グループ名>
#PBS -l elapstim_req=00:10:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
#PBS -N dt_smoke
#PBS -r n
#
# 環境が壊れていないことを DBG キュー（最大10分）で確認する。
# 本番ジョブを投入する前に必ずこれを通すこと。GPU はポイント消費が CPU の約6倍で、
# 環境不備で落ちるジョブは丸ごと無駄になる。
#
# 投入: qsub jobs/smoke.sh

cd "${PBS_O_WORKDIR}"
source jobs/_common.sh

echo "=== host GPU / driver ==="
nvidia-smi

echo "=== container: versions and CUDA availability ==="
run_gpu python -c "
import torch, numpy, pandas, sklearn, scipy
print('python  ', __import__('sys').version.split()[0])
print('torch   ', torch.__version__)
print('numpy   ', numpy.__version__)
print('pandas  ', pandas.__version__)
print('sklearn ', sklearn.__version__)
print('scipy   ', scipy.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    # 実際に計算を流して、初期化だけでなく演算が通ることを確認する
    x = torch.randn(1024, 1024, device='cuda')
    print('matmul ok:', float((x @ x).sum()))
"

echo "=== model smoke test (UNet1D backbone) ==="
run_gpu python src/models/DDPM_Aggregate/model.py --smoke
status_unet=$?

# Tang バックボーンは attention を全解像度に持つ別構造なので、UNet1D が通っても
# こちらが通るとは限らない。train_ddpm_tang.sh の本学習前にここで確認する。
echo "=== model smoke test (Tang backbone) ==="
run_gpu python src/models/DDPM_Aggregate_Tang/model.py --smoke
status_tang=$?

# DiT バックボーンは畳み込みを一切持たない attention のみの構造なので、
# U-Net 系2本が通ってもこちらが通るとは限らない。train_ddpm_dit.sh の本学習前に確認する。
echo "=== model smoke test (DiT backbone) ==="
run_gpu python src/models/DDPM_Aggregate_DiT/model.py --smoke
status_dit=$?

echo "exit status: unet1d=${status_unet} tang=${status_tang} dit=${status_dit}"
