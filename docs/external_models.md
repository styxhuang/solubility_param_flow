# 外部模型接入

当前仓库将 `Uni-Mol` 与 `uni-elf` 作为外部模型后端接入。

## 设计原则

- 当前仓库负责数据预处理、调用编排、结果汇总和评估对比。
- `Uni-Mol` 与 `uni-elf` 的源码和运行环境不在当前仓库维护。
- 当前仓库会生成训练/推理脚本、输入工件、Bohrium 提交命令和 manifest。
- 外部模型任务统一通过 Bohrium 镜像 `registry.dp.tech/dptech/dp/native/prod-13375/unielf:v1.21.0-manual` 提交。
- `Uni-Mol` 与 `uni-elf` 分别维护自己的 Bohrium 机型与任务组配置。
- 高通量场景优先先建 `job_group`，再通过 `-g <job_group_id>` 提交任务。

## uni-elf

- 外部项目目录：`/root/software/uni-elf`
- 独立 conda 环境：`/root/software/uni-elf/.conda-env`
- 环境初始化脚本：`scripts/setup_uni_elf_env.sh`

根据外部项目自带说明，镜像内任务仍然遵循：

1. 先安装 `lib/unicore`
2. 再安装 `unielf`

当前仓库假定镜像内已经具备或可访问以下依赖：

- `torch`
- `joblib`
- `rdkit`
- `pyyaml`
- `addict`
- `tqdm`
- `matplotlib`
- `huggingface_hub`
- `seaborn`
- `numpy==1.22.4`
- `pandas==1.4.0`
- `scikit-learn==1.5.0`
- `unimol_tools`

默认 Bohrium 配置：

- 机型：`c4_m8_cpu`
- 默认不自动创建 `job_group`

## Uni-Mol

当前仓库通过外部 runner 生成 `Uni-Mol` 训练脚本和 Bohrium 提交命令，不直接维护 `Uni-Mol` 源码。

默认 Bohrium 配置：

- 机型：`c16_m62_1 * NVIDIA T4`
- 默认自动创建 `job_group`

默认会在工件目录中生成：

- `artifacts/external_models/unimol/dataset.csv`
- `artifacts/external_models/unimol/run_unimol.py`
- `artifacts/external_models/unimol/train_manifest.json`

当前脚本已经参考远端 `notebook.py` 与 `07_train_predict.py` 的模式：

- 使用 `unimol_tools.UniMolRepr` 提取分子 `cls_repr`
- 拼接 sigma/体积等标量描述符
- 使用 `PyTorch` 回归头做三目标训练与预测
- 输出 `cv_fold_metrics.csv`、`cv_summary.csv`、`prediction.csv`、`prediction_scatter.png`

## CLI

准备外部模型工件：

```bash
spflow prepare-external-models artifacts/hsp_dry_run/results/hsp_workflow_results.csv --output-dir artifacts/external_models
```

这条命令会输出两条可直接执行的 Bohrium 提交命令；若对应 backend 开启了 `job_group`，命令会先创建任务组再提交：

- `uni-elf` 训练提交命令
- `Uni-Mol` 训练提交命令

如果希望在生成工件后直接触发提交：

```bash
spflow prepare-external-models artifacts/hsp_dry_run/results/hsp_workflow_results.csv --output-dir artifacts/external_models --submit
```

准备 `uni-elf` 推理清单：

```bash
spflow prepare-unielf-inference valid.csv model.pt --config-path train_config.yaml --scaler-path scaler.pkl --output-dir artifacts/external_models
```

输出中同样会给出 Bohrium 推理提交命令。

如果要直接触发推理提交：

```bash
spflow prepare-unielf-inference valid.csv model.pt --config-path train_config.yaml --scaler-path scaler.pkl --output-dir artifacts/external_models --submit
```
