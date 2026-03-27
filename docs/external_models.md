# 外部模型接入

当前仓库将 `Uni-Mol` 与 `uni-elf` 作为外部模型后端接入。

## 设计原则

- 当前仓库负责数据预处理、调用编排、结果汇总和评估对比。
- `Uni-Mol` 与 `uni-elf` 的源码和运行环境不在当前仓库维护。
- 当前仓库只保存调用配置、命令模板、输入输出工件和文档说明。

## uni-elf

- 外部项目目录：`/root/software/uni-elf`
- 独立 conda 环境：`/root/software/uni-elf/.conda-env`
- 环境初始化脚本：`scripts/setup_uni_elf_env.sh`

根据外部项目自带说明，安装顺序应为：

1. 先安装 `lib/unicore`
2. 再安装 `unielf`

当前脚本也会补充代理和基础依赖：

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

## Uni-Mol

当前仓库通过外部 runner 生成 `Uni-Mol` 训练脚本和命令清单，不直接维护 `Uni-Mol` 源码。

默认会在外部环境中生成：

- `artifacts/external_models/unimol/run_unimol.py`
- `artifacts/external_models/unimol/train_manifest.json`

后续只需要把 bootstrap 脚本替换成真实 `unimol_tools` 训练 API 调用即可。

## CLI

准备外部模型工件：

```bash
spflow prepare-external-models artifacts/hsp_dry_run/results/hsp_workflow_results.csv --output-dir artifacts/external_models
```

准备 `uni-elf` 推理清单：

```bash
spflow prepare-unielf-inference valid.csv model.pt --config-path train_config.yaml --scaler-path scaler.pkl --output-dir artifacts/external_models
```
