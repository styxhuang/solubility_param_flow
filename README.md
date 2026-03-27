# 溶解度参数计算工作流

这是一个用于计算 Hansen 溶解度参数（HSP）并预测溶解度的工作流项目，结合了分子表示、量子化学工作流以及机器学习建模的思路。

## 功能特性

- **HSP 计算**：计算 `δD`、`δP`、`δH`
- **分子描述符**：从 `SMILES` 或结构信息计算化学描述符
- **溶解度预测**：使用机器学习模型进行溶解度预测
- **结果可视化**：绘制 HSP 球和溶解度分布图

## 安装

```bash
uv venv .venv
source .venv/bin/activate
uv sync --extra dev --extra md
```

仓库会提交 `uv.lock`，而本地 `.venv/` 会被 `.gitignore` 忽略。后续推荐优先使用：

```bash
uv sync --frozen --extra dev --extra md
```

如果只想直接使用命令，也可以不手动激活环境：

```bash
uv run spflow --help
```

## 基本用法

```python
from solubility_param_flow import HSPCalculator

calc = HSPCalculator()
hsp = calc.calculate_from_smiles("CCO")  # 乙醇
print(f"δD={hsp.delta_d:.2f}, δP={hsp.delta_p:.2f}, δH={hsp.delta_h:.2f}")
```

## HSP 计算流程

当前项目包含两条与 HSP 相关的流程：一条是轻量级本地原型流程，另一条是基于 ORCA 的结构驱动流程。

### 1. 本地原型流程

这是目前最简单的一条路径，主要服务于当前的 API 和 CLI：

```text
SMILES
  -> HSPCalculator.calculate_from_smiles()
  -> HSPParameters(δD, δP, δH)
  -> distance_to()
  -> predict_solubility()
```

步骤如下：

1. 输入分子的 `SMILES` 字符串。
2. 调用 `HSPCalculator.calculate_from_smiles()`。
3. 获取 `δD`、`δP`、`δH` 和 `δTotal`。
4. 通过 HSP 距离比较溶质和溶剂的相似性。
5. 将距离转换为一个简化的溶解度评分。

说明：这条流程目前仍是占位实现，主要用于流程验证、命令行演示和接口联调。

### 2. 基于 ORCA 的流程

这条路径面向更偏结构驱动的 HSP 估算：

```text
SMILES
  -> RDKit 生成三维结构
  -> XYZ 几何文件
  -> 生成 ORCA 输入文件
  -> 提交 Bohrium 任务
  -> ORCA 输出 / XYZ
  -> HSPCalculatorCOSMO.calculate_from_orca_output()
  -> HSPResult(δD, δP, δH)
  -> 溶剂比较 / 溶解度评分
```

步骤如下：

1. 从 `SMILES` 字符串开始。
2. 用 `SmilesToOrcaWorkflow.smiles_to_xyz()` 构建三维结构。
3. 用 `generate_orca_input()` 生成 ORCA 输入文件。
4. 用 `submit_to_bohrium()` 提交计算任务。
5. 任务完成后，读取生成的 ORCA 输出文件和对应的 XYZ 结构。
6. 调用 `HSPCalculatorCOSMO.calculate_from_orca_output()` 估算 HSP 值。
7. 将结果与 `COMMON_SOLVENTS` 中的常见溶剂进行比较，并排序得到溶剂匹配情况。

相关模块：

- `src/solubility_param_flow/core/hsp_calculator.py`
- `src/solubility_param_flow/core/hsp_cosmo.py`
- `src/solubility_param_flow/workflow/smiles_to_orca.py`
- `src/solubility_param_flow/workflow/hsp_workflow.py`

### 3. 典型 CLI 用法

快速计算一个原型 HSP 结果：

```bash
spflow calculate-hsp "CCO" --name ethanol
```

从 `SMILES` 提交一个 ORCA 任务：

```bash
spflow submit-orca "CCO" --name ethanol
```

查看 Bohrium 任务状态：

```bash
spflow monitor 123456
```

从 `CSV` 读取 `SMILES`，执行 dry-run 的 `ORCA/OpenCOSMO-RS` 流程编排，并生成模拟 `HSP` 与 `COSMO-RS` 描述符结果：

```bash
spflow prepare-dataset molecules.csv --output-dir artifacts/hsp_dry_run
```

这条命令当前不会真正调用本地 `ORCA` 或 `OpenCOSMO-RS`，而是用于先打通：

- `CSV` 读取与 `SMILES` 校验
- 三维结构与 `ORCA` 输入文件生成
- 结构优化、单点能、`COSMO` 三阶段命令编排
- `OpenCOSMO-RS` 结果位点和描述符落盘
- 汇总结果表输出

从 `CSV` 直接生成真实 `ORCA` 批任务目录（`r1/`, `r2/`, ...）与 `run.sh`：

```bash
spflow prepare-orca-jobs molecules.csv --output-dir artifacts/orca_jobs
```

这条命令会完成：

- 基于 `SMILES` 的三维构型生成
- `mol.xyz` 写出
- `job_opt.inp` / `job_sp.inp` 或 `job.inp` 写出
- 本地批量执行脚本 `run.sh` 生成

把已经生成好的 `r{i}` 任务目录批量提交到 Bohrium 镜像：

```bash
spflow submit-orca-jobs artifacts/orca_jobs --project-id 929872
```

如果希望直接从 `CSV` 生成并提交：

```bash
spflow submit-orca-csv molecules.csv --jobs-dir artifacts/orca_jobs --project-id 929872
```

任务结束后，先下载 Bohrium 输出，再把结果拷回本地 `r{i}` 目录：

```bash
spflow download-orca-results artifacts/orca_jobs/bohrium_submission_manifest.json --completed-only
spflow stage-orca-results artifacts/orca_jobs/bohrium_submission_manifest.json artifacts/orca_jobs/bohrium_downloads
```

在 `ORCA` 任务跑完后，把 `r{i}` 目录下的 `.cpcm` 回写到原始 `CSV` 的 `d/p/h/Vm` 列：

```bash
spflow post-orca molecules.csv artifacts/orca_jobs --backup
```

如果还要导出 sigma moments 与 profile 描述符：

```bash
spflow export-sigma artifacts/orca_jobs --csv-path molecules.csv --output-path artifacts/orca_jobs/descriptors.csv
```

使用 dry-run 结果表训练传统 `ML` 基线模型，并输出指标和对比图：

```bash
spflow train-ml artifacts/hsp_dry_run/results/hsp_workflow_results.csv --output-dir artifacts/ml_baseline
```

默认会比较三组特征：

- `rdkit`
- `cosmo`
- `combined`

并输出：

- `metrics_summary.csv`
- `predictions.csv`
- `feature_manifest.json`
- `model_comparison.png`
- `best_model_scatter.png`

准备外部 `Uni-Mol/uni-elf` 调用工件：

```bash
spflow prepare-external-models artifacts/hsp_dry_run/results/hsp_workflow_results.csv --output-dir artifacts/external_models
```

这条命令现在会生成：

- 上传到 Bohrium 的数据/配置工件
- `uni-elf` 训练 Python 脚本
- `Uni-Mol` 训练 Python 脚本
- 对应的 Bohrium 提交命令与 manifest
- `Uni-Mol` 默认会使用 `c16_m62_1 * NVIDIA T4`
- 开启 `job_group` 的 backend 会先建组再提交

外部模型统一提交到镜像：

```text
registry.dp.tech/dptech/dp/native/prod-13375/unielf:v1.21.0-manual
```

如果希望生成工件后立即提交：

```bash
spflow prepare-external-models artifacts/hsp_dry_run/results/hsp_workflow_results.csv --output-dir artifacts/external_models --submit
```

准备 `uni-elf` 推理清单：

```bash
spflow prepare-unielf-inference valid.csv model.pt --config-path artifacts/external_models/unielf/train_config.yaml --scaler-path scaler.pkl --output-dir artifacts/external_models
```

该命令同样会输出一个 Bohrium 推理提交命令。

如果要生成后直接提交：

```bash
spflow prepare-unielf-inference valid.csv model.pt --config-path artifacts/external_models/unielf/train_config.yaml --scaler-path scaler.pkl --output-dir artifacts/external_models --submit
```

初始化 `uni-elf` 独立环境：

```bash
bash scripts/setup_uni_elf_env.sh
```

### 4. 当前实现状态

- 本地 HSP 计算流程目前仍是占位实现。
- ORCA 提交流程已经可以生成输入文件并提交任务。
- 已新增 `CSV -> ORCA 任务目录 -> .cpcm 后处理 -> d/p/h/Vm 回写` 主链 CLI。
- 已新增 `ORCA 任务目录 -> Bohrium 镜像提交 -> 结果下载/回填` 这一层调度能力。
- 已将 `Uni-Mol/uni-elf` 工件生成改为 Bohrium Python 脚本提交模式。
- `Uni-Mol` Bohrium 默认机型已切到 `c16_m62_1 * NVIDIA T4`。
- 外部模型高通量提交已支持 `bohr job_group create` + `bohr job submit -g`。
- 已新增 sigma descriptors 导出，可作为后续 `ML`/`DL` 特征输入。
- `ORCA -> HSP` 这一步当前使用的是基于 sigma moments/体积的经验估算，而不是完整的 COSMO-RS 热力学求解。
- 新增了 `CSV -> dry-run ORCA/OpenCOSMO-RS -> mock HSP/COSMO 描述符` 主链，可用于先验证流程编排。
- 这个仓库已经具备完整流程骨架，下一步重点是提升各阶段的化学精度。

## 项目结构

```text
solubility_param_flow/
├── src/                    # 源代码
│   ├── core/               # 核心计算模块
│   ├── descriptors/        # 分子描述符计算
│   ├── models/             # 机器学习模型
│   └── utils/              # 工具函数
├── tests/                  # 单元测试
├── docs/                   # 文档
├── examples/               # 示例脚本
└── data/                   # 示例数据
```

## 许可证

MIT
