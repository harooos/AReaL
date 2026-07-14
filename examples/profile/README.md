# Qwen3-30B-A3B SFT Profile

这个目录提供一个端到端 SFT profile 样例，用 Qwen3-30B-A3B 和确定性的 fake 128K-token SFT 数据分别采集 kernel
profile 与 memory profile。

## 文件

- `train_sft_profile.py`: SFT 入口。它不读取外部 JSONL，而是用真实 tokenizer 编码一段结构化 SWE/代码修复对话，再重复截断到
  131072 token。
- `qwen3_30b_a3b_sft_profile.yaml`: 1 节点 8 GPU 的 Megatron MoE profile 配置， 默认使用
  `megatron:(attn:d2p1t2c2|ffn:d2p1e4)`。
- `run_qwen3_30b_a3b_sft_profile.sh`: 一键运行 kernel/memory 两类 profile。
- `postprocess_profile.py`: 生成 kernel Chrome trace 视图和 profile summary。

## 快速运行

在仓库根目录执行：

```bash
MODEL_PATH=/path/to/Qwen3-30B-A3B \
FILEROOT=/path/to/shared/experiments \
bash examples/profile/run_qwen3_30b_a3b_sft_profile.sh
```

默认会运行两次 trial：

- `kernel`: 打开 `perf_tracer.profile_steps`，在 SFT train step 中启动 PyTorch profiler，产出
  CPU/CUDA/GPU kernel trace。
- `memory`: 打开 `memory_profiler.profile_steps`，产出 PyTorch CUDA allocator snapshot。

默认设置：

```bash
PROFILE_STEP=1
TOTAL_STEPS=2
PROFILE_RANKS=0
PROFILE_KINDS=kernel,memory
PROFILE_FAKE_SEQ_LEN=131072
PROFILE_FAKE_DATASET_SIZE=8
PROFILE_FAKE_LOSS_START_RATIO=0.5
TRAIN_BATCH_SIZE=4
PROFILE_N_MBS=4
```

只跑 kernel profile：

```bash
PROFILE_KINDS=kernel \
MODEL_PATH=/path/to/Qwen3-30B-A3B \
bash examples/profile/run_qwen3_30b_a3b_sft_profile.sh
```

只跑 memory profile，并采集多个 rank：

```bash
PROFILE_KINDS=memory \
PROFILE_RANKS=0,1,4-7 \
MODEL_PATH=/path/to/Qwen3-30B-A3B \
bash examples/profile/run_qwen3_30b_a3b_sft_profile.sh
```

## Fake 128K 数据

`train_sft_profile.py` 的 fake 数据生成逻辑有三个约束：

1. 使用目标模型 tokenizer 编码结构化对话文本，而不是直接填充同一个 token。
1. 重复同一段 token 序列直到固定长度 `PROFILE_FAKE_SEQ_LEN=131072`，保证不同 parallel layout 看到完全相同的 token
   内容。
1. `loss_mask` 默认从 50% 位置开始为 1，前半段作为 prompt/context，后半段作为 assistant target。

这样可以避免真实数据 IO 与动态过滤影响 profile，同时保留长上下文、代码块、JSON 片段和自然语言 target 的 tokenizer 分布。

## 产物

脚本默认把运行侧产物放在 `examples/profile/` 下：

```text
examples/profile/profile_data/<timestamp>_qwen3-30b-a3b_fake128k_sft_profile/
  profile_settings.log
  summary.tsv
  qwen3_30b_a3b_fake128k_kernel_<timestamp>/
    launcher.log
    nvidia_smi.csv
    profile_summary.json
    profile_summary.md
    kernel_traces/master/
      traces-r0.chrome.json
      traces-r0.split_clean.chrome.json
      traces-r0.gpu_only.chrome.json
      traces-r0.cpu_only.chrome.json
      traces-r0.cuda_api_only.chrome.json
  qwen3_30b_a3b_fake128k_memory_<timestamp>/
    launcher.log
    nvidia_smi.csv
    profile_summary.json
    profile_summary.md
    memory_snapshots/step_<PROFILE_STEP>/
      snapshot_*.pickle
```

AReaL 原始日志仍在 `FILEROOT` 下：

```text
${FILEROOT}/logs/<user>/qwen3-30b-a3b-sft-profile/<trial_name>/
  trainer.log
  perf_tracer/<role>/traces-r*.jsonl
  memory_snapshots/step_<PROFILE_STEP>/snapshot_*.pickle
```

kernel profile 后处理会在 trace 文件旁生成 Chrome trace 视图，并复制一份到
`profile_data/.../kernel_traces/<role>/`：

```text
traces-r0.chrome.json
traces-r0.split_clean.chrome.json
traces-r0.gpu_only.chrome.json
traces-r0.cpu_only.chrome.json
traces-r0.cuda_api_only.chrome.json
```

用 Chrome `chrome://tracing` 或 Perfetto 打开这些 `.chrome.json` 文件即可查看。

## 单独后处理

如果 profile 已经跑完，只想重新生成 summary 和 kernel trace views：

```bash
python examples/profile/postprocess_profile.py \
  --profile-kind kernel \
  --profile-step 1 \
  --log-dir /path/to/FILEROOT/logs/<user>/qwen3-30b-a3b-sft-profile/<trial_name> \
  --run-dir examples/profile/profile_data/reprocess/<trial_name> \
  --trainer-log /path/to/trainer.log \
  --nvidia-smi-csv /path/to/nvidia_smi.csv
```

## 注意事项

- kernel profile 与 memory profile 建议分开跑；torch profiler 的 memory 记录会扰动 kernel 时间线，本样例默认对
  kernel trial 设置 `AREAL_TORCH_PROFILER_PROFILE_MEMORY=false`。
- `PROFILE_RANKS` 控制采集 rank。为空表示全部 rank；`0,2-4` 表示 rank 0、2、3、4。
- Qwen3-30B-A3B 需要 8 GPU profile 环境；没有对应 GPU/模型权重时只能验证脚本和后处理， 无法实际产出 CUDA trace 或 memory
  snapshot。
