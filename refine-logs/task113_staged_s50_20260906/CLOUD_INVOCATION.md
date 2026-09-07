# Task113 本轮部署调用

日期2026-09-06，用户已授权上传必要代码/配置/合成数据、DeepSeek用户模拟器、400训练+66评测，并明确要求开始。这里只允许此预算，不自动重跑或扩步数。

SSH身份路径 `C:/Users/xiaoy/.ssh/codex_autodl_policyagent`（不读取/输出私钥），端点 `root@connect.bjb1.seetacloud.com:26689`。实例 `autodl-container-b6f24c9ac7-3bf7e58e`。

部署目录 `/root/autodl-tmp/policyagent-deployments/20260906-task113-staged-acc8-s50-v1`。

复用环境账本：`_local_private_runs/tr0905/s100/probe_v2/full_preflight_evidence.json` 与 `云端运行记录.md`，Python3.12.3/Torch2.13.0+cu130/Transformers5.14.1/TRL1.9.0；本轮不重建不升级。上游 `/root/autodl-tmp/tau2-bench-58e5e1a`，提交58e5e1ace69302e6982d27014569c03e0ffccdd2。环境真实可用性由本轮检查，不只靠旧账本。

GPU驱动报告RTX4090/49140MiB，启动前0MiB占用；磁盘清理后约24.65GiB可用。费用单价未知；首个实际优化器更新后再估算时间，用户负责实例开关，不自动关机。

29文件部署包511818字节，SHA256 `4482c4706626d57ef024171022388fe0569eae5d24f62485f8c4a96865206e6c`。
本轮source_commit `5e50138f13c4fcc408c714d074449c1ef05549dc` + dirty源码，实际文件以deployment_manifest逐项哈希为准。部署Git基底为 `5be4b9803588aafe25b2436c140f09cc54145d55`，二者不得混同。

## 无API预检（一次）

在部署目录中执行：

```bash
/root/autodl-tmp/venvs/policyagent/bin/python _local_private_runs/task113_staged_acc8_20260906/cloud_launch_v2.py --mode preflight
```

入口验证payload哈希、训练合同；运行聚焦测试、完整模型/数据/版本/模板预检、审核过的TRL源哈希、原生GRPOConfig的n4/acc8兼容性和seed=2026090601的BF16小矩阵前向反向有限值检查。它不是实际4B模型backward验收；真实显存和更新证据取自随后第一步。状态 `pipeline_preflight.json`，日志 `preflight_tests.log`、`preflight_model.log`。不能调用API或启动轨迹；失败保留，不自动重试。

## 正式后台入口（预检成功且经主Agent核验后）

```bash
set -a
. /root/.policyagent_secrets
set +a
nohup /root/autodl-tmp/venvs/policyagent/bin/python _local_private_runs/task113_staged_acc8_20260906/cloud_launch_v2.py --mode execute > pipeline_launcher.log 2>&1 < /dev/null &
```

仅本机既有凭据注入用户模拟器，不输出凭据。独占 `pipeline_state.json` 防重复启动。严格顺序SFT33条→GRPO50更新/400条→四图→GRPO33条；错误立即停止，无自动换参/重跑。训练前后评测的33条均来自冻结seed与opening。两组n4各自标准化再梯度累积，不合并为n8。

## 启动入口v2修正

主Agent核对实际绘图CLI发现run_dir为位置参数，v1草案误写`--run-dir`。训练尚未启动；使用独立cloud_launch_v2.py改为位置参数，并绑定deployment_manifest_v2.json。v1包/脚本/清单及预检证据保留。v2补充包5252字节，SHA256 `71a32146e1e58e3173f39108bc520d6b90fbcd6f66f9d1b818a997cc244a8a78`。v2预检状态为pipeline_preflight_v2.json，日志preflight_v2_tests.log、preflight_v2_model.log；上文未带v2的状态/日志名仅对应首版。训练参数和奖励完全不变。

## 实际执行记录

- 首次scp连接超时；只读SSH证实实例在线且未残留部署包后，限定15秒连接超时重传成功。并非实验失败或训练重启。
- v1预检PID2085和v2预检PID2825均退出0；各自108 passed、4 known skipped。4项依赖测试注入的`_policyagent_audited_trl_loop`，不得计为通过；真实TRL与Transformers源码哈希另已验证。
- v2预检状态PREFLIGHT_PASSED_NO_ROLLOUTS，清单SHA `4DEE158CFBDF16172B098C1EBB0A1FD78D2157C01237F19CCDB3F79292B033B7`。模型、输入、版本、模板、n4/acc8和BF16小矩阵backward均通过。此非实际4B backward验收。
- vLLM版本警告保留，配置use_vllm=false，未为此改变环境。
- 既有凭据注入只验证存在性，不输出密钥。
- 正式后台已启动PID3480；pipeline_state.json显示RUNNING，SFT配置准备退出0，当前首个n4基线子进程PID3487。尚未确认任何GRPO更新，不得称训练已完成或已有提升。
- v1/v2预检状态回传至本地`_local_private_runs/task113_staged_acc8_20260906/cloud_preflight/`。训练/评测结果仍需后续回传，不自动关GPU。
