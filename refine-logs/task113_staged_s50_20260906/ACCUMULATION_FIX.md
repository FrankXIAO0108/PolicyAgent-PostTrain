# Task113 epoch 末尾提前更新修复

日期：2026-09-06。用户授权：暂停并修复；不自动重启或续训。

## 暂停事实

端点 root@connect.bjb1.seetacloud.com:26689，实例 autodl-container-b6f24c9ac7-3bf7e58e。
先核验 supervisor PID3480 与训练 PID6043 的完整命令，随后依次发送 SIGTERM。
复查两个 PID 均消失，nvidia-smi 计算进程列表为空。没有删除实验文件。
原 pipeline_state.json 的 RUNNING 是被终止监督器的旧记录，不能继续作为活跃状态依据。
旧部署：/root/autodl-tmp/policyagent-deployments/20260906-task113-staged-acc8-s50-v1。
停止时完整 raw/evidence 各60条，日志到15次更新；已落盘 trainer/checkpoint-10。
第16组可能存在未完成的生成事件，不冒充完整轨迹或已更新参数。
checkpoint-10 含 adapter、optimizer、scheduler、rng_state、trainer_state；不是第15步参数快照。

## 根因和修复范围

原代码只有1条 opening。TRL1.9.0 RepeatSampler 与生成缓冲使每epoch只有4个微批次；
Transformers5.14.1 在epoch末尾强制更新，实际累积4次，而不是配置声明的8次。
这是项目数据迭代与预检遗漏，不是Reward失败或模型能力结论。

- 新增 src/training/accumulation_contract.py：按最小公倍数等比例重复整个prompt池。本次1行变2行，仍为同一Task113、同一opening，不增加任务或教师数据。
- run_retail_agentic_grpo.py 仅在显式工程合同、OPTIMIZE、累积窗口大于生成窗口时接入。纯采样、单生成窗口旧运行不改变。
- 不复制completion：每个窗口调用原生生成，两个n4组独立标准化。不变成n8组。
- 实际dataloader长度必须与预期一致且整除累积步数；每次optimizer更新前核验8微批次、2生成批次、8新rollout以及raw日志累计条数。不符合就抛错，不执行该次optimizer更新。
- accumulation_plan.json / accumulation_events.jsonl 保存并纳入最终产物哈希。
- 多组累积checkpoint续训尚未审计，明确拒绝。旧checkpoint不可拼成修复后的batch8实验；正式重跑须从同一冻结SFT起点、独立目录开始。
- 不改配置、Reward、KL beta、学习率、temperature、n或旧部署源码，不升级库。

## 当前验证

- 最终本地纯计数测试15 passed（系统Python含Transformers）；项目合同/启动/旧续训/Task44配置回归107 passed（tau2环境）；合计122项。Ruff通过。
- 最初误把依赖Transformers的计数测试放进tau2轻量环境，7项因缺少Transformers失败、41项通过；这是解释器选择错误，随后分环境重跑，不安装依赖，不记成训练失败。
- 云端CPU witness_v1：真实TRL的sampler、get_train_dataloader、_prepare_inputs、training_step与真实Transformers优化器循环；生成和loss是合成测试替身，绝非真实RL指标。
- 旧循环2次更新共8条合成样本；修复后2次更新共16条，各自只消费一次；每更新恰好8微步、2生成批次；错误loader在任何生成/更新前阻断。
- 没有调用API、下载模型或占用GPU。不据此声称4B训练效果提升。

## 独立复现调用（无API、CPU）

环境账本：CLOUD_INVOCATION.md 与 _local_private_runs/tr0905/s100/probe_v2/full_preflight_evidence.json。
复用 /root/autodl-tmp/venvs/policyagent，Torch2.13.0+cu130 / Transformers5.14.1 / TRL1.9.0，不重建。
修复检查目录 /root/autodl-tmp/policyagent-deployments/20260906-task113-accumulation-fix-check-v1。
只上传修复模块和合成计数脚本，没有真实对话或密钥。

在本地PowerShell执行一次，输出目录不得复用：

```powershell
ssh -i C:/Users/xiaoy/.ssh/codex_autodl_policyagent -p 26689 -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=2 -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes root@connect.bjb1.seetacloud.com "cd /root/autodl-tmp/policyagent-deployments/20260906-task113-accumulation-fix-check-v1 && /root/autodl-tmp/venvs/policyagent/bin/python scripts/check_grpo_accumulation_runtime.py --output-dir witness_review_v1"
```

必须exit0、result.json status PASS、旧loader4/新loader8，修复后更新计数[8,16]。
脚本不覆写既有输出目录。生成/loss替身是该验证的明确边界，不用于Reward评估。

## 收尾与对抗式复核

独立agent严格执行上述witness_review_v1一次，exit0，旧/新计数与文档一致。指出首版witness没有传入raw行数读取函数，不能验证这一分支；另指出计数护栏不验证completion内容语义或身份唯一性。

针对第一点，最终witness_v2增加合成JSONL ledger并走同一个行数读取护栏，两次更新的logged_rollouts_total分别8和16，通过。最终模块在调用父类生成前保存model.training，防止父类临时切换模型模式影响计数。v2已由主agent在同一真实框架复测；独立agent执行的是先前v1，二者不混称。

第二点作为边界保留：护栏验证生成调用数、输入数量、微步数和日志条数，不证明轨迹内容多样性。当前原生sampler/缓冲由CPU witness验证16个合成样本各消费一次；真实completion正确性仍由原有证据/Reward审计处理。本次不为此重写GRPO或引入新奖励。

最终模块SHA256：44C10481593E753429854D5F477AB47BEA4B60AB4AB986AA59F012FAD5FA5C61。
最终CPU脚本SHA256：2BAD3DECE3629611E5B742131A8F48EE03CF82EC4C2891657A423B8FE493E3A4。
云端与本地两者哈希一致。真实库源文件SHA见runtime_witness_v2.json；运行环境未改变。

暂停归档已回传并解包至 `_local_private_runs/task113_staged_acc8_20260906/pause_fix_v1/archived/`。
压缩包SHA256（远端/本地一致）：9269726838BA9E791C72BFCE99EC08208C0059156CF28F2C6DB00EA859566B2E。
包含训练日志、原pipeline状态、60条raw与60条evidence、生成事件、配置/命令、checkpoint10状态。
检查点权重与优化器文件仍保留在原云端目录，未声称完整权重已经回传。单文件未压缩SCP传输曾停滞，已终止该传输并改用压缩归档；第一轮未完成的零字节副本不能作为证据，验收只用archived目录。

最新检查GPU计算进程为空。未重启正式训练、未自动执行后评测。修复runner仅在本地；云端仅部署了隔离的CPU验证模块/脚本，原实验部署未覆盖。

后续如批准重新训练：新目录、从冻结SFT重新初始化，重新绑定源码哈希和运行预算；不能直接再用旧cloud_launch_v2或checkpoint10续跑。旧60条为另一次实际batch4实验，不计入新batch8的400条。
