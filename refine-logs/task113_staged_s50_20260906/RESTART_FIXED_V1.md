# 修复后重新启动 Task113

2026-09-06，用户明确授权“修复好了就开始”。从同一冻结SFT重新初始化；不续接旧checkpoint。
本轮新增400训练rollout + 33训练后评测；核验并复用已完成33条SFT基线。旧15次更新/60条单独保留。
配置不变：分层v7.1、n4、micro1、累积8、每更新两组、50updates、LR2e-6、beta0.02、temperature0.8。
只改本轮部署启动文件与清单，包含上轮已验收的累积修复；不修改奖励、训练参数或环境。

## 环境账本与最小载荷

沿用CLOUD_INVOCATION.md环境：端点root@connect.bjb1.seetacloud.com:26689；
实例autodl-container-b6f24c9ac7-3bf7e58e；环境/root/autodl-tmp/venvs/policyagent。
Torch2.13.0+cu130 / Transformers5.14.1 / TRL1.9.0不变，不重建。
GPU驱动报告RTX4090/49140MiB，启动前占用0MiB；磁盘可用约24GiB；不删除文件。
GPU小时价格未知。原单组约数分钟，新两组会更慢；用首次真实更新耗时再估计，用户管理实例开关。

独立目录：/root/autodl-tmp/policyagent-deployments/20260906-task113-staged-acc8-s50-fixed-v1。
从旧部署的Git基底只读克隆到新目录，再覆盖新目录的冻结最小载荷；旧部署不变。
Git基底与dirty源码不混淆：源码HEAD5e50138f13c4fcc408c714d074449c1ef05549dc + 文件清单。
载荷33文件，清单SHA256 7DE6A12A46EF11067137AA3877ED1472DAA31EFDCA5DE42A42A143CC30F56066；
ZIP SHA256 C0F18483436F4D3E899CBDD2BCD1FE62C1D99E1FE57E85CD0E16E77FAF919CF3。
配置SHA256 2EBA9D821E3EB645540235F069ABDC16D8293FA6A09372FD019DB654A1B5A98F。

## 独立agent按文档预检（只执行一次）

以下命令验证所有载荷、33条基线的配置/模型/产物哈希，运行聚焦测试、真实TRL CPU计数见证、完整模型数据预检。不运行rollout，不调用DeepSeek。

```powershell
ssh -i C:/Users/xiaoy/.ssh/codex_autodl_policyagent -p 26689 -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=2 -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes root@connect.bjb1.seetacloud.com "cd /root/autodl-tmp/policyagent-deployments/20260906-task113-staged-acc8-s50-fixed-v1 && /root/autodl-tmp/venvs/policyagent/bin/python _local_private_runs/task113_staged_acc8_20260906/cloud_launch_fixed_v1.py --mode preflight"
```

必须exit0，pipeline_preflight.json为PREFLIGHT_PASSED_NO_ROLLOUTS；counter_witness/result.json为PASS，修复后两次合成更新累计[8,16]。
如失败，不修改、不重试、不启动训练，保留日志报告差异。

## 正式启动（仅主agent核验预检后）

在新目录，注入/root/.policyagent_secrets已有凭据（不输出），通过nohup后台启动上述入口 --mode execute。
pipeline_state.json排他创建防止重复运行。直接GRPO→图→冻结33条后评测，不重复SFT。
新分层代码训练开头必须写accumulation_plan.json，真实loader8，每次更新前8微步/8raw/2生成批次PASS。
进程存活不等于首个优化更新成功，必须读取该计数事件与实际step1日志确认。
失败不自动重试/换参，完成后仍需回传与分析。完成训练不是已证明提升。

## 本轮实际结果

载荷部署成功，新目录bundle验证通过。独立agent按文档只执行一次预检，PID9219，exit0。
pipeline_preflight.json=PREFLIGHT_PASSED_NO_ROLLOUTS，三个阶段exit0；聚焦测试42 passed。
真实TRL计数见证：旧loader4、修复loader8；修复后累计[8,16]，每次8微步/8条/2批次PASS。
清单与配置哈希一致，external_api_called=false。未修改训练参数、Reward、库版本，不删除文件。

随后主agent提交nohup正式启动请求，但执行审批在本地CreateProcess之前拒绝，理由是要求明确本轮向DeepSeek发送Task113合成对话的授权。
该拒绝不属于实验或代码错误；没有绕过、换入口重试，未启动正式训练，尚无真实4B首步结果。
需要用户明确本轮新400条训练rollout+33条后评测的DeepSeek用户模拟器传输授权后，才重新提交原定启动命令。

## 用户补充明确授权后的启动

用户随后明确授权：Task113合成对话发往DeepSeek用户模拟器，修复版400条训练rollout及33条后评测。
重新核验独立目录无pipeline_state/无training、GPU无计算进程后，提交同一启动命令，exit0。
实际pipeline监督PID9970、训练PID9971；状态RUNNING、phase=train，开始时间Unix1788701236.978886。
复用33条SFT基线的哈希核验已通过；未重复采样，未从旧checkpoint续训。
当前启动代码/配置保持原清单绑定，未新增修改。刚启动时仅加载环境，不能称已完成首步；后续必须核验实际accumulation事件。

## 首步真实运行核验

已读到真实loader=8，以及before_optimizer_step第1步PASS：microsteps8、fresh_rollouts8、generation_batches2、logged_rollouts_total8。
随后train.log明确进度1/50、epoch1，证明优化器循环完成首步，不只是生成前置计数。GPU95%、35317/49140MiB，PID9971仍运行。
首步打印指标：reward0.775、reward_std0.45、loss-0.02907、grad_norm0.01077、clipped_ratio0、KL0、learning_rate0。
LR0对应冻结配置warmup起始步，此时不能声称权重已有有效变化，更不能声称任务效果提升。
计时约351.39秒/步（日志step_time348.8）；按首步外推剩49步约4.8小时，存在轨迹长度和API延迟不确定性，不含33条后评测。
日志中的DeepSeek v4-flash价格映射错误为成本统计提示，用户模拟回复正常返回；未因此改变模型或评分。
本地首步快照位于 `_local_private_runs/task113_staged_acc8_20260906/fixed_launch_status/`。
当前仍在运行，未关实例、未改参数、未执行额外实验；完成后自动绘图并运行既定后评测，效果尚待评测。
