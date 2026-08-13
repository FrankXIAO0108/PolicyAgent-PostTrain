# Qwen3-4B 多步 Tool-SFT 实跑结果

> 日期：2026-08-13
> 运行范围：隔离的合成多步工具协议实验
> 结论：工程闭环完成且同分布留出改善；不等同于 Retail 业务提升或正式训练数据门开放。

## 1. 实验绑定

- 起始模型：已完成的 Qwen3-4B 单步 Tool-SFT merged checkpoint；
- 起始模型 SHA-256：`2028C809473FA52434BE00583D3C0E5C29633C447A479B17A51E28EEE9D0B4D0`；
- 项目提交：`63dd9253b9700c021a8b99e7bf735167a86057b1`；
- 数据：24 条训练轨迹切为 136 个决策点，6 条留出轨迹切为 34 个决策点；
- 训练：4-bit NF4 QLoRA，136 steps，4 epochs，学习率 `5e-5`；
- GPU：单卡 RTX 4090；
- 训练耗时：766.9145 秒；
- Train loss：0.0338598；
- merged checkpoint SHA-256：`030B393F7181224C9C15A8689276258AEB83DD704E798C2C693CACF13E1FC289`。

完整小型证据位于 `experiments/20260813_qwen3_4b_multistep_tool_sft_v1/`，模型权重保留在 AutoDL，未提交仓库。

## 2. 冻结留出结果

| 指标 | 训练前 | 多步 SFT 后 |
|---|---:|---:|
| Valid Tool Call | 76.47% | 100.00% |
| 总体 Tool Match | 52.94% | 82.35% |
| 总体参数 Match | 47.06% | 82.35% |
| 首步 Tool / 参数 Match | 100% / 100% | 100% / 100% |
| Post-tool Tool Match | 42.86% | 78.57% |
| Post-tool 参数 Match | 35.71% | 78.57% |
| 确认后写操作 Tool / 参数 Match | 100% / 100% | 100% / 100% |

结果支持一个有限结论：在同分布的合成协议留出上，多步决策点 SFT 改善了读取工具返回后的下一步动作，且没有破坏首步能力。

## 3. 不能过度解释的原因

1. 留出只有 34 个决策点，其中确认后写操作只有 4 个；
2. 训练与留出由相同模板生成，末段 token accuracy 达到 1，存在明显同分布拟合；
3. 某些场景允许多条业务上合理的路径，而 exact-match 只接受唯一工具；
4. 合成轨迹未经独立人工双审，不是正式 Retail 金标；
5. 该结果尚未证明真实动态用户交互中的完成率、策略合规或终态正确性改善。

因此 `run_manifest.json` 正确保持：正式 Retail gate 未开放、业务提升声明禁止、Agentic GRPO 尚不允许。

## 4. 下一步实验

使用新 merged checkpoint 复跑与 v5 完全相同的 8-task、32-rollout 多轮诊断。除起始模型外，任务、opening、seed、温度、DeepSeek 用户模拟器、reward 与 rollout 上限保持冻结。重点比较：

- 29/32 只执行两次工具调用是否下降；
- 平均工具调用数与 required-action recall 是否提升；
- unfinished 32/32 是否改善；
- 工具错误、重复调用和未确认写操作是否增加；
- 同任务组内 reward/action variance 是否足以支持 GRPO。

只有真实多轮诊断也改善，才能说明多步 SFT 缓解了原始停滞问题；只有 reward 具备稳定区分度时，才继续 GRPO。
