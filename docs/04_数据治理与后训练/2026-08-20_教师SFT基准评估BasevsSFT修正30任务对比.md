# 教师 SFT 基准评估：Base vs SFT 修正 30 任务对比

日期：2026-08-20  
状态：30/30 均有有效 reward；开发级组合证据，非正式业务门禁

## 0. 一句话结论

修复评测 driver 的基础设施失败统计口径，并在不覆盖原产物的前提下重跑 task 100、107 后，
Qwen3-4B-Instruct Base 与教师 SFT 的 30 任务开发级对比为：
`12/30 = 40.0%` → `17/30 = 56.7%`，净增 5 个任务（+16.7 个百分点）。
其中 train_candidates 为 `6/13` → `7/13`，test_clean 为 `6/17` → `10/17`。

该结果由原 SFT 运行的 28 个有效任务和 2026-08-20 新增的两个单任务重跑结果组成，
不是一次连续完成的 30 任务新全量运行，也不是 4-trial 正式结论。

## 1. 为什么必须修正

原 driver 将 tau2 已返回但 `reward_info=None` 的 simulation 强制写成 `reward=0.0`，同时
`system_failures` 只统计 driver 抛出的异常。结果是 task 100、107 的
`ContextWindowExceededError` 被混入普通模型失败，原 summary 显示 `15/30 = 50.0%`，
但实际只有 28 个任务得到有效 reward。

修复提交 `6267ccff1216e22d44359caf0eebb31db9ab0bf3` 做了三件事：

1. 无 reward 的 simulation 不再伪装为模型 `reward=0`；
2. summary 单列 `infrastructure_failures` 和 `coverage`；
3. 成功率分母只包含真正得到 reward 的任务。

原冻结目录未修改。task 100、107 分别写入新的独立目录。

## 2. 运行与证据绑定

| 项目 | 原 Base / SFT 全量运行 | task 100 / 107 新增重跑 |
| --- | --- | --- |
| 评估 config | `configs/retail_teacher_eval_v1.json`，SHA-256 `83DC56A118A4BD6FEA35E38A2820E30F1D6922761821439EB3AD404BF7D2D947` | 相同 |
| 项目 commit | `0462e71d02b15dae931a552fae5a1cbd0b063952` | `6267ccff1216e22d44359caf0eebb31db9ab0bf3`；差异为报告口径及文档，不改变模型 checkpoint 或冻结评估 config |
| 上游 tau2 | `58e5e1ace69302e6982d27014569c03e0ffccdd2` | 相同 |
| SFT checkpoint | SHA-256 `2A19D74C527AD01290D9E70C049F5604A8167C5BBC32BCD77D08E84B3740D289` | 相同 |
| 协议 | 1 trial，temperature 0，seed 20260818，`ALL_WITH_NL_ASSERTIONS` | 相同 |
| 用户 / NL Judge | config 均为 `deepseek/deepseek-chat`，temperature 0 | 相同 |
| vLLM 上下文 | `20480` | `24576`，用于解除已确认的上下文基础设施上限 |
| vLLM 重跑日志 | — | `_local_private_runs/vllm_sft_rerun_100_107_20260820.log`，SHA-256 `E866D05DD44DDB004CDCCFC209BDAC40AB30D37E47E3C9924FD8862DF6EC92A2` |

原始汇总文件：

- Base `eval_summary.json`：SHA-256 `5DEBC7A3F6A5221A7CDCEB119071534453A60EFFD5106D43ACC63E935371F43E`
- 原 SFT `eval_summary.json`：SHA-256 `3C54D7622D6AF7A2A3F8F8A065F84019FE51252C05E35DA6167747B9420511F3`
- task 100 重跑 `eval_summary.json`：SHA-256 `540E82A994890AE6616CAF370816711EBF865D4510A0F1033D509AB3BB2C99BB`
- task 107 重跑 `eval_summary.json`：SHA-256 `6D77ABFB4837680EFE7A593639E5C7CE239A32195C118442B50CAB71A1D5B228`

## 3. 修正后的 Base vs SFT 对比

| 指标 | Base | 修正后 SFT | 变化 |
| --- | --- | --- | --- |
| 整体成功率 | 12/30 = 40.0% | 17/30 = 56.7% | +5 任务，+16.7pp |
| train_candidates（13） | 6/13 = 46.2% | 7/13 = 53.8% | +1 任务，+7.6pp |
| test_clean（17） | 6/17 = 35.3% | 10/17 = 58.8% | +4 任务，+23.5pp |
| 有效 reward 覆盖 | 30/30 | 30/30 | 任务覆盖完整；SFT 为组合结果 |
| 基础设施失败 | 0 | 0（组合后） | 原 SFT 的 2 条已由独立重跑替换 |

逐任务变化共 8 升、3 降、19 持平；净变化为 `8 - 3 = +5`。

| task | source | Base | 修正后 SFT | SFT 证据来源 | 变化 |
| --- | --- | ---: | ---: | --- | --- |
| 59 | train_candidates | 0 | 1 | 原全量运行 | 升 |
| 72 | train_candidates | 0 | 1 | 原全量运行 | 升 |
| **107** | train_candidates | 0 | 1 | **2026-08-20 新增重跑** | **升** |
| 40 | test_clean | 0 | 1 | 原全量运行 | 升 |
| 74 | test_clean | 0 | 1 | 原全量运行 | 升 |
| 79 | test_clean | 0 | 1 | 原全量运行 | 升 |
| **100** | test_clean | 0 | 1 | **2026-08-20 新增重跑** | **升** |
| 108 | test_clean | 0 | 1 | 原全量运行 | 升 |
| 66 | train_candidates | 1 | 0 | 原全量运行 | 降 |
| 67 | train_candidates | 1 | 0 | 原全量运行 | 降 |
| 90 | test_clean | 1 | 0 | 原全量运行 | 降 |

其余 19 个任务结果持平：
`18、19、24、36、37、38、39、43、50、64、65、68、71、76、77、89、101、102、105`。

## 4. 两个新增重跑任务

### Task 100（test_clean）

- 原结果：`termination_reason=infrastructure_error`，`reward_info=null`，
  `error_type=ContextWindowExceededError`；不是业务失败。
- 新结果：`termination_reason=user_stop`，reward `1.0`，DB `1.0`，NL assertion `1.0`。
- 新结果动作检查：`modify_pending_order_items`、`return_delivered_order_items` 均匹配。
- 原 `returned_results.json` SHA-256：
  `9C87CE648E007366B25C47509029AF1DE0BD019B1D78D75B317A02C147267747`
- 新 `returned_results.json` SHA-256：
  `A62876E90C8FD44E3FC462F84932054761CC39EE37685D3CBFD68DFC0E28BD12`
- 新 `run_manifest.json` SHA-256：
  `C5A4D4EC645778978CE6C51F941C50104739373D465A3FD11E7AE29737C89C2D`

### Task 107（train_candidates）

- 原结果：`termination_reason=infrastructure_error`，`reward_info=null`，
  `error_type=ContextWindowExceededError`；不是业务失败。
- 新结果：`termination_reason=user_stop`，reward `1.0`，DB `1.0`，NL assertion `1.0`。
- 新结果动作检查：两个 `exchange_delivered_order_items` 均匹配；NL Judge 判定完成两个订单的换货。
- 原 `returned_results.json` SHA-256：
  `22B91B2B5017036CBC6BBB3A2120E6D38DD001A43E7E944720D7307AE02BC7F9`
- 新 `returned_results.json` SHA-256：
  `81AB9BE20465D64C84E277610214ED813BD8F833F4E8B8F1BD9133415B06A939`
- 新 `run_manifest.json` SHA-256：
  `52ACE4FFABE4FE657D8D57773B44264B73CEC930DF01D8F6F8E9AB28F96419F6`

## 5. 结论边界

1. 这张表解决的是评测覆盖和统计口径问题：30 个任务现在都有有效 reward。
2. 修正 SFT 数字是组合结果：28 条来自 2026-08-19 全量运行，2 条来自 2026-08-20 独立重跑。
3. 重跑把 vLLM 上限从 20480 提到 24576。模型、checkpoint、任务、seed、temperature、评估 config
   以及用户/Judge 的配置 model string 均未改变，但它仍不是完全相同服务参数下的一次全量复跑，
   外部 API 实际后端也不由本仓库控制。
4. 1 trial 只能作为开发级证据；没有均值、方差或置信区间，不能证明提升稳定。
5. test_clean 来自官方 test split，但当前用途是开发分析；config claims 明确禁止业务改进和正式门禁声明。
6. 当前 `run_manifest.json` 尚未直接记录 vLLM 启动参数；本次通过独立日志和 SHA-256 补充追溯。

因此，面试中可以准确表述为：

> 在实体隔离的 30 个 Retail 开发任务上完成 Base 与教师 SFT 的 1-trial 冻结评估；修复两条上下文
> 基础设施失败并独立重跑后，有效覆盖达到 30/30，组合结果由 12/30 提升到 17/30。该结果是开发级
> 单次证据，后续仍需 4-trial 协议验证稳定性。

本文档修正并取代
`2026-08-19_教师SFT基准评估BasevsSFT对比与回退分析.md` 中的总体成功率、分层成功率和升降任务数量；
原文的 task 66、67、90 回退归因仍有效。
