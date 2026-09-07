# 2026-09-06 云端空间清理

用户授权：允许核查并删除可清理文件以腾出训练空间；前文同时说明了本轮上传、DeepSeek用户模拟器及400训练+66评测预算。此次操作只做空间清理，尚未启动训练/API。

端点：root@connect.bjb1.seetacloud.com:26689；hostname：autodl-container-b6f24c9ac7-3bf7e58e。GPU查询为空闲，驱动报告49140 MiB。

## 删除前核验

两次旧训练manifest均为COMPLETED。按项目directory_sha256的相同排序与分隔语义，流式复核两份adapter目录和两份合并模型目录，均与各自manifest匹配。两份adapter_config均指向保留的共同底模：

`/root/autodl-tmp/policyagent-runs/20260824-sft-v3-agentic-protocol-bridge-s20-v1/teacher_sft_merged`

底模目录哈希实测为 `0A2E06C9BCA6082F3FE6723EC54A46D4CE1D37A8C6DD6B9BC116BBA4BAB16576`，匹配训练绑定。运行代码以BF16底模加载对应adapter再merge_and_unload生成合并模型；恢复需要重新合并，不是回收站还原，未在本轮实际重建。

## 精确删除目标

只对以下两个经过绝对路径解析和非符号链接检查的普通文件执行 `rm --`，没有递归删除：

1. `/root/autodl-tmp/policyagent-deployments/20260905-teacher-refresh-sft20-v1/training/teacher_sft_merged/model.safetensors`
2. `/root/autodl-tmp/policyagent-deployments/20260905-teacher-refresh-sft50-v1/training/teacher_sft_merged/model.safetensors`

各文件删除前8044982080字节。保留合并目录的配置/tokenizer，以及全部adapter、训练checkpoint、manifest、日志和评测。旧manifest不改写；其中merged_model路径在恢复前不再可直接加载，此记录解释其缺失。

本轮使用的 `20260905-teacher-refresh-sft100-v1/training/teacher_sft_merged/model.safetensors`（所选checkpoint30）未删除，清理后stat确认仍为8044982080字节。旧SFT20/SFT50 adapter权重也再次确认存在。

清理前df显示9.7G可用，删除后首次df显示18G可用；随后只读复核df -B1显示可用26467401728字节（约24.65GiB，使用率88%），两个目标文件均确认不存在。文件删除后空间统计有短暂延迟，以最终实际df为准。未继续扩大删除范围。

诊断中一次裸python3调用因远程非交互PATH缺失失败；随后使用已有 `/root/autodl-tmp/venvs/policyagent/bin/python` 完成同一只读核验，未安装环境或修改代码。
