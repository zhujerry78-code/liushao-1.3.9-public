# 流哨 1.3.9｜实验室网络异常研判助手

流哨是元启智联团队开发的本地网络异常研判助手。它读取兼容的流级特征 CSV，支持按异常优先级筛查记录、查看特征贡献，并保存人工复核结论。它不直接抓包，也不替代人工判定。1.3.9 在此基础上以事件为单位组织研判，并对研判结论的形成、继承、撤回与失效进行全过程管理。

## 这个仓库包含什么

本仓库从 1.3.9 总决赛版（封版提交 67579c2）提取可公开部分，包括界面、自研研判代码、部分配置和测试。它是源码快照，不是完整的 Windows 安装包，也不包含原开发仓库的历史。

- `app.py` 和 `app_*.py`：应用界面、研判流程和证据导出。
- `liushao/`：输入数据、特征和复核记录处理，以及事件级研判机制——Search 反例优先检索、Gate 证据门控与批次授权、Recovery 局部撤回与依赖失效、Dynamic 动态一致性、Workbench 单事件研判工作台。
- `liushao_engine/`：异常评分相关统计、证据融合、归因和不确定性判断。
- `delivery/`：导出文件与输出目录的安全处理。
- [引擎输出约定](docs/usage/ENGINE_OUTPUT_CONTRACT.md)：字段含义与适用范围。

冻结模型、打包后的引擎、演示数据、评估原始材料、用户记录和 Windows 离线运行包均不随本仓库公开。没有相应且获授权的运行工件，不能复现总决赛整套实机演示。

## 阅读和检查源码

建议使用独立的 Python 环境，按 `requirements.txt` 安装源码检查所需依赖。可从数据接口和漂移判断的测试开始：

```bash
python -m pip install -r requirements.txt
python -m pytest tests/test_data.py tests/test_drift.py
```

## 许可

自研代码按 [Apache-2.0](LICENSE) 提供。请保留 [NOTICE](NOTICE) 中的署名和说明。第三方许可、数据引用及 AI 辅助开发披露见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
