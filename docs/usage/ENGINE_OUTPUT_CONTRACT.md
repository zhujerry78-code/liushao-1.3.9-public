# 引擎输出接口约定（1.3.8）

- 版本：contract-7（2026-11-14，保证适用性披露与 O1；contract-6 为 T12 + O1，contract-5 为 T11，contract-4 为 T10，contract-3 为 T12a，contract-2 为 T09，首版 contract-1 为 2026-11-12，阶段 P0）。
- 读者：做界面（T19）、出图（T24）、验收的同学。照这份约定，可以用夹具开发，不必等引擎定版。
- 维护：**只有算法线可以改这份约定**。改动时更新本文件的版本号和文末的变更记录，并通知队友。界面代码不得自己往预测表里加引擎列，也不得改 `liushao_engine/`。
- 配套夹具：`tests/engine_contract_fixture.py`。测试 `tests/test_engine_output_contract.py` 保证本文、夹具和代码中的列名一致。

## 1. 数据从哪里来

`app_service.with_engine_status(pred, features, bundle, artifacts)` 返回 `(预测表, EngineStatus)`。

- 预测表是冻结底座的输出（`record_id`、`rf_score`、`lr_score`、`rf_threshold` 等），引擎只在后面**追加列**，不改行数和行序。
- 引擎加载或运行失败时，返回未改动的底座预测表，`EngineStatus.degraded` 为真。界面必须仍然可用（见第 5 节）。
- 界面只读这些列，不重算，不猜测缺失的值。

列分三档，文中用标记区分：

| 标记 | 含义 |
|---|---|
| 【已实现】 | 当前 `artifacts/engine.joblib` 已经输出 |
| 【已定名】 | 名称、类型、取值已定，P1 中实现；现在只在夹具里有 |
| 【可选】 | 任务可能被砍（T13），列可能永远不出现 |

## 2. 逐条记录的列

### 2.1 基础列（`liushao_engine.COLUMNS`）【已实现】

| 列 | 类型 | 取值与含义 |
|---|---|---|
| `ecod_score` | float | ECOD 尾部分数，只用于证据页的逐字段分解 |
| `ds_belief` | float [0,1] | Bel(已知攻击) |
| `ds_plausibility` | float [0,1] | Pl(已知攻击) |
| `ds_conflict` | float [0,1] | 冲突度 K。开放框架下 K **只来自 RF 与 LR 的分歧**（方案中写作 K_RF,LR），不另设列 |
| `ds_betp` | float [0,1] | BetP(已知攻击) |
| `cp_set` | str | 共形预测集：`正常` / `攻击` / `不确定`。注意：双标签和空集目前都写成 `不确定`，用 `cp_set_size` 区分 |
| `cp_set_size` | int | 0 = 空集（两类都不像），1 = 单标签，2 = 双标签 |
| `cp_p_attack`、`cp_p_normal` | float [0,1] | 两类的共形 p 值 |
| `va_low`、`va_high` | float [0,1] | Venn–Abers 概率端点，不是置信区间 |
| `engine_review` | bool | 是否进入复核队列。P1 之后等于「模糊队列 ∪ 新异队列 ∪ 风险控制队列」 |
| `engine_version` | str | `engine-…`（format 1）或 `engine2-…`（format 2） |

身份列（`app_service` 追加）：`engine_sha256`、`engine_descriptor_sha256`、`engine_inference_version`，均为 str。

### 2.2 开放框架列（`liushao_engine.OPEN_COLUMNS`）【已实现】

只有开放框架引擎（format 2，`fusion.frame == 'open'`）输出。可用 `'ds_bel_unknown' in pred` 判断。

| 列 | 类型 | 含义 |
|---|---|---|
| `typ_score` | float ≥ 0 | 典型性分数（−log p，越大越不典型） |
| `ds_bel_normal` | float [0,1] | Bel(正常) |
| `ds_bel_unknown` | float [0,1] | Bel(未见模式 U) |
| `ds_betp_normal`、`ds_betp_unknown` | float [0,1] | BetP(正常)、BetP(未见模式) |

### 2.3 四态与队列（【已实现】：`ow_conflict_gate` 为 T12a，其余各列为 T11；`ow_queue` 的「风险控制」取值待 T12）

| 列 | 类型 | 取值与含义 |
|---|---|---|
| `ow_state` | str | 【已实现】四态：`正常` / `攻击` / `模糊` / `新异`。每条记录恰好一个。判定按顺序：共形空集 → 新异；双标签 → 模糊；冲突门控 → 模糊；单标签且对应的 Bel 严格最大 → 正常 / 攻击；单标签但 Bel(U) 严格最大 → 新异；其余 → 模糊（`configs/t11_rules.json`） |
| `ow_queue` | str | 【已实现】所在复核队列：`''`（不在队列）/ `模糊` / `新异` / `风险控制`。同时满足多项时按 新异 > 模糊 > 风险控制 取一个。T11 起 `engine_review` = 本列非空；「风险控制」在 T12 实现前不出现 |
| `ow_normal_reject` | str | 【已实现】哪条检验拒绝了「正常」：`''`（都没拒绝）/ `supervised`（监督证据）/ `typicality`（典型性证据）/ `both` |
| `ow_conflict_gate` | bool | 【已实现】是否因 RF 与 LR 的冲突 K 超过门限而入队。T12a 选中「K ≥ 验证 A 半区 K 的 95% 分位」（0.1354）；只有带 `conflict_gate` 参数的引擎（T12a 起）输出这一列。注意：它与共形集合可以同时成立，「只因冲突入队」= 本列为真且 `cp_set_size == 1` |
| `cp_p_normal_supervised`、`cp_p_normal_typicality` | float [0,1] | 【已实现】正常类两条检验各自的 p 值（报漂移的批次上为加权 p 值） |
| `engine_rank` | int，1..n，不重复 | 【已实现】复核优先级的全序，1 最先复核。覆盖全部记录，不只是队列内的；队列内的记录排在前面 |

界面文字对应：`supervised` 显示为「监督证据（RF、LR）认为不像正常」，`typicality` 显示为「典型性证据认为不像见过的流量」。

### 2.4 事件编号与新异分簇【已实现】（O1，取代原 T13；`ow_event` 为 contract-6 新增）

| 列 | 类型 | 含义 |
|---|---|---|
| `ow_event` | 可空整数 | 【已实现，新增】疑似事件编号（从 0 起）：模糊、新异两队列内的记录，按「四态 + 典型性尾部贡献最大的 3 个特征」归并；0 号事件含最急的记录。队列外记录为空。只有带 `events` 参数的引擎（T12 起）输出 |
| `novelty_cluster` | 可空整数 | 【已实现】新异记录的簇号，等于该记录的 `ow_event`；非新异记录为空 |

A24 配置为 k=3、reps=3。界面显示共同四态、冻结签名的特征、事件规模与代表抽样覆盖。代表与已有直接复核存在分歧时退回逐条复核。代表一致后仍由用户确认是否应用到其余成员。整簇继承快照保留签名特征、事件规模及代表操作编号。0.95 纯度用于离线统计，线上以可见复核分歧识别已观察到的混簇。整簇处置属于辅助建议，逐条 CRC 的统计保证仅适用于预先限定的逐条风险。开发集结果仍属探索性。

## 3. 批次级的列（漂移五列【已实现】，T09；`guarantee_ess`【已实现】，T10；其余【已定名】，T12）

批次级的值对本次上传的所有记录都相同，**每行重复写一份**，这样保存证据快照时可以直接从记录读到。界面取第一行即可。

| 列 | 类型 | 取值与含义 |
|---|---|---|
| `drift_flag` | bool | 【已实现】本批是否报漂移（统计量超过其零分布的 95% 分位） |
| `drift_delta` | float [0,1] | 【已实现】整批漂移强度 δ |
| `drift_delta_rf`、`drift_delta_lr`、`drift_delta_typ` | float [0,1] | 【已实现】三路专家各自的 δₛ |
| `risk_alpha` | float (0,1) | 【已实现】本次生效的风险水平 α，默认 0.05 |
| `guarantee_status` | str | 【已实现】`有效` / `加权有效` / `失效` / `不适用`，见 `configs/guarantee_status_rules.json`。只有带 `risk_control` 参数的引擎（T12 起）输出 |
| `guarantee_ess` | float 或空 | 【已实现】加权共形的有效样本量（B2 校准集上正常类与攻击类 ESS 的较小者）。只在本批报漂移时加权；不报漂移、记录太少或引擎不加权时为空。只有带 `weighting` 参数的引擎（T10 起）输出这一列 |

漂移五列的取值补充（contract-2）：

- 只有带漂移模型的引擎（`'drift' in engine`，T09 起的开放框架引擎）输出这五列，名称见 `liushao_engine.DRIFT_COLUMNS`；旧引擎没有这五列，按第 5 节降级。
- 本批少于 200 条（漂移检测的最小批大小档）时不检测：`drift_flag` 为 False，四个 δ 为空（NaN），界面显示「—」并可注明「记录太少，未检测」。
- 漂移检测按整批做一次。界面若只分析上传文件的一部分，这五列仍是整批的值。
- T09 的选择结果是**不折扣**（λ = 0）：报漂移不改变本批的判定和队列，只作提示；折扣机制保留在引擎里，供消融。

展示要求（来自评估记录，界面文字须遵守）：

- 只说「本批与校准数据的分布有差异」，不得说成「检出了攻击」。δ 的数值大小不代表保证失效的程度。
- 保证状态只在整批级别显示，不按时间块或单条记录显示。
- `不适用`：模糊＋新异占比 ≥50%，或未见已知攻击证据（四态中无「攻击」），或小批次未检测漂移。界面需写明原因。线上证据来自引擎判定，真实已知攻击数只在带标签评估中单列。漂移未检测用 `drift_delta` 为空识别。
- `有效` / `加权有效` 的含义是「已知攻击被放行的比例不超过 α」；未见攻击不在保证之内。`失效` 时要明说本批不在保证范围。

## 4. α 设置与预算反查【已实现】（T12）

这两个是函数，不是列。旧引擎（没有 `risk_control`）调用会抛 `ValueError`；界面先判断 `'guarantee_status' in pred`，没有就隐藏对应控件。

```python
liushao_engine.apply_alpha(frame, engine, alpha) -> DataFrame
# 按新的 α 重算 ow_queue、engine_review、engine_rank、risk_alpha、guarantee_status；不重跑底座。
# alpha 只能取 liushao_engine.RISK_ALPHAS 中的值（协议的 risk_alpha_grid：0.005、0.01、0.02、0.05、0.10）。

liushao_engine.alpha_for_budget(frame, engine, k) -> float | None
# 给定复核预算 k（条数），返回能达到的最小 α；达不到任何档位时返回 None。
```

## 5. 缺列时界面怎么降级

规则：**逐列判断 `列名 in pred`，缺哪一项只隐藏哪一项**，不报错，也不用别的列拼凑。

| 情况 | 预测表里有什么 | 界面表现 |
|---|---|---|
| 引擎不可用（文件缺失、校验失败、版本不符、运行出错） | 只有底座列 | 显示降级横幅（`EngineStatus.message`），队列和排序退回 RF/LR 规则，与现在一致 |
| format 1 引擎（1.3.7） | 2.1 | 现有三态展示（正常 / 攻击 / 不确定） |
| 当前开放框架引擎 | 2.1 + 2.2 | 同上，证据页显示三路证据的质量分配 |
| 缺 `ow_state` | — | 不显示四态计数，退回 `cp_set` 的三态计数 |
| 缺 `ow_queue` | — | 「需要再检查」不拆分，用 `engine_review` 作为单一队列 |
| 缺 `engine_rank` | — | 用 `liushao_engine.engine_ranked` / `engine_review_ranked` 排序 |
| 缺 `ow_normal_reject` 或两条 p 值 | — | 证据页不显示「拒绝来源」一行 |
| 缺 `drift_flag` / `drift_delta` | — | 不显示漂移提示；不要调用旧的 `drift_check` 代替 |
| 缺 `guarantee_status` 或 `risk_alpha` | — | 不显示保证状态，也不显示 α 控件 |
| 缺 `novelty_cluster` 或 `ow_event` | — | 新异队列、模糊队列按记录逐条列出，不按簇或事件 |
| 某列存在但该行为空（NaN / None） | — | 该项显示「—」 |

`EngineStatus.code` 取 `ok`、`missing`、`tampered`、`version_mismatch`、`runtime_error`；文字在 `app_service.ENGINE_DEGRADED`。

## 6. 证据快照（`engine` 块，schema_version 2）

保存复核时由 `app_engine_evidence.build_engine_evidence` 生成。

| 字段 | 内容 |
|---|---|
| `schema_version` | 2 |
| `method_version` | `engine-evidence-v1` |
| `status` | `verified` / `not_used`（本次分析没用引擎，同时给出 `reason`） |
| `engine_format` | 1 或 2 |
| `identity` | `engine_sha256`、`engine_descriptor_sha256`、`engine_version`、`base_model_version`、`base_model_sha256`、`engine_inference_version` |
| `results` | 2.1 的全部基础列，保存时与重算值核对过 |
| `open_world` | 见下表；引擎没有产出的值为 `null`，不补造 |
| `parameters` | `epsilon`、`conflict_cut`、`va_method`、`va_calibration_source`、`evidence_frame`、`conformal_scores`、`calibrator_kind`、`benchmark_protocol_sha256` |
| `source_masses` | 每路证据的质量分配（列表） |
| `ecod_decomposition` | ECOD 逐字段贡献 |
| `limitations` | 固定的两句说明 |

`open_world` 的字段与预测表列的对应（`liushao_engine.OPEN_WORLD_FIELDS`）：

| 快照字段 | 预测表列 |
|---|---|
| `state` | `ow_state` |
| `bel_unknown` | `ds_bel_unknown` |
| `drift_delta` | `drift_delta` |
| `alpha` | `risk_alpha` |
| `guarantee_status` | `guarantee_status` |
| `cluster_id` | `novelty_cluster` |

读取规则：

- 用 `.get` 读，容忍缺字段和不认识的字段。P1 中算法线可能往 `open_world` 里**追加**字段，`schema_version` 仍为 2；已有字段不改名、不改含义。
- schema 1 的旧快照按保存时的内容原样显示，不升级。旧快照里的 `不确定` 原样显示，**不映射**到四态。
- 历史记录只展示保存的值，不用当前引擎重算。

## 7. 用夹具开发

```python
from tests.engine_contract_fixture import with_contract_columns
pred = with_contract_columns(pred)            # 在任意预测表上补齐第 2、3 节的【已定名】列
pred = with_contract_columns(pred, drop=['guarantee_status'])   # 演练缺列降级
```

夹具里的值是按固定种子合成的，只保证列名、类型和取值范围符合本约定，**不是引擎的真实输出**。用夹具得到的任何数字都不得写进材料。

## 变更记录

| 版本 | 日期 | 内容 |
|---|---|---|
| contract-1 | 2026-11-12 | 首版。2.1、2.2 和第 6 节为已实现；2.3、第 3、4 节已定名待实现 |
| contract-7 | 2026-11-14 | 保证状态加入「不适用」，O1 采用 A24 的 k=3、reps=3 |
| contract-6 | 2026-11-13 | T12 + O1：`risk_alpha`、`guarantee_status`、`novelty_cluster`、`apply_alpha`、`alpha_for_budget`、`RISK_ALPHAS` 改为已实现；**新增列 `ow_event`**；`ow_queue` 可出现「风险控制」（开发集上门限都是 +∞，尚未出现） |
| contract-5 | 2026-11-13 | T11：`ow_state`、`ow_queue`、`ow_normal_reject`、`engine_rank`、`cp_p_normal_supervised`、`cp_p_normal_typicality` 改为已实现；`engine_review` 改为等于 `ow_queue` 非空。只有带 `states` 参数的引擎（T11 起）输出。列名、取值不变 |
| contract-4 | 2026-11-13 | T10：`guarantee_ess` 改为已实现；报漂移的批次上共形 p 值改为密度比加权（权重截断 10），`cp_p_*`、`cp_set`、`cp_set_size` 的取值随之变化。列名、类型不变 |
| contract-3 | 2026-11-13 | T12a：`ow_conflict_gate` 改为已实现（阈值为验证 A 半区 K 的 95% 分位）；`engine_review` 随之包含冲突门控入队的记录。列名、类型不变 |
| contract-2 | 2026-11-13 | T09：`drift_flag`、`drift_delta`、`drift_delta_rf/_lr/_typ` 改为已实现；补充小批次（< 200 条）的取值、只有带漂移模型的引擎才输出、不折扣的选择结果。列名、类型不变 |

### T10 v2 扩展

启用工件策略 t10-v2 时，B2 输出 b2_mode（weighted、unweighted、fallback）及 b2_fallback_reason。逐类条数、ESS、p 值下限、D_plus、delta_res、截断比例和失败原因保存在批次诊断。D_plus 包含测试点权重，两个截断诊断均不改变拒绝阈值。ESS 为工程筛查。

逐 α 的 B3 记录独立保存 loss_path、b3_weight_status、calculation_status、statistical_status 及 contract-7 display_status。B2 回退时以非加权判态重算 B3 损失。漂移下未验证权重与适用前提时，加权数值仅供诊断。可计算的匹配判态非加权门限用于尽力复核。既有 contract-7 不适用条件优先展示。预测表以 b3_calculation_status、b3_statistical_status 和 guarantee_status 记录当前 α。schema 2 快照可选地保存四个新字段，旧快照不补写。
