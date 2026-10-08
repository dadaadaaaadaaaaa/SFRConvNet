# 二维 fine 时间上下文迁移

## 1. 范围与前提

`migrate_fine_time_context.py` 将完整二维 format-version 6 fine 的绝对时间、日志上下文和
SFR supervision 迁移到独立目录。当前支持 `filename_index` 的盘星系 HDFRA 运行，必须
显式指定 `--galaxy`。主要用于 Fiducial/Fiducial4/Fiducial7 的 flat 时间修复，不支持
legacy fine、coarse、椭圆星系或三维 ATHDF。

目录名称 v7 表示本次修复产品版本；HDF5 的 `processed_format_version` 仍是 6。
不更改二维单位、字段映射、物理算子版本、Direct/Closure 接口或训练配置。

前提：旧 fine 必须保存完整 metadata、raw SHA256、源文件大小、坐标中心/边界、权威
体积、必要物理场、Data-Set-11 总 SFR 和 canonical supervision。迁移不能修复旧 fine
本身的字段串写、错误单位、损坏数组或遗漏标签；遇到这些情况应回到 raw 重建。

## 2. 数据处理与验证

- 原样保留全部气体场、坐标、边界、Data-Set-15 体积、真值 Y 和 target 单位。
- 复用新版 raw→fine 的原始文件排序、`_build_snapshot_schedule` 和 snapshot 筛选。
  需要挂载 raw 目录以枚举文件和核验大小，但默认不读取 HDF4 数组、不重新通读 raw 哈希。
- 每个运行只读取一次日志 Parquet，使用有序时间索引截取窗口，再调用共享
  `ll_dg_from_frame`；窗口平均、重复时刻平均、空窗口线性插值和端点规则不变。
- 替换全部日志上下文及诊断，明确恢复从 snapshot 保存的 Y，不能以日志量替代标签。
- 重算 `data_compatible` 与 `paper_physical` 两套逐格/总 baseline、signed closure、
  context 和 `physics_product_uid`；`parent_hr_uid/sample_uid` 保持不变。
- metadata 增加可选 `time_context_migration`（`migration_version=1`），保存旧 fine
  SHA256、日志 SHA256、旧时间、筛选/排除清单、时间标签验证方式。schema/format 不升版。
- 每帧保存后执行 HDF5 round-trip、UID/closure 检查及物理内容逐元素一致性检查。

默认只复用旧 fine 的 raw 内容哈希，并检查当前 raw 文件大小；这不能发现“同大小篡改”。
`--verify-raw-time` 额外通过 pyhdf 检查 Data-Set-11 的 `AT TIME=`，也不等于全字段/全哈希
审计。需要更强保证时，应对少量帧直接 raw→fine 对照，或另外开展全量原始数据核验。

## 3. 安全与续跑

输入/输出必须是不同的单层目录。脚本没有覆盖选项，也不能覆盖 raw、旧 fine 或已有
raw 重建产品。默认续跑只跳过来源、旧 fine 哈希、日志哈希、时间映射、字段/target、
筛选契约及迁移版本匹配的输出；否则停止。`--no-resume` 对任何同名输出都报错。
抽样时也检查目录中已有产品，拒绝混写不同日志或筛选契约。

先写同目录临时 HDF5，round-trip 成功后用硬链接独占发布，目标已存在则失败，最后删除
仅由本次创建的临时文件。输出文件系统须支持同目录硬链接（本机 NTFS 已验证）；不支持
时会停止，不回退为覆盖写入。中途失败已成功发布的帧保留，下次可按相同参数续跑。

默认跳过排序后前 46 个 raw 文件，不是按现有 fine 枚举位置判断。显式传入
`--exclude-snapshot INDEX` 后，仅执行用户排除清单，取代默认规则；可重复。Python API
传 `exclude_snapshot_indices=[]` 也可明确保留全部帧。包含/排除交集直接报错。
全量迁移要求当前应纳入的 raw 帧都有旧 fine；缺少任意 fine 时停止，不静默缩小样本集。

## 4. 命令

从项目根目录运行。迁移本身需要 numpy、pandas、h5py 和 Parquet 引擎；默认不需要 pyhdf。
本机已有 `D:/conda_envs/physics_ai/python.exe` 可运行；先激活正确环境，不必修改 base。

少量 smoke（独立目录，额外核验 HDF4 时间标签）：

```powershell
python raw_data_processing/migrate_fine_time_context.py `
  --galaxy disk_galaxy_fiducial `
  --input-fine-dirname fine_sfr_v6 `
  --fine-dirname fine_timefix_smoke `
  --snapshot-index 46 --snapshot-index 999 --snapshot-index 1000 `
  --verify-raw-time
```

全量迁移三个运行（本次尚未执行）：

```powershell
python raw_data_processing/migrate_fine_time_context.py `
  --galaxy disk_galaxy_fiducial `
  --galaxy disk_galaxy_fiducial_4 `
  --galaxy disk_galaxy_fiducial_7 `
  --input-fine-dirname fine_sfr_v6 `
  --fine-dirname fine_sfr_v7_timefix
```

现场检查发现 Fiducial 的 `fine_sfr_v7` 已有 raw 重建产物（核验时 261 帧），不要向其混写
迁移产品。因此上述命令使用另一个独立目录 `fine_sfr_v7_timefix`，旧 v7 保留不动。

随后构建 coarse 时指定 `--fine-dirname fine_sfr_v7_timefix`；Model0.75/1.1 和 torus 不因本次时间
修复而迁移 fine，可继续读取各自已有 fine，统一输出到新的 coarse 目录。再重新生成
CachePT/训练统计量并重训。旧结果不能因为修复数据已经生成就自动成为修复后的实验结果。

## 5. 本次验证（2026-10-07）

使用现有 `D:/conda_envs/physics_ai/python.exe`，未安装或变更任何环境依赖。
真实 smoke 取三个运行各 `[46,999,1000,9531]`，共 12 帧，覆盖早期、重启段两侧和末期。
修正时间分别为 `[0.62,1.81125,1.81375,12.4875]` Gyr。

迁移与独立 raw→fine 对照：物理数组、坐标/边界、体积、Y、target 单位和身份逐元素相同；
全部日志量在 `rtol=1e-12, atol=1e-14` 内一致；两套逐格 baseline、baseline/closure
及 `physics_product_uid` 一致。原始时间标签全部通过。相同参数续跑跳过 12 帧，新写出 0。

两组 fine 分别生成 12×20 R_h coarse，各 12 帧；比较物理数组、两套逐格/总 B/C 和 UID
一致，三个迁移 coarse 目录均可由完整 metadata 的 `GalDataSet` 加载。粗化由现有入口执行
其守恒检查。未生成正式 CachePT，未进行正式训练或全量迁移，未测量全量速度。

每个运行的测试产品保存在以下独立子目录（不是生产目录）：

- `fine_timefix_migration_smoke_20261007`
- `fine_timefix_raw_reference_smoke_20261007`
- `coarse_timefix_migration_smoke_20261007`
- `coarse_timefix_raw_reference_smoke_20261007`

6 个回归模块合计 39 项通过，包括零 Y/负 closure、窗口/重复时间/插值/端点、筛选、
拒绝覆盖/混写、写入失败清理及现有 raw→fine/coarse 契约。3 个相关 Python 源文件的
内存编译检查通过（未写入 pyc）。回归命令见研究计划第 33 节。
