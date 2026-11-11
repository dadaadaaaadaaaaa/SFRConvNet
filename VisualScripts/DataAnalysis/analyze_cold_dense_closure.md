# 冷致密气体与 Exp2 闭合诊断

只读已有fine气体场与正式预测，使用已核验时标分析的26132帧来源清单。
Fiducial系列采用fine_sfr_v7，Model系列采用fine_sfr_v6。
每帧核对parent、物理时间、真实标量Y、数组维度、权威体积与质量守恒。

成星条件为T=93e/rho<4e4 K且1.02rho>1；质量比例与格子数量比例分别统计。
全域及r<30kpc内分别计算

\[
f_{cd}=M_{cd}/M_g,\quad q=\langle\sqrt{\rho}\rangle_{M,cd},\quad
S_{ff}=K M_g f_{cd}q,\quad K=0.025\times0.01\sqrt{32G/(3\pi)},\ G=112.3.
\]

q为代码密度的平方根。M为代码质量；乘2.5e7才是Msun。Sff已为Msun/yr。
Sff为连续极限、自由落体近似，不能替代Data-Set-11真值；逐帧比较其总量误差。
全域热CGM会影响f的分母，所以同时报告inner30口径。
LR气体以12×20守恒块归约，从平均rho/e重新判断阈值；
hidden_cd_fraction为HR满足阈值的气体质量中，位于LR不满足阈值粗块内的比例。

参考来自20261111闭合分析已保存的b10训练近邻；选择不使用HR指标。
每折按实际train身份单独检验特征min/max和p01/p99覆盖；
后者是中心区间外比例，不是严格域外概率。按预测parent连接HR物理量。
报告全部匹配和输入距离<=0.5子集、已知运行test与留出out。

三因子乘积用对称Shapley代数分解为质量、比例、密度矩三项，三项和严格等于Sff差。
真实闭合差另扣B差，并保留真实Y差与近似Sff差之间的残差。
这是固定配对的描述性代数归因，不是因果干预；改变区域会改变M和f的分工。
零成星质量时q置零只为恒等式，不能将其当作实际测得的致密气体密度。

```powershell
D:/conda_envs/physics_ai/python.exe -m unittest tests.test_cold_dense_closure -q
D:/conda_envs/physics_ai/python.exe VisualScripts/DataAnalysis/analyze_cold_dense_closure.py --output-dir tmp/cold_dense_smoke_20261111 --smoke
D:/conda_envs/physics_ai/python.exe VisualScripts/DataAnalysis/analyze_cold_dense_closure.py --output-dir VisualScripts/DataAnalysis/outputs/cold_dense_closure_20261111
```

smoke每组3帧，不生成正式覆盖或分解结论。主输出目录必须不存在。
`--reuse`仅复用已完成的逐帧parquet，重算汇总与图；不再扫描fine。
新增折可用 `--cases fiducial4 --reference-dir <该折闭合分析目录> --moments-dir <已有五组逐帧parquet目录> --output-dir <新的目录>`，
只重新连接该折实际训练清单与正式预测，不覆盖旧分析。`--moments-dir`不能与`--reuse`或`--smoke`同用。
逐帧parquet及配对csv.gz仅本地保存。主要输出为galaxy_summary、ff_validation、fold_coverage、paired_decomposition、errors_by_fraction和CDF图。
errors_by_fraction按每折实际训练质量比例min/max分层，并另列Y超过该折训练Y最大值的高SFR子集。
原始HR逐格SFR本地fine未保存，本次不将重建分支当成真实逐HR格点标签；标签是已保守归约的真实总量。

四折对照入口为 `compare_cold_dense_folds.py --output-dir <新的目录>`，来源在其SOURCES中显式登记。
它保留冷致密比例相对参考变化超过±5个百分点的固定分层、训练确定门槛下的闭合上调/下调分层、
0.5 Gyr绝对时间分箱，以及相同UID在留出与已知运行test两种角色下的误差对照。
跨折角色比较核对parent、Y/B/time，不将不同帧或不同模型的结果当成同一训练种子的受控消融。
