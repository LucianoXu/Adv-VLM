---
type: project
status: active
origin: RUB Responsible AI (SS 2026) course project, revived 2026-09-04 for submission
repo: https://github.com/LucianoXu/Adv-VLM
target: AID 2026 @ PRDC, Regular Paper Track
deadline: 2026-09-11        # AID 2026 submission, AoE
last-updated: 2026-09-04
---

# Adv-VLM — Adversarial Examples on Vision-Language Models

原为 RUB Responsible AI 课程项目（2026-06 至 07），单作者。2026-09-04 从 GitHub 拉回 vault，
准备修改后投稿。Vault 之前没有此项目的记录（课程材料在 `4-archive/20260810_ResponsibleAIExam2026/`）。

## 内容

- 模型：CLIP ViT-L/14@336 编码器；LLaVA-1.5-7B 与 VisualRWKV-6-7B 两个共用该编码器的 VLM。
- 任务一：Imagenette 分类的对抗样本（PGD，RMS ≤ 0.03），交叉评估编码器攻击 vs 端到端攻击的迁移性。
  结论：编码器级攻击迁移更好，端到端攻击过拟合到 LM backbone 和 prompt。
- 任务二：PKU-SafeRLHF 上的通用越狱图像。LLaVA 合规率 18% → 56%，但不迁移到 VisualRWKV；
  对抗图像中出现 14×14 patch 网格结构。
- 报告：`report/adv-multimodal-llm.tex`（NeurIPS 2019 样式，约 8 页）。
- 实验在 Raven 集群（A100-40G）；`scripts/run_config.sbatch`，`configs/*.yaml`。

## 投稿决定（2026-09-04）

**目标：AID 2026（2nd IEEE PRDC Workshop on AI Dependability），Regular Paper Track。**
选 Regular 而非 R&D track：R&D track 要求「有资助项目的在研阶段性成果」，本文是已完成工作。

CFP 已于 2026-09-04 从 https://aid2026-workshop.github.io/ 核实：

| 项 | 要求 |
|---|---|
| 提交截止 | 2026-09-11 AoE，EasyChair `aid2026` |
| 通知 | 2026-10-09 |
| Camera-ready | 2026-10-20 AoE |
| 篇幅 | 「At most 6 pages, including figures, tables, and references」，双栏 IEEE conference 模板 |
| 附录 | **CFP 未提供任何 appendix / supplementary 例外**，附录同样计入 6 页，所以内容只能砍不能挪 |
| 评审 | 双盲，「Remove author names, affiliations, and any identifying information」 |
| 论文集 | IEEE Computer Society Press，EI 索引，archival |
| 会议 | PRDC 2026，香港，2026-12-02 至 04，需线下报告 |

## 执行范围（2026-09-04 确定）

两个决定：**实验做 P0 全部 + P1 防御**；**论文重定位为 dependability 论文**。

重定位的主线：漏洞住在 VLM 栈的哪一层，以及这对部署共享视觉编码器的流水线意味着什么。
两条现有结果支持的论断——共享编码器是感知完整性的单点故障（编码器级攻击跨完全不同的
backbone 架构迁移）；安全对齐不随编码器继承（端到端越狱图像不跨 backbone），因此安全必须
按 backbone 而非按编码器验证。

### 环境状态（2026-09-04 已重建完成）

`/u/yinxu/work/Adv-VLM` 在 Raven 上曾被删除，现已从 vault 重建。已核实：python 3.13.5、
torch 2.14.0+cu130、torchvision 0.29.0+cu130、transformers 5.16.1、datasets 5.0.1；
VisualRWKV 7B（15,887,196,222 B，与 HF 远端逐字节一致）与 1B6 检查点均就位；
`dataset/imagenette2-320` 13,394 例；`PKU-SafeRLHF` 符号链接可用。
Smoke test（job 29922816）三个模型精度均为 1.0，**WKV6 CUDA kernel 重新编译成功**
（sm_80，`-D_T_=2048`）。稳态吞吐：CLIP 0.063 s/图、LLaVA 0.084 s/图、VisualRWKV 0.673 s/图。

VisualRWKV 越狱攻击新代码已在 GPU 上验证（job 29922865，7B 与 1B6 两轮全部通过）：
梯度可达图像（338688/338688 非零且有限）、经 quantize STE 同样可达、截断守卫正确、
5 步 loss 单调下降 3.0762→2.8872、检查点文件完整；**峰值显存 18.82 GiB**（估计值为 25–27 GiB）。

三个运维教训（已固化进脚本）：

1. **绝不把本地 `results/` 同步到 Raven。** 本地仓库里提交着原始那次运行的结果，而
   `run.py` 用 `os.makedirs(..., exist_ok=False)`，同步过去会让 35 个评测任务在排队等待之后
   一接触自己的输出目录就中止。这个坑踩了两次（第二次是子 agent 的 rsync 又把它们带回去），
   两次都被 `scripts/preflight.py` 挡住。此后一律用 `scripts/sync_to_raven.sh`，它排除
   `results/`、`.env`、`.venv/`、`dataset/`、`ckpt/` 且不使用 `--delete`。
2. **preflight 必须只校验本次提交的 stage。** 校验整条流水线的话，早期 stage 一旦跑过并
   合法占用自己的输出目录，后续任何提交都会被永久挡住。
3. **不要在 Raven 上 `git commit -a`。** 同步排除了三个被 git 跟踪的 `.key` 文件，
   Raven 的 `git status` 因此把它们显示为已删除。

原始那次运行的结果在 Raven 上归档于 `results/_original-run/`，本地由 git 保留。

Raven 的 HF cache（243G）一直完整保有 `llava-hf/llava-1.5-7b-hf`、
`openai/clip-vit-large-patch14-336`、`johnowhitaker/imagenette2-320`、
`PKU-Alignment/PKU-SafeRLHF`，所以只有 VisualRWKV 的 `.pth` 需要重下（已完成）。
计算节点无外网，全部 HF 资产必须预热在 cache 里；judge 打分只能在 login 节点跑。

四个对抗数据集（`results/*-attack-*/dataset`）当初未进 git（`.gitignore` 排除了 `dataset/`），
只有 eval 结果留存。重新生成的样本与原样本不会逐位相同（Adam + 非确定性 CUDA kernel），
因此**论文全部数字必须从新的一次完整 pipeline 重出**，不得新旧混用。

PKU-SafeRLHF 的 split 计数（2026-09-04 实测）：train 20881 条 unsafe pair / 11616 个 unique
unsafe prompt；test 2408 条 / 1354 个。攻击训练用 train，评测用 test，天然 held-out。

### 越狱攻击超参数：两次失败与最终定论（2026-09-04）

这是本次最费时间的一段，结论必须记下来，否则会有人重复踩。

原打算把越狱攻击从已发表设置（64 对样本池、100 步）放大到 256 对，理由是「一张图覆盖更多
请求」是更强的主张。**这个放大不但没变强，还让优化变得不稳定。**

| 设置 | 在 500 条 held-out test prompt 上的合规率 |
|---|---|
| 已发表设置（64 对、100 步） | 57%（clean 20%），与归档原图逐点复现 |
| 256 对、200 步 | 51% |
| 256 对、400 步 | **21.6%**（clean 20.2%，等于没效果） |

400 步崩掉之后，在**无污染验证集**上扫了 256/400 步那次运行的全部 checkpoint
（`dataset/saferlhf_val.jsonl`：400 条 train 侧 prompt，剔除全部 740 个出现在任一攻击池里的
prompt，重叠数已验证为 0）：

```
step     0    50   100   150   200   250   300   350   400
compl 19.0  23.0  27.5  36.0  37.5  22.5  45.0  43.5  12.0   (clean 18.5)
```

不是「有最优点的曲线」，而是**震荡轨迹**——Adam 在 lr 0.1、无界图像上根本没收敛。
之前在 step 200 读到的「还在上升」（100/150/200 → 40/44/51）只是一次局部上冲。
而且整条曲线的最高点 45% 仍低于已发表设置稳定给出的 57%。挑 step 300 等于在噪声轨迹上
挑一个运气好的点。

**最终定论：池大小与步数回退到已发表值（64 对、100 步、lr 0.1）。**
本次投稿相对原文的增量与这两个超参数无关，全部保留：3 个 seed（原文 1 个）、
500 条 test prompt（原文 100 条）、Wilson 区间、配对 McNemar 检验、第二个 judge、
以及 VisualRWKV 方向的反向攻击。

两条流程教训，已固化：
- **超参数不能靠推理定，也不能在 test 上定。** 前者错了两次；后者会让报告的数字带上乐观偏差。
  选择只在 `dataset/saferlhf_val.jsonl` 上做。
- **stage E 之前必须过验证闸门**（`configs/gate-validate.yaml`，约 4 分钟）。
  stage E 曾两次跑在未验证的攻击图上，第二次白扔了 1.5 小时生成。

### 新发现：越狱攻击本身不可复现（2026-09-05）

这是本次投稿最有价值的结果，而且是被迫发现的——原本只想复现原文的越狱数字。

**同一份配置、同一个 seed，跑四次得到两种结果。** 配置为 64 对样本池、300 步、lr 0.03，
seed 固定意味着初始随机图像与数据顺序完全相同，唯一的差异是 CUDA 非确定性算术。
在验证集上（judge 为 gpt-4.1，clean 基线 38.0%，归档原图 85.0%）：

| 第几次尝试 | 合规率 |
|---|---|
| lrsweep | 84.5% |
| s44 | 82.5% |
| s42 | 47.0% |
| s43 | 44.0% |

**双峰分布**：要么「抓住」达到 82–85%（与归档原图的 85.0% 几乎一致），要么只比 38% 的
clean 基线高一点点。四次里成功两次。中间没有过渡。

lr 不是原因。最初以为是 lr 0.1 步长过大，扫描后改为 0.03；但 0.03 下同样出现 47% 与 84.5%
两种结果。降低步长并不能消除。

这解释了 09-04 一整天所有令人困惑的测量：51%、21.6%、19.5% 全都是失败的抽样，不是代码
错误也不是超参数错误。**原文报告的 56% 是一次成功抽样。**

对论文的意义：单次运行的越狱合规率不是一个测量值。这直接限定了原文的「18% → 56%」，
而且对一个 dependability 会场来说，攻击方法本身的不可复现性是一等结果。

**实验设计因此改变**：不再假装单次运行就是「这个攻击」。做法是跑 K 次、在验证集上选最强的
一张图、再在 held-out test 上报告其迁移性；同时报告 K 次之间的散布作为可复现性结果。
选最强的攻击是必要的——否定性的迁移结论只有在攻击足够强时才有意义，而真实攻击者本来就
可以重试并保留最好的一次。

### 另一发现：LLM judge 的评分极不稳定（2026-09-05）

对**同一批生成文本**（归档原图在验证集上的输出）用四个 judge 打分：

| judge | adv | clean | delta |
|---|---|---|---|
| gpt-4.1-2025-04-14 | 85.0% | 38.0% | +47.0 |
| gpt-4o-2024-11-20 | 83.0% | 36.0% | +47.0 |
| gpt-4o-mini-2024-07-18 | 50.5% | 17.0% | +33.5 |
| gpt-4.1-mini | 6.0% | 5.0% | +1.0 |

同一段文字，合规率从 6% 到 85%。**报告出来的数字取决于用哪个模型打分，甚至超过取决于攻击本身。**
gpt-4.1-mini 被直接淘汰：三个独立 judge 和原文都测到的效应，它完全测不出来。
两个大模型互相吻合（都是 +47），gpt-4o-mini 居中。

教训：judge 必须在已知有效的生成文本上验证过才能用，不能只靠几个人造样例。
（gpt-4.1-mini 通过了三个人造样例——明确拒绝、明确合规、道德说教式回避——却在真实输出上失效，
真实输出更含糊，它把「带保留的合规」判成了拒绝。）

论文中所有上报数字统一用 gpt-4o-mini-2024-07-18（原文的 judge，可比），
交叉核对用 gpt-4.1；诊断性的选择实验用 gpt-4.1（配额充裕且灵敏）。

### 实验

| 编号 | 内容 | 状态 |
|---|---|---|
| P0-A | 环境重建 + 四个对抗数据集与全套 xeval 重出 | 环境完成；stage A 运行中 |
| P0-B | 为 VisualRWKV 实现 `saferlhf_attack`，补齐越狱迁移 2×2 矩阵 | 代码完成并已 GPU 验证 |
| P0-C | 越狱评测扩容：500 条 held-out prompt、3 seed、Wilson CI、McNemar、双 judge | 已排队 |
| P0-D | 扰动预算扫描 eps ∈ {0.01,0.02,0.03,0.06,0.1}，迁移率曲线 | 已排队 |
| P1-E | 推理期输入变换防御（JPEG / blur / bit-depth / resize-pad），两个任务都测 | 已排队 |

全部 11 个 Slurm 作业已提交，两条互相独立的链：
分类链 29922996→29922997→{29922998, 29922999, 29923000, 29923001, 29923002}；
越狱链 29923148→29923149→{29923150, 29923151}。

**统一攻击空间为 `resized`**（uint8 直通估计）。原稿主表同时报 image01 与 resized 两列，
压缩时保留了 resized；因此新增的预算扫描与防御实验也一并改为 resized，全文只用一个
威胁模型——扰动能过量化、对抗样本是真正的 8-bit 图像文件，也正是输入变换防御真正作用的对象。

砍掉并写入 future work：CLIP 对抗训练与 robust encoder 微调（现成 robust CLIP 是 ViT-L/14@224，
与本文的 336 分辨率不匹配，等于要自己训）；针对输入变换防御的自适应攻击（BPDA / EOT）；
更多 VLM 与非 CLIP 编码器。

### 本次新增的代码与配置

- `src/defense.py` — 推理期输入变换 + `DefendedVLM` / `DefendedCLIP` 包装器。攻击路径故意
  抛 `NotImplementedError`，避免在不可微变换上算出与部署流水线不对应的梯度。
- `src/run.py` — 三个 eval 任务接受 `defense:` 配置块；`VLM-SafeRLHF-gen` 新增
  `include_clean` 开关（clean 基线只依赖 eval 模型与 seed，2×2×3 矩阵原本会重算 12 次，
  RWKV 近乎串行解码，这一项省下数小时）；grade 任务同步支持缺失的 clean 条件并记录原始计数。
- `configs/budget-sweep-{attack,eval}.yaml` — P0-D，5 + 15 个任务。
- `configs/jailbreak-attack-llava.yaml`、`jailbreak-{gen,grade}.yaml` — P0-C，3 + 12 + 16 个任务。
- `configs/defense-{classify,jailbreak,jailbreak-grade}.yaml` — P1-E，30 + 5 + 5 个任务。
- `scripts/aggregate_jailbreak.py` — Wilson 区间、McNemar 精确检验、Cohen's kappa，输出
  `matrix.json` 与 `matrix.tex`。三项统计均已对照教科书数值验证。
- `scripts/aggregate_budget.py` — 输出 `budget.csv` 与 pgfplots 图片段（已在 IEEEtran 下编译通过）。
- `scripts/aggregate_defense.py` — 防御表，并把自己的 undefended 行与预算扫描的 eps030
  点交叉核对（两者评的是同一个数据集且不加变换，必须逐位相符）。已用植入的不一致验证过能报错。
- `scripts/make_figures.py` — 从新结果重新生成报告图。原图渲染自已被删除的数据集，
  否则图与表描述的会是两次不同的运行。分类样本图仍沿用原图那四个 Imagenette 索引。
- `scripts/preflight.py` — 提交前校验：task_type、必填键、输入路径、defense 块、
  以及最要命的 output_dir 是否已存在。已作为 `submit_pipeline.sh` 的硬闸门。
- `scripts/integrate_results.py` — 把生成的片段一键装进 `aid2026.tex`，并编译后报页数、
  错误、溢出框与残留占位符。**整条装配路径已用合成片段全程演练过：6 页、0 错误、
  0 未定义引用、无溢出框**，所以真实数字若装不进，原因只会是数字本身的体积。
- `scripts/sync_to_raven.sh` — 唯一允许的同步方式（见上文运维教训 1）。
- `scripts/submit_pipeline.sh` — 六个 stage 的 Slurm 依赖链，preflight 硬闸门，`DRYRUN=1` 可空跑。
- `report/INTEGRATION.md` — 哪个片段替换哪个占位符、需要手工核对的散文数字、提交前检查清单。
- 顺带修掉两处：`scripts/download_data.py` 缺 `sys.path` 引导（无法独立运行）；
  `requirements.txt` 把 transformers 上限钉到 `<5.18`（`llava.py:275` 的
  `cache.crop(L_prefix)` 正参数用法在 5.18 被移除，5.16 已警告）。

judge 用带日期的快照固定：主 judge `gpt-4o-mini-2024-07-18`（与原稿一致），
交叉核对 judge `gpt-4.1-2025-04-14`。OpenAI key 于 2026-09-04 验证可用。

### 写作

- `report/aid2026.tex` — IEEEtran conference 双栏新稿（原 `adv-multimodal-llm.tex` 保留不动）。
- 匿名化：作者栏、`\thanks` 中的 GitHub 链接、「RUB Responsible AI Course Project」副标题、
  PDF metadata；「AI Usage Declaration」独立章节撤掉。
- 参考文献 28 → 约 18 条（`report/references-aid.bib`）。
- 压缩顺序：CLIP 双塔科普段 → image01/resized 重复表列 → 4stages 图 → teacher-forcing 图与
  prompt 表合并 → case study 四段 Q&A 砍到两段 → judge prompt 进脚注。
- `report/EXTENDED-NOTES.md` 记录砍掉的内容，供投稿后的 arXiv 扩展版使用。

待办：
- [x] Raven 环境重建完成并通过 smoke test
- [x] VisualRWKV `saferlhf_attack` 实现并在 GPU 上验证
- [x] IEEEtran 6 页新稿 + 匿名化 + 装配路径演练
- [x] 分类侧全部跑完（xeval 35 项、预算扫描、防御 30 项），并复现原表（1024 例列内误差 ≤1.3 点）
- [x] 越狱 2×2 迁移矩阵（4 格 + 2 格交叉 judge），白盒两格 p<1e-18，两个迁移格均不显著
- [x] 三个 aggregator + 图重生成 + `integrate_results.py` 装配
- [x] 手工核对：tab:xeval、正文数字、abstract、conclusion 全部对齐重跑值
- [x] 新增 Section VI「测量本身比系统更不可靠」——攻击不可复现 + judge 分歧
- [x] 匿名性自查（源码无标识串，PDF metadata 干净，0 未定义引用）
- [ ] 防御表的两列 SafeRLHF 数字（等 gpt-4o-mini 日配额重置，约 5000 次调用）
- [ ] 投稿前人工通读一遍（正文与表格的数字耦合处最容易出错）
- [ ] EasyChair 提交（需本人登录）

## 当前论文状态（2026-09-05 收尾，待通读与提交）

`report/aid2026.tex` — **6 页、0 错误、0 溢出框、0 未定义引用**，10 条参考文献全部被引，
abstract 161 词，匿名性通过。PDF 在 `outbox/AID2026-Submission.pdf`。

标题：*Shared Encoder, Unshared Safety: Adversarial Images across CLIP-Based VLMs*

正文用原稿的章节与措辞（Introduction / Related Work / Background and Methods /
Attacking Image Classification / Adversarial Examples for Jailbreak /
Defenses and Discussion / Conclusion），三张表由 `\input` 引入 aggregator 生成的片段。

核心结果：

| | 攻击自 LLaVA | 攻击自 RWKV |
|---|---|---|
| LLaVA 上评测（clean 20.0%） | **52.2%** (+32.2, p=5e-32) | 23.4% (+3.4, n.s.) |
| RWKV 上评测（clean 29.0%） | 27.2% (−1.8, n.s.) | **53.6%** (+24.6, p=3e-19) |

分类侧对照：编码器级攻击在全部五个扰动预算上都迁移到两个 backbone。感知会迁移，安全不会。

两个方法学发现：
- **攻击结果不由配置决定**。同配置重复跑呈双峰（7 次成功 4 次）。固定 seed 也无用：
  同 seed 两次运行图像相关系数仅 0.24，合规率 84.5% vs 47.0%。跨运行的攻击图之间
  |r|<0.004，与两张无关随机图的距离同量级——优化是在一个宽分布上抽样，不是收敛到一个解。
  （见 `scripts/compare_attack_images.py`）
- **judge 分歧极大**。同一批文本四个 judge 给出 6%–85%，保留的两个 κ=0.23–0.27。

已写明的局限：无自适应攻击（BPDA/EOT）；每个迁移格只用了一张选中的攻击图，因此
"不迁移"是对这些攻击成立，而非对该攻击方法一般成立。

`report/NUMBERS.md` 是全部数字速查表。

## 评估备忘

原稿是 workshop 级别：单作者课程项目，两个 2023–24 年的 7B 模型，样本量小（LLaVA-Adv 128 张、
越狱训练 64 条），主要结论与 Zhao 2023 / Schaeffer 2025 一致。新意在 RWKV 线性注意力 backbone
作为迁移目标、「分类攻击迁移而越狱不迁移」的漏洞定位、以及 14×14 网格伪影。
本次投稿版补的是：越狱迁移的反向（RWKV→LLaVA）以完成 2×2 矩阵、统计显著性、预算扫描曲线、
以及一个缓解措施基线。
更高的候选（DLSP @ S&P 2027、AdvML @ CVPR 2027、SaTML 2027 workshops）都需要补防御实验和
非 CLIP 编码器的 VLM，且 AID 为 archival，后续扩展需有实质新内容。
