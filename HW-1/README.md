# 作业一 文本分类实现

使用课程提供的 NYT 数据完成 7 组分类实验：Binary BoW、Word Frequency、TF-IDF、GloVe 6B 100d、AG News Word2Vec、NYT Word2Vec、BERT-base-uncased。

## 文件说明

- `code/experiment.py`：数据校验、统一划分、训练、验证集选模、测试集评价。
- `code/download_assets.py`：下载指定预训练资源，验证 GloVe SHA256。
- `code/plot_results.py`：由真实结果生成对比图、混淆矩阵和训练曲线。
- `code/audit_results.py`：从预测重新核对七组指标，并核查精确重复文本的影响。
- `code/test_experiment.py`：划分、分词、词表隔离、均值池化和评价指标检查。
- `results/`：划分索引、数据统计、完整指标、参数、环境和图表。
- `report/main.pdf`：实验报告。该目录其余文件均被 Git 忽略。

## 安装与运行

在仓库根目录运行，建议 Python 3.12。CPU、Apple MPS、NVIDIA CUDA 均有对应路径。CUDA 环境可根据驱动安装 PyTorch 官方提供的相应版本；跨设备训练结果可能存在小幅差异。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r HW-1/requirements.txt
python HW-1/code/fix_gensim_blas.py
python HW-1/code/download_assets.py
PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false python HW-1/code/experiment.py --task all
python HW-1/code/plot_results.py
python HW-1/code/audit_results.py
python -m unittest discover -s HW-1/code -p 'test_*.py'
```

`fix_gensim_blas.py` 需要本机 C 编译器（macOS Command Line Tools、Linux GCC 或 Windows MSVC）。它针对 Gensim 4.4.0 的 BLAS 点积指针 `except -1` 声明进行最小修复，改为 `noexcept`，避免将合法的 -1 点积误当异常；仅重编译当前虚拟环境的 Word2Vec 扩展，源文件保存在被忽略的 `assets/`，不改变 Skip-gram 算法或训练参数。正式结果使用此修复后的扩展。

`requirements-lock.txt` 记录本次运行的完整依赖。下载器默认使用课程镜像中的原版 GloVe 6B 100d 文件，校验 SHA256 为 `95dde4dfd627ab26608d33e76d1195ec059734bd29089ea52cadb08d07c64544`；也可传 `--glove-source stanford` 下载 Stanford 官方完整压缩包。BERT 来自 `google-bert/bert-base-uncased`，固定模型快照 `86b5e0934494bd15c9632b12f734a8a67f723594`。这些资源只缓存到 `assets/`，首次运行需要联网。

分项运行示例：

```bash
python HW-1/code/experiment.py --task bow
python HW-1/code/experiment.py --task glove
PYTHONHASHSEED=42 python HW-1/code/experiment.py --task w2v_ag
PYTHONHASHSEED=42 python HW-1/code/experiment.py --task w2v_nyt
TOKENIZERS_PARALLELISM=false python HW-1/code/experiment.py --task bert --device auto
```

可通过 `--glove PATH` 指定原版 GloVe 文本，通过 `--model-path PATH` 指定已下载的 BERT 快照。`--device` 支持 `auto/cpu/mps/cuda`，默认 batch size 为 32；显存不足时可用 `--batch-size 16`，但这会改变批次和优化轨迹。完整 BERT 实验始终训练 3 epochs、输入长度固定 64，然后选择验证集 Macro-F1 最佳轮次，仅对该模型评价测试集。

## 划分与实验约定

1. 作业要求给出 80%/10%/10% 比例，种子值固定 seed=42，以 `np.random.RandomState(42).permutation(N)` 打乱，按 `floor(0.8N)` 与 `floor(0.9N)` 分段。三组样本量为 9,215 / 1,152 / 1,152。
2. `results/split.json` 保存原始 CSV 的 0 起始行索引（不计表头）、SHA256 和种子。所有实验使用这些索引；原始文件或种子改变时拒绝覆盖现有划分，可用 `--output NEW_DIRECTORY` 另建实验目录。
3. 词袋词表、IDF、StandardScaler 仅在训练集拟合。NYT Word2Vec 也只使用训练集文本。AG Word2Vec 使用全部 90,000 条 AG 文本，不使用 NYT 验证/测试文本训练词向量。
4. LR 在验证集上按 Macro-F1 从 `C ∈ {0.01, 0.1, 1, 10}` 选参；同分保留较小 C。词向量使用 100 维向量的有效 token 出现次数均值，OOV 跳过，全 OOV 文档置零，再用训练集统计量标准化。
5. 作业文字称“三种”词袋方法，但仅列举 Binary 和 Word Frequency，故第三组补充 TF-IDF；报告明确说明这一解释。
6. 原始数据有 72 条重复文本记录，统一随机划分后 8 个相同文本跨训练/测试集。为保持指定实验一致性，主结果保留原始数据；报告说明该限制，并另外核查移除测试集中已在训练集出现文本后的指标。该核查不用于选模型。
7. 固定随机种子、Word2Vec 单工作线程与稳定词哈希；GPU 算子和库版本仍可能导致少量数值差异。

每组 JSON 保存测试集 Accuracy、Macro-F1、每类 precision/recall/F1、混淆矩阵以及验证选参轨迹。`results/predictions/` 的逐样本预测与运行日志保留本地，Git 忽略；运行程序可重新生成。图表中的耗时是特征生成、验证选参与评价总墙钟时间，多个任务并行运行时不用于严格速度排名。

## 参考资源

- [作业指定 BERT 模型](https://huggingface.co/google-bert/bert-base-uncased)
- [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)
- [GloVe 6B 100d 课程镜像](https://huggingface.co/datasets/SLU-CSCI4750/glove.6B.100d.txt)

## 本次实测结果

| 方法 | Test Accuracy | Test Macro-F1 |
|---|---:|---:|
| Binary BoW | 98.70% | 97.11% |
| Word Frequency | 99.22% | 98.07% |
| TF-IDF | 99.13% | 98.03% |
| GloVe 6B 100d | 98.87% | 97.56% |
| Word2Vec AG | 98.35% | 96.37% |
| Word2Vec NYT | 98.96% | 97.68% |
| BERT | 97.92% | 95.30% |

