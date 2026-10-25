# RAG Retrieval Lab

**中文技术资料问答：BM25 + BGE 语义检索 + RRF + Cross-Encoder 重排 + 本地 Qwen 生成。**

本项目用一条可追踪的 RAG 链路研究三个问题：关键词和语义检索如何互补，重排序是否改善结果，以及“找对资料”和“答对问题”为何是两种评估。适合学习 AI 应用开发与检索算法实践。

主程序包含详细中文注释、向量形状说明、排名融合算例与错误归因提示。实现使用预训练模型推理，没有在本项目中训练或微调模型。代码由 AI 辅助开发，学习者的实践记录与已完成的实现分别记录。

## 工作流程

```text
Markdown → 标题/段落切分 → BM25关键词召回 ─┐
                         BGE语义向量召回 ─┴→ RRF融合前8条
                                            ↓
                                 Cross-Encoder重排序
                                            ↓
                           前3片段 → 分数门控 → Qwen回答与引用
```

- 文档与问题通过同一 BGE 模型编码，向量归一化后使用 NumPy 点积作精确搜索。
- BM25 与向量检索各取前8条；RRF合并后保留8条，重排后选3条作为回答上下文。
- 固定比较 BM25、纯向量、混合检索和混合加重排，记录逐题结果。
- 生成阶段保存原始回答、引用、证据与耗时，不以人工改写答案替换失败输出。
- 小规模本地 CPU 原型，无需付费 API；没有部署生产服务或接入向量数据库。

## 快速开始

建议 Python 3.12。首次运行需要联网下载模型，预留数 GB 磁盘空间。

```bash
git clone https://github.com/zzp13553046106/rag-retrieval-lab.git
cd rag-retrieval-lab
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./run.sh prepare
./run.sh evaluate
./run.sh demo
./run.sh ask '为什么搜索错误编号时保留关键词检索？'
```

`run.sh` 优先使用项目 `.venv`。`RAG_PYTHON` 可指定解释器，`RAG_CACHE` 可指定模型与向量缓存。本机原始工作目录另有已配置的环境，脚本会自动识别。

`model-lock.json` 记录实际模型提交版本。`requirements.txt` 固定直接依赖，`requirements-lock.txt` 记录原环境新增依赖；新环境完整依赖解析仍由 pip 完成。修改资料、向量模型版本或向量编码截断长度会触发文档向量重建。

## 首轮检索结果

语料是12份自编合成文档、36片段，描述虚构的 Atlas 研发平台。问题与标签均为 AI 初稿，尚待人工审核：开发集8题，保留测试集32题，其中28题有答案、4题无答案。下表的检索指标仅对28道有答案测试题计算。

| 方案 | Recall@3 | MRR@3 | nDCG@3 | 检索链路 P50 |
|---|---:|---:|---:|---:|
| BM25 | 0.750 | 0.696 | 0.694 | 0.2 ms |
| BGE向量 | 0.839 | 0.768 | 0.777 | 3.1 ms |
| RRF混合 | 0.857 | 0.798 | 0.802 | 3.3 ms |
| 混合 + 重排 | 0.946 | 0.875 | 0.882 | 41.5 ms |

以上是 CPU 4线程、预热后单轮测量。混合阶段累计两路检索与融合时间，重排方案再累计重排时间；不含模型加载、建库或回答生成，不代表并发压测或生产吞吐。

**这些结果是内部合成数据上的检索实验，不是最终答案正确率或业务提升。** 资料与问题由同一 AI 起草，保留测试集仅表示本轮未用于调参，不是外部独立基准。

完整结果：[检索实验报告](reports/实验报告.md) · [逐题排序](reports/retrieval-details.json) · [模型回答案例](reports/回答案例.md) · [首轮错误分析](reports/首轮问题分析.md)。

## 已观察到的失败

- **融合丢失正确资料**：q025的正确片段在向量召回第5名，但融合截断后被挤出，后续重排无法找回。
- **资料正确但答案错误**：q016提供了语义检索原文，0.5B生成模型仍混淆BM25与语义检索。
- **标签也需要审核**：q019标为需要两个证据，但单个片段可能已经包含完整答案，需要审核可替代证据。
- **拒答有代价**：开发集阈值拦截了4道无答案测试题，也误拦了3道有答案题；小样本不足以证明可靠拒答。

这些失败保留在报告中，供下一轮改进与复测。引用ID合法性检查只验证ID是否存在，不能证明引用支持结论。

## 阅读与学习

1. [第一课：追踪 search() 的四份排名](learning/01_search.md)。先理解输入输出，再看公式。
2. [技术实践与简历记录](你的参与与简历.md)。围绕代码修改、实验与错误分析记录实际贡献。
3. [rag.py](rag.py)：主要实现；`search → rrf/rerank → generate → metrics/evaluate → encode` 是推荐阅读顺序。

建议每次只完成一个小改动：先记录假设，在开发集验证，再用 Git 提交保留配置、观察与结论。若已根据本轮测试题设计改进，下一轮需要新增未参与设计的测试题。

## 目录

```text
rag.py                 核心实现与中文教学注释
run.sh                 本地运行入口
model-lock.json        模型提交版本
requirements*.txt      Python依赖
data/docs/            合成知识文档
data/questions.jsonl  问题、参考答案、证据标签与数据划分
learning/              分步代码阅读练习
reports/               原始运行结果、汇总与错误分析
tests/test_core.py     核心检索与评测逻辑验证
```

文档里提及的权限过滤、缓存失效和服务版本管理是问题题材，并非脚本已实现的生产功能。本脚本实现文档切分、向量缓存、检索融合、重排、门控、生成与评测。

## 本地验证

```bash
.venv/bin/python -m unittest discover -s tests -v
```

现有23项验证覆盖多证据Recall、MRR/nDCG、RRF融合、错误码匹配、证据ID完整性、开发集阈值校准，以及空候选、无效配置、空问题、非有限分数、引用缺失、拒答标记和截断长度配置等回归情况。它们不下载模型；模型相关单元测试使用小张量和替身对象。

`embedding_max_tokens` 与 `reranker_max_tokens` 分别控制对应模型的输入截断长度；当前实现只支持CPU。问答输出的 `missing_citations` 标记非拒答结果是否缺少引用；`invalid_citations` 列出已经出现但不属于提供证据的编号。两者均不判断引用是否在语义上支持结论。

完整真实模型复测与保留的结果见 [完整测试记录](reports/完整测试记录.md)，首轮模型输出仍保存在原有 `reports/` 文件中。

报告中的 `run-manifest.json` 保留原运行的文件摘要与环境；后来添加注释会改变源码摘要，不覆盖原实验记录。

## 参考资料

- [Sentence Transformers：Retrieve & Re-Rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
- [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [多语言 Cross-Encoder](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
