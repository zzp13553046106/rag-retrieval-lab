# 中文技术资料 RAG：混合检索、重排序与评估

一个可以拆开观察、运行对照实验的小规模学习项目。目标是支持简历技能栏和面试讲解：知道每一步为什么存在，能检查结果并作出判断。代码由 AI 辅助实现，用户的审核和实验判断另有任务卡。

## 实际链路

Markdown 按标题切块 → BM25 关键词召回 + BGE 中文语义向量召回 → RRF 融合前8候选 → 多语言 Cross-Encoder 重排 → 前3片段 → 本地 Qwen 生成并引用。

不需要付费 API。CPU 运行；模型初次下载需要联网和约数 GB 的磁盘空间。小规模向量直接使用 NumPy 精确搜索，未引入向量数据库或服务部署。

本机已经配置运行环境，可直接在本目录运行：

```bash
./run.sh evaluate
./run.sh demo
./run.sh ask '为什么搜索错误编号时保留关键词检索？'
```

移动项目后：使用 Python 3.12 创建环境并安装依赖。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./run.sh prepare
./run.sh evaluate
./run.sh demo
```

`RAG_PYTHON` 可指定 Python 解释器，`RAG_CACHE` 可指定模型与向量缓存。`model-lock.json` 固定实际使用的模型提交版本；首次运行没有锁文件时使用上游 main 并写入锁文件。`requirements-lock.txt` 是本机实际依赖版本记录。

## 数据与实验

- 12份自编合成文档、36片段，描述虚构的 Atlas 研发平台；没有接入企业真实材料。
- 40条 AI 起草的问题及证据标签：开发集8题，测试集32题；全部待用户审核。
- 测试集28题有答案、4题无答案。多证据题的 Recall 按所需证据数计算。
- 固定对照：BM25 / 纯向量 / 混合 / 混合加重排；最终上下文 top-3 一致，候选 top-8。
- 拒答门控阈值只用开发集校准；未根据测试错误调参。
- 保留测试集只是不参与本轮调参；题目与语料均由同一 AI 起草，不能视为外部独立基准。
- 延迟记录 CPU 4线程预热后对应处理阶段，未做重复负载测试，不代表服务吞吐。

## 看结果

1. `reports/实验报告.md`：四种方案的检索指标、延迟和变化案例。
2. `reports/retrieval-details.json`：每题的四路排序、相关性标签、重排分数与耗时。
3. `reports/回答案例.md`：六个实际运行案例，包括模型原样回答和待填人工评审。
4. `reports/answer-cases.json`：答案、提供的证据、引用 ID 检查和生成耗时。
5. `reports/首轮问题分析.md`：已定位的召回、融合、生成和标签问题。
6. `你的参与与简历.md`：你亲手完成的三个小任务及完成后的技能栏表述。

## 三个概念与代码的对应

|概念|一句话理解|对应位置|
|---|---|---|
|混合检索|既搜关键字，也搜含义，再合并排名|`BM25`、`Engine.encode`、`rrf`|
|重排序|让模型把问题和候选资料一起看，重新排先后|`Engine.rerank`|
|效果评估|用固定题目和正确证据，对比找到多少、排得多靠前|`metrics`、`evaluate`|
|生成与引用|把选中的资料交给小模型，检查答案及依据|`Engine.generate`|

你不需要先逐行理解代码。先看一次问题在四个方案中的结果，再回到对应函数。

## 已知边界

本地0.5B生成模型能力有限，可能漏掉引用、错误拒答或误读条件。引用 ID 存在只说明引用格式合法，不说明证据支持结论。检索评测不等于回答正确率。开发集仅2条无答案题，无法证明拒答可靠。未经人工审核的合成问题不能支持业务提升结论。

文档中的权限过滤、缓存失效、版本管理是知识库的题材，并不是这个脚本已实现的服务功能。脚本实际实现的是文档切分、向量缓存、四组检索、重排、门控、生成及评测。

没有生产部署、真实用户量或可宣称的生产性能提升。

## 验证

```bash
RAG_PYTHON=.venv/bin/python .venv/bin/python -m unittest discover -s tests -v
```

测试覆盖多证据指标计算、RRF 合并、错误码关键词检索、证据 ID 完整性及阈值校准。实际模型端到端结果见 reports。

## 实现依据

- [Sentence Transformers：Retrieve & Re-Rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
- [BAAI BGE 中文模型说明](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [多语言 Cross-Encoder 模型说明](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
