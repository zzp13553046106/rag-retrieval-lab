"""RAG核心函数练习骨架：TODO是练习内容，不被rag.py调用。

先对照core_functions_reference.py理解输入输出，再尝试自行补全。
不包含模型下载逻辑；导入本文件不会加载权重。
"""
from __future__ import annotations
import collections
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from rag import ROOT, CACHE, CONFIG, tokens, Engine as _Engine

def load_chunks():
    """读取Markdown并切块；返回包含id/title/text/source的片段列表。"""
    raise NotImplementedError("请自行补全load_chunks")


def rank(scores, k):
    """将已有分数排序，返回前k个片段下标；不负责计算相关性。"""
    raise NotImplementedError("请自行补全rank")


def rrf(rankings, constant=60):
    """按两路名次累加贡献；返回融合排名，候选截断由search完成。"""
    raise NotImplementedError("请自行补全rrf")


def metrics(ids, relevant, k=3):
    """比较排名与标准证据，返回Recall、MRR、nDCG；不评价生成答案。"""
    raise NotImplementedError("请自行补全metrics")


class BM25:
    """完成词频/IDF准备与关键词打分。"""

    def __init__(self, texts):
        raise NotImplementedError("请自行补全BM25.__init__")

    def scores(self, query):
        raise NotImplementedError("请自行补全BM25.scores")


class EngineCore(_Engine):
    """先补全方法再创建实例；初始化仍沿用原项目。"""

    def encode(self, texts):
        """文字列表→归一化语义向量；调用现有BGE模型，不训练权重。"""
        raise NotImplementedError("请自行补全encode")

    def rerank(self, query, candidates):
        """原始问题＋候选下标→Cross-Encoder打分后的排名及分数字典。"""
        raise NotImplementedError("请自行补全rerank")

    def search(self, query):
        """组织BM25、语义检索、融合和重排；返回排名、分数和耗时。"""
        raise NotImplementedError("请自行补全search")

    def generate(self, question, ranked, scores, threshold):
        """选取证据、检查门控、生成答案并诊断引用；保留模型原始输出。"""
        raise NotImplementedError("请自行补全generate")

