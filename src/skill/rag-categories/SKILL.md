---
name: rag-categories
description: RAG knowledge retrieval with categorized Milvus filtering. Use when the user asks about 支付 (payment), 分润 (profit sharing), or 进件 (merchant onboarding) topics, or when the query relates to Chinese payment/clearing domain knowledge. Triggers retrieve_knowledge with the appropriate category tag and Milvus filter expression to scope retrieval to the correct knowledge partition.
---

# RAG 分类检索技能

## 使用方式

当用户问题涉及以下领域时，调用 `retrieve_knowledge` 函数，传入对应的 `category` 参数。该函数会使用对应的 Milvus 过滤表达式限定检索范围。

调用格式：`retrieve_knowledge(query=<用户问题>, category=<分类名>)`

## 分类定义

### 支付

- **描述**：支付相关的内部知识，涵盖支付流程、退款处理、到账时间、支付渠道、交易流水、支付异常等问题。
- **触发关键词**：支付、退款、到账、扣款、支付渠道、收银台、交易流水、支付失败
- **Milvus 过滤表达式**：`category_l1 == "支付"`

### 分润

- **描述**：分润相关的内部知识，涵盖分润规则、佣金比例、返佣机制、结算周期、分账逻辑、手续费等问题。
- **触发关键词**：分润、佣金、返佣、结算、分账、手续费、利润分配
- **Milvus 过滤表达式**：`category_l1 == "分润"`

### 进件

- **描述**：进件相关的内部知识，涵盖进件流程、入驻条件、资质要求、审核标准、资料提交、开户等问题。
- **触发关键词**：进件、入驻、资质审核、商户注册、资料提交、开户
- **Milvus 过滤表达式**：`category_l1 == "进件"`

## 分类判断规则

1. 分析用户问题中的核心主题，与各分类的关键词进行匹配。
2. 如果匹配到某一分类，使用该分类的 `category` 参数调用 `retrieve_knowledge`。
3. 如果问题跨越多个分类，分别对每个相关分类调用 `retrieve_knowledge`。
4. 如果无法匹配任何分类，使用 `category="通用"` 进行全库检索。
