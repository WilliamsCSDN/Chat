# RAG 分类检索技能

<!-- config
name: 支付
description: 支付相关的内部知识，涵盖支付流程、退款处理、到账时间、支付渠道、交易流水、支付异常等问题
trigger: 调用 retrieve_knowledge(query=用户问题, category="支付")
milvus_expr: category_l1 == "支付"
keywords: 支付, 退款, 到账, 扣款, 支付渠道, 收银台, 交易流水, 支付失败
-->

<!-- config
name: 分润
description: 分润相关的内部知识，涵盖分润规则、佣金比例、返佣机制、结算周期、分账逻辑、手续费等问题
trigger: 调用 retrieve_knowledge(query=用户问题, category="分润")
milvus_expr: category_l1 == "分润"
keywords: 分润, 佣金, 返佣, 结算, 分账, 手续费, 利润分配
-->

<!-- config
name: 进件
description: 进件相关的内部知识，涵盖进件流程、入驻条件、资质要求、审核标准、资料提交、开户等问题
trigger: 调用 retrieve_knowledge(query=用户问题, category="进件")
milvus_expr: category_l1 == "进件"
keywords: 进件, 入驻, 资质审核, 商户注册, 资料提交, 开户
-->
