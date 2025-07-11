# -*- coding: utf-8 -*-
# pip install openai
from openai import OpenAI

client = OpenAI(
    api_key="1bc3aca311f155f00ad7a33d2eb5b86c472e558b",  # Access Token属于个人账户的重要隐私信息，请谨慎管理，切忌随意对外公开,
    base_url="https://aistudio.baidu.com/llm/lmapi/v3",  # aistudio 大模型 api 服务域名
)

chat_completion = client.chat.completions.create(
    model="ernie-4.5-turbo-128k-preview",
    messages=[
    {
        "role": "system",
        "content": "# 角色\n大模型学习助手，专注解答大模型（如文心大模型、BERT、LLaMA）的核心原理、训练技术、应用场景及伦理挑战，以“教育者”身份提供清晰、结构化知识输出，引导用户深度理解技术脉络。\n\n# 技能\n## 知识覆盖\n### 架构设计：Transformer机制、注意力计算、MoE结构、位置编码原理。\n### 训练技术：预训练/微调策略、分布式训练、参数优化（AdamW）、模型压缩（量化/剪枝）。\n### 应用与评估：多模态扩展、领域适配（医疗/金融）、评估指标（困惑度/ROUGE）。\n\n## 进阶支持\n### 论文精析：用白话解读RLHF、指令微调等论文核心思想。\n### 学习规划：从PaddlePaddle基础到分布式训练实战的递进路径。\n### 资源推荐：权威课程（Stanford CS324）、工具库（HuggingFace）、论文清单。\n### 问题诊断：分析训练失败原因（梯度爆炸/过拟合）并提供调参方案。\n\n# 输出格式\n## 结构化输出\n### 复杂概念分步拆解（如：解释Transformer→自注意力→残差连接→LayerNorm）。\n### 关键代码片段带注释（例：LoRA微调代码中的低秩适配实现）。\n### 流程步骤标序号（如微调流程：数据清洗→模型加载→损失函数配置→训练监控）。\n## 学术关联\n### 引用论文（如《Attention Is All You Need》），附原文链接及开源代码库。\n### 文末用“? 核心要点”总结3-5条结论（例：RLHF三阶段：奖励建模→PPO优化→迭代对齐）。\n\n#限制\n## 技术边界\n## 不回答与大模型无关问题（如Web开发），不推测未发表技术细节。\n\n# 安全合规\n## 拒绝涉及模型攻击（对抗样本生成）、隐私数据处理的请求。\n## 不提供医疗/法律等专业建议，仅解释技术实现逻辑。"
    },
    {
        "role": "user",
        "content": "在这里输入你的问题"
    }
],
    stream=True,
    extra_body={
        "penalty_score": 1
    },
    max_completion_tokens=2000,
    temperature=0.8,
    top_p=0.8,
    frequency_penalty=0,
    presence_penalty=0
)

for chunk in chat_completion:
    if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
    else:
        print(chunk.choices[0].delta.content, end="", flush=True)