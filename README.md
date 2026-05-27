# 心理咨询对话三级标签自动标注系统

基于传统 NLP + 机器学习方法，对心理咨询对话进行 **S1 / S2 / S3** 三级分类标注。

- **S1 — 日常困扰（轻度心理不适）**：学业、职场、家庭矛盾、失眠、压力、社交等 17 个子类
- **S2 — 中度心理障碍**：抑郁、焦虑、双相、PTSD、饮食障碍、强迫等 9 个子类
- **S3 — 紧急危机**：正在自杀、自杀计划、自残、伤害他人、报复等 5 个子类

数据来源：`psy525` 心理咨询平台真实问答对话。

---

## 技术栈

- **分词**：jieba
- **特征提取**：TF-IDF（1-3gram，sublinear tf）
- **分类器**：LogisticRegression（多分类，max_iter=1000）
- **启发式基线**：关键词匹配（S3→S2→S1 优先级覆盖）

---

## 项目结构

```
data-annotation/
├── data/                           # 数据目录
│   ├── student-01.json             # 原始未标注数据
│   ├── student-01_labeled_a2.json  # 自动标注结果
│   ├── student-01_labeled_refined.json  # 加权重训精炼结果
│   ├── student-01_corrections.json      # 人工/LLM 修正标签
│   ├── student-01_correction_tasks.json # 待修正样本任务
│   ├── student-01_review.csv            # 审核 CSV
│   └── ...                          # student-02, student-03 同理
├── pipeline_a.py                   # 思路A: 批量自动标注主流程
├── generate_review.py              # 生成不确定样本审核 CSV
├── prepare_correction_tasks.py     # 从审核 CSV 生成修正任务
├── auto_correct.py                 # 用 LLM (DeepSeek) 自动修正标签
├── refine_loop.py                  # 两阶段自动修正循环
├── merge_corrections.py            # 合并多个子代理修正结果
├── retrain.py                      # 人工修正后加权重训
├── venv/                           # Python 虚拟环境
├── .gitignore
└── README.md
```

---

## 脚本说明

### `pipeline_a.py` — 批量自动标注

处理 `data/` 下所有 `student-*.json`（排除已标注的）。

流程：加载 → 预处理（去重） → 启发式关键词标注 → TF-IDF 向量化 → LogisticRegression 训练预测 → 输出 `*_labeled_a2.json`

```bash
python3 pipeline_a.py
```

### `generate_review.py` — 生成审核 CSV

训练模型后，按置信度排序选出最不确定的样本，供人工/LLM 审核修正。

```bash
python3 generate_review.py --data data/student-01.json --n 500
```

输出：`data/student-01_review.csv`

### `prepare_correction_tasks.py` — 生成修正任务

从 review CSV 生成 JSON 格式修正任务，包含标题、内容、对话上下文。

```bash
python3 prepare_correction_tasks.py --data data/student-01.json --review data/student-01_review.csv
```

输出：`data/student-01_correction_tasks.json`

### `auto_correct.py` — LLM 自动修正

用 DeepSeek API 自动修正不确定样本的标签。分批调用，按置信度升序（最不确定的先修）。

```bash
export DEEPSEEK_API_KEY=sk-xxx
python3 auto_correct.py \
  --input data/student-01_review.csv \
  --data data/student-01.json \
  --output data/student-01_corrections.json \
  --batch-size 25
```

### `refine_loop.py` — 两阶段修正循环

核心循环：**prepare → LLM/人工修正 → apply（加权重训）**，可迭代至目标准确率。

**阶段1（prepare）**：训练模型 → 选取不确定样本 → 生成 review CSV + correction_tasks.json

```bash
python3 refine_loop.py --phase prepare --file student-01
```

**阶段2（apply）**：加载修正 → 加权（修正样本权重×10）→ 重训 → 输出 refined JSON

```bash
python3 refine_loop.py --phase apply --file student-01 \
  --corrections data/student-01_corrections.json
```

### `merge_corrections.py` — 合并修正

当多个子代理/人做了各有修正时，合并去重。

```bash
python3 merge_corrections.py --output data/student-01_corrections.json \
  data/corrections_part1.json data/corrections_part2.json
```

### `retrain.py` — 加权重训（简化版）

加载原始数据 + 修正文件 → 加权重训 → 输出 refined JSON。

```bash
python3 retrain.py \
  --data data/student-01.json \
  --corrections data/student-01_corrections.json \
  --output data/student-01_labeled_refined.json
```

---

## 标签体系

| 层级 | 子类数 | 说明 |
|------|--------|------|
| **S3 紧急危机** | 5（3.1~3.5） | 自杀、自残、伤害他人等需立即干预 |
| **S2 中度障碍** | 9（2.1~2.9） | 抑郁、焦虑、PTSD、强迫等需要专业干预 |
| **S1 日常困扰** | 17（1.1~1.17） | 学业、职场、家庭、失眠、压力等生活话题 |

优先级规则：**S3 > S2 > S1**，选择最匹配的子类。

---

## 标注优化流程

```
原始数据 ──→ pipeline_a.py（初始标注）
                │
                ▼
         generate_review.py（选不确定样本）
                │
                ▼
         prepare_correction_tasks.py（生成任务）
                │
                ▼
         人工 / LLM 修正标签
                │
                ▼
         refine_loop.py apply（加权重训）
                │
                ▼
         检查准确率 ──→ 达标 → 完成
                │
                ▼（不达标）
         进入下一轮修正循环
```

---

## 依赖

```bash
pip install jieba numpy scikit-learn
# auto_correct.py 额外需要: openai
pip install openai
```

---

## 备注

- `.gitignore` 已忽略 `venv/`、`__pycache__/`、`*.pyc`、锁文件及 `data/classify_p2.py`
- 数据从 `psy525` 心理咨询平台爬取，含完整问答对话
- 远程仓库：`git@github.com:Jared-Linn/data-annotation.git`（SSH）
- 项目为课程项目（滇池学院理工学院 · NLP 方向）
