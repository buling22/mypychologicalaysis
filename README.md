# 心理健康文本情感分析

**数据来源：** Kaggle Mental Health Dataset  
**技术栈：** Python · NLTK · scikit-learn · TF-IDF · Logistic Regression  

---

## 一、项目背景与目标

基于 Kaggle 心理健康数据集，对 1412 条用户发布的心理健康相关文本进行探索性分析与分类建模，识别各心理健康维度的语言特征规律。

**项目目标：**
- 通过词频统计与 TF-IDF 分析，识别各心理健康维度的关键词特征
- 构建逻辑回归分类基线模型，量化文本特征对维度的区分能力
- 通过混淆矩阵验证 EDA 阶段的分析猜想，形成完整的分析闭环

---

## 二、数据集说明

本数据集包含 1412 条心理健康相关文本，共 4 列：
- `text`：用户发布的原始文本内容
- `Explanations`：提取自 text 的核心表达片段，去除主语等结构性语言后保留的情感核心内容
- `labels`：心理健康维度标签（0-5）

| 标签 | 英文 | 含义 | 样本数 |
|------|------|------|--------|
| 0 | Intellectual | 认知层面 | 153 |
| 1 | Vocational | 职业层面 | 149 |
| 2 | Spiritual | 精神层面 | 189 |
| 3 | Physical | 身体层面 | 295 |
| 4 | Social | 社交层面 | 405 |
| 5 | Emotional | 情感层面 | 221 |

标签 4（Social）样本最多（405条），标签 1（Vocational）最少（149条），最多与最少相差约 2.7 倍，存在一定程度的类别不平衡。

---

## 三、数据处理流程

### 3.1 文本清洗
1. 统一转小写
2. 去除标点符号与多余空格
3. 使用 NLTK `word_tokenize` 分词
4. 使用 NLTK 标准停用词库结合自定义领域停用词过滤无意义词

> **设计决策：** 考虑到本项目为情感分析场景，保留词的原始形态，未做词形还原，以保留 `crying`、`suicidal` 等词的情绪强度信息。

### 3.2 特征工程
- **词频统计：** 使用 Counter 统计各标签 Top 15 高频词
- **文本长度：** 计算每条文本清洗后的词数
- **词汇丰富度：** 不重复词数 / 总词数，衡量词汇多样性
- **TF-IDF：** 对每条原始文本单独向量化（max_features=1000），再按标签取均值，避免拼接大文档导致的统计失真
- **词云可视化：** `text` 与 `Explanations` 字段各生成一套词云，从两个角度对比各标签核心词汇

---

## 四、探索性数据分析（EDA）

### 4.1 文本长度分析
![文本长度分布](charts/02_label_length.png)
- 标签 0（Intellectual）平均文本最长（32.84词），箱体最高且分布最广，用户表达认知困扰时倾向于详细叙述
- 标签 2（Spiritual）平均文本最短（20.86词），箱体最低、分布最集中，表达简短，可能与该类别情绪极端、难以展开有关
- 标签 4（Social）离群点最多，部分用户会写非常长的文字倾诉人际困扰
- 各标签均存在大量离群点，说明不同用户的表达习惯差异显著

### 4.2 词汇丰富度分析

- 标签 3（Physical）词汇丰富度最高（0.38），涉及大量医学词汇（anxiety、ptsd、antidepressants），词汇多样性最强
- 标签 2（Spiritual）词汇丰富度最低（0.29），高频词高度集中在 life、anymore、suicide 等少数词汇，说明这类困扰难以用语言充分展开
- 标签 0（Intellectual）文本最长（32.84词）但丰富度偏低（0.30），用户倾向于重复表达同一主题

### 4.3 各类别关键词特征
![Text词云](charts/03_text_wordcloud_all.png)
**Text 词云观察：**
- 标签 0（Intellectual）：future、life、seem 为主，体现对前途与自我方向的迷茫
- 标签 1（Vocational）：work、job、career、financial 高度集中，职业与经济压力主题最清晰
- 标签 2（Spiritual）：suicide、anymore、life、lost，是六类中唯一以极端词汇为核心的标签
- 标签 3（Physical）：anxiety、depression、sleep、ptsd、antidepressants，医学词汇密集，说明用户有明确的诊断背景
- 标签 4（Social）：family、friends、relationship、lonely、alone，人际关系词汇主导
- 标签 5（Emotional）：anxiety、depression、crying、sad，情绪描述词最丰富
![Explanations词云](charts/04_exp_wordcloud_all.png)
**Explanations 词云观察：**
- 与 Text 词云整体趋势一致，去除结构性语言后核心词更为突出
- 标签 0 中 scared、lack、motivation 更显著，说明认知层面困扰的核心是自信心不足与方向迷失
- 标签 1 中 unemployed、financial、bills 更突出，经济压力指向更明确
- 标签 2 中 suicidal、hate、guilty 出现，比 Text 词云情绪更为负面
- 标签 5 中新增 flashbacks、emotionally、panic，暗示与创伤经历相关

### 4.4 TF-IDF 分析
![TF-IDF分析](charts/05_tfidf_all.png)
- 标签 1（Vocational）的 work（0.09）、job（0.08）得分远高于其他词，主题最集中、边界最清晰
- 标签 3（Physical）Top10 几乎全为医学词汇（anxiety、depression、diagnosed、antidepressants、disorder），与其他标签形成明显差异
- 标签 0（Intellectual）、2（Spiritual）、5（Emotional）的高频词高度重叠（feel、im、like、just），TF-IDF 层面区分度低，预示这三类在分类建模中最容易混淆

---

## 五、机器学习建模

### 5.1 方法说明
- **模型：** 逻辑回归（Logistic Regression）
- **特征：** 对 `text_cleaned` 进行 TF-IDF 向量化（max_features=3000）
- **数据划分：** 80% 训练集（1129条）/ 20% 测试集（283条），random_state=42
- **类别不平衡处理：** 使用 `class_weight='balanced'`，对样本量少的类别自动加权

### 5.2 结果

**整体准确率：55.48%**

| 标签 | 类别 | Precision | Recall | F1-score |
|------|------|-----------|--------|----------|
| 0 | Intellectual | 0.41 | 0.45 | 0.43 |
| 1 | Vocational | 0.69 | 0.58 | 0.63 |
| 2 | Spiritual | 0.33 | 0.41 | 0.37 |
| 3 | Physical | 0.69 | 0.74 | 0.72 |
| 4 | Social | 0.68 | 0.65 | 0.67 |
| 5 | Emotional | 0.35 | 0.31 | 0.33 |

- 表现最好：标签 3（Physical，f1=0.72）和标签 1（Vocational，f1=0.63）
- 表现最差：标签 5（Emotional，f1=0.33）和标签 2（Spiritual，f1=0.37）

### 5.3 混淆矩阵分析
![混淆矩阵](charts/06_confusion_matrix.png)
**分类效果好的类别：**
- Physical（标签3）答对 43 条，正确率最高，与其 TF-IDF 特征词区分度强一致
- Social（标签4）答对 54 条，family、friends、relationship 等词汇主题明确

**误分类集中的类别：**
- Emotional（标签5）答对仅 16 条，有 15 条被误判为 Social，说明情感层面与社交层面文本表达高度相似
- Spiritual（标签2）有 9 条被误判为 Emotional，两类词汇层面难以区分
- Intellectual（标签0）有 8 条被误判为 Emotional，主题词模糊导致分类困难

**与 EDA 的对应：**
TF-IDF 分析中标签 0、2、5 高频词高度重叠，混淆矩阵印证了这一发现——这三类之间的误分类是全图中最密集的区域，说明纯词频特征不足以区分语义相近的心理健康维度，后续可考虑引入语义嵌入模型（如 BERT）进一步提升分类精度。

---

## 六、综合结论

1. **职业层面（标签1）** 文本主题最集中，TF-IDF 区分度最强，模型分类效果较好（f1=0.63）
2. **身体层面（标签3）** 医学词汇密集，词汇丰富度最高，模型分类效果最佳（f1=0.72）
3. **精神层面（标签2）** 文本最短、词汇最单一，却包含最极端的词汇（suicide、death），是六类中最需要关注的高风险类别
4. **认知（标签0）、精神（标签2）、情感（标签5）** 词汇高度重叠，混淆矩阵中这三类误分类最集中，验证了 EDA 阶段的判断
5. 纯词频/TF-IDF 特征对语义相近类别的区分能力有限，后续引入 BERT 等语义模型可进一步提升分类精度

> 六类文本共同指向一个核心词：**life**——它出现在每一个标签的高频词中。心理健康的困扰从来不是某个单一维度的问题，而是渗透在日常生活的每一个角落。

---

## 七、技术栈

| 类别 | 工具 |
|------|------|
| 语言 | Python 3 |
| 数据处理 | pandas、numpy |
| 自然语言处理 | NLTK（分词、停用词） |
| 特征工程 | scikit-learn TfidfVectorizer |
| 机器学习 | scikit-learn LogisticRegression |
| 可视化 | matplotlib、WordCloud |
| 数据来源 | Kaggle Mental Health Dataset |