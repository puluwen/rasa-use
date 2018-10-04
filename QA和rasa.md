典型对话系统
* IR-BOT：检索型问答系统，一问一答，不涉及上下文
* Task-bot：任务型对话系统
* Chichat-bot：闲聊系统
# 1任务型对话系统
![](index_files/273095256.png)
## 1.1 Natural Language Understanding,NLU,自然语言理解
## 1.2 Dialogue State Tracking（DST）状态追踪
已经填充的槽位，历史对话
## 1.3 Dialogue Policy对话决策
系统如何做出反馈动作
![](index_files/268821375.png)
## 1.4 Natural Language Generation，NLG，自然语言生成
![](index_files/268960699.png)
# 2 其他框架
![](index_files/269088152.png)
# 3 Rasa
## Rasa nlu：自然语言理解框架
* 实体识别
* 意图识别：意图识别：用词向量的平均值作为句向量，然后用sklearn的svm分类
https://github.com/RasaHQ/rasa_nlu
https://nlu.rasa.ai/
## Rasa core：对话管理框架
* 状态跟踪
* licy训练
* Online learning
https://core.rasa.ai/
#### NLU 的难点主要在语料的准备， 接下来就自己了解到的经验进行一一记录
*   每个意图要有关键字，意图中的每句都要有关键字。
*   每个关键字要扩充20左右的语句。
*   所有语句之间要够发散、离散（即除关键字外尽量不用重复的词语）。
*   除关键字之外，所有的词字，在每个意图中重复率要低、要低，最好不重复。
*   整个文件中，除关键字之外，所有的词字，重复率要低、要低，最好不重复。
*   上面两条造成的现象就是,你我他啊是的吗之类的词都要去掉（语义可以稍微不通顺，可接受）。
*   句式相同，参数不同的意图进行合并，通过后期校验参数进行分辨。
#### 意图识别的准确度跟两方面有关
*   关键字在当前意图中出现的频率
*   关键字在整个文件中出现的频率
#### 使用rasa core 的online leaning 进行对话决策的训练
在进行对话模块训练之前，需要准备两个文件：
*   domain.yml # 定义对话所有的意图、槽、实体和系统可能采取的action
*   story.md # 对话训练语料， 开始可以用于预训练对话模型，后期加入在线学习过程中生成的语料，更新完善


