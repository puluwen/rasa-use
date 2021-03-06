# 目录
[TOC]
# 1 Rasa安装
`推荐在Linux中安装`。因为要使用mitie的模型，而在windows里这个模型安装的时候需要进行编译，所以安装会非常麻烦。我自己是在虚拟机中安装了`deepin`的虚拟机，可以安装pycharm这个IDE，方便debug。
- 最好使用功能Anaconda新建一个虚拟环境，以避免包依赖混乱
- 安装代码如下，安装rasa core会自动安装对应版本的rasa nlu
```
pip install rasa_core==0.9.8
pip install -U scikit-learn sklearn-crfsuite
pip install git+https://github.com/mit-nlp/MITIE.git
pip install jieba
```
Rasa nlu github网址：https://github.com/RasaHQ/rasa_nlu
Rasa core github网址：https://github.com/RasaHQ/rasa_core

# 2 项目目录
```
yue-chatbot
  |--consolution
      |--answer  问答库相关文件
      |    |qa.json 从官网上采集的问题样本
      |    |qa_by_entity.json  单轮Fallback时，entity相关问题和答案
      |    |qa_by_intent.json  单轮Fllback时，intention相关问题和答案
      |--core_data  
      |    |domain.yml  定义意图，实体，槽，action，模板
      |    |story.md  故事
      |--models 训练保存的模型
      |    |nlu  训练好的nlu模型
      |    |dialogue  core模型
      |--nlu_data 
           |chatito  定义句子模板，用于生成训练nlu的标注样本
           |train_data  
           |  |rasa_dataset_training.json  chatito生成的json格式的样本，定义了同义词
           |  |regex.json  定义的正则，用于正则特征提取
         static  网页版的咨询机器人
         bot.py  系统python训练与运行接口
         myregex_entity_extrator.py  自定义的实体提取类
         pipeline_config.yml  nlu的流水线定义文件
         webchat..py  网页版机器人启动哦python脚本
         yue_action.py  定义所有的action
   
```
# 3 nlu
nlu的训练需要两个文件，一个文件是配置pipeline的文件，定义nlu的流水线;一个是标注了`entity`和`意图`的语料库，格式可以是Markdown（后缀为.md）和json两种格式，但`不能混用`，。

pipeline文件如下：
```
language: "zh"

pipeline:
- name: "nlp_mitie"
  model: "nlu_data/total_word_feature_extractor.dat"
- name: "tokenizer_jieba"
  dictionary_path: "nlu_data/jieba_dictionary.txt"  jieba自定义词典
- name: "ner_mitie"
- name: "myregex_entity_extractor.MyRegeexEntityExtractor" 自定义的正则实体提取器 
- name: "ner_synonyms"  #同义词替换
- name: "intent_entity_featurizer_regex"  额外正则特征，对应于文件nlu_data/train_data/regex.json
- name: "intent_featurizer_mitie"
- name: "intent_classifier_sklearn"
```

## 3.1 nlu的训练
- 训练可以直接调用系统的train放，用命令行的方式
```
python -m rasa_nlu.train -c pipelined的yaml文件 -d 训练样本(json格式化或Markdown格式) --fixed_model_name 模型目录名字 -o 输出文件夹
```
- 或使用代码的方式
```python
def train_nlu():
    from rasa_nlu.training_data.loading import load_data # 新api,会将目录下的所有文件合并
    from rasa_nlu.config import RasaNLUModelConfig#新 API
    from rasa_nlu.model import Trainer
    from rasa_nlu.config import load

    training_data = load_data("nlu_data/train_data")#加载训练数据
    trainer = Trainer(load("pipeline_config.yaml"))#加载pipeline
    trainer.train(training_data)
    model_directory = trainer.persist("models/", project_name="nlu",fixed_nmodel_name="model_ner_reg_all")

```

## 3.2 jieba分词自定义词典问题
因为分词结果中各个词的位置和实体的位置不匹配，会使得系统丢掉样本。
> 例：“粤通卡如何绑票根网”中，分词结果是“绑票”为一个词，但是实体“票根网”才是我们想要的，“票根网”开始位置不是分词结果某个词的开始位置，所以这条样本会被丢弃，目前被丢弃的大概有1500条

## 3.3 使用chatito生成nlu训练样本
chatito工具用于生成nlu的训练数据，为json格式。在里面分别定义好意图，实体，同一词。具体语法参照github项目网址：https://github.com/rodrigopivi/Chatito。
![Alt text](https://github.com/rodrigopivi/Chatito/raw/master/screenshot.jpg?raw=true)
安装方法：
```
npm i chatito -g
```
使用如下命令生成json格式的样本，在pycharmn可以使用`Ctrl+Alt+L`快捷键对json文件进行格式化。
```
npx chatito chatito文件或者目录 --format rasa
```

## 3.4 额外正则特征用于意图分类
nlu的训练数据中，可以使用一些正则特征来增强特征的表示，以用于意图分类。
```json
{
  "rasa_nlu_data":{
    "regex_features":[
      {
        "name":"reg1",
        "pattern":"(信用卡|储值卡|记账卡|银联卡|交通卡)"
      },
      {
        "name":"reg2",
        "pattern":"(公众号|城市服务|微信公众号|粤通卡公众号|微信城市服务)"
      },
      {
        "name":"rege3",
        "pattern":"(APP|app|App|应用|手机应用|IOS应用|安卓应用|application|Application)"
      },
      {
        "name":"reg4",
        "pattern":"(票根网|票根|收据|票卡|纸票)"
      }
    ],
    "entity_synonyms":[

    ],
    "common_examples":[

    ]
  }
}
```
每增加一个正则，则额外特征维度增加1，若该条正则匹配上，其值为1.0，反之为0.0。
源码如下：
```python
#位置rasa_nlu/featurizers/regex_featurizer.py
def features_for_patterns(self, message):
        """Checks which known patterns match the message.

        Given a sentence, returns a vector of {1,0} values indicating which
        regexes did match. Furthermore, if the
        message is tokenized, the function will mark the matching regex on
        the tokens that are part of the match."""

        found = []
        for i, exp in enumerate(self.known_patterns):#遍历全部正则
            match = re.search(exp["pattern"], message.text)
            if match is not None:#当前正则匹配上
                for t in message.get("tokens", []):#若分词token和正则结果有交叉，则设置一个标记
                    if t.offset < match.end() and t.end > match.start():
                        t.set("pattern", i)
                found.append(1.0)#匹配上，增加一个维度，值为1.0
            else:
                found.append(0.0)
        return np.array(found)
```
>正则里面用的词也可以用于chatito中，生成更多的句子，提高泛化能力

## 3.5 自定义NLU流水线组件
nlu流水线中，各个组件就像是流水线上的工人一样，从流水线上取一些东西，计算完毕后把结果放回流水线上，而这个流水线就是message变量。每个组件从message中取数据，计算完毕后又把计算结果更新到message中。
nlu的流水线组件都继承于Component类，其有些重要的属性和方法。
```python
class Component(object):
    name = ""#组件名字，
    provides = []#当前组件能够计算出什么
    requires = []#当前组件需要提供什么
    defaults = {}#组件的默认参数，可以被pipeline文件中覆写
    language_list = None#定义组件能处理什么语言，值为None则默认能处理所有语言
    
    def train(self, training_data, cfg, **kwargs):
        """训练组件，如果不需要训练，可以不实现，如正则"""
        pass
    def process(self, message, **kwargs):
        """使用组件进行处理，从message中取想要的数据，计算完成后更新到message中"""
        pass
    def persist(self, model_dir):
        """保存组件模型到本地，如果没有需要保存的东西，可以不实现"""
        pass
    def load(cls,
             model_dir=None,   # type: Optional[Text]
             model_metadata=None,   # type: Optional[Metadata]
             cached_component=None,   # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        """从本地加载保存的东西，若没有保存东西到本地，也不需要实现"""
        pass
```
将自定义组件加入到pipeline中：
```yaml
language: "zh"

pipeline:
- name: "nlp_mitie"
  model: "nlu_data/total_word_feature_extractor.dat"
- name: "tokenizer_jieba"
  dictionary_path: "nlu_data/jieba_dictionary.txt"
- name: "ner_mitie"
- name: "myregex_entity_extractor.MyRegeexEntityExtractor"  //自定义用正则提取实体的组件
- name: "ner_synonyms"
- name: "intent_entity_featurizer_regex"
- name: "intent_featurizer_mitie"
- name: "intent_classifier_sklearn"
```

# 4 Rasa core
rasa core的训练也需要两个文件：一个是定义domain的文件，定义了意图、实体、槽、actin和模板回复；一个story.md文件，用意图和action构建了会话的训练数据。相关文件在`core_data`目录下。
- domain.yaml文件

```yaml
slots:
  槽名1：
    - type: text
  槽名2：
    - type: text
intents:
  - 意图名1
  - 意图名2
entities：
  - 实体名1
  - 实体名2
templates：
  utter_greet:
    - "Hello"
    - "Hi"
  utter_goodbye:
    - "再见，为您服务很开心^_^"
    - "Bye，下次再见"
actions:
  - action名1
  - action名2
```
- story.md文件

```markdown
## story greet 故事name，训练用不到，官方文档提示在debug的时候会显示story的名字
* greet
  - utter_greet

## story goodbye
* goodbye
  - utter_goodbye

## story greet goodbye
* greet
  - utter_greet
* goodbye
  - utter_goodbye

## story inform num
* inform_num{"num":"1"}  包含的实体
  - Numaction
```

## 4.1 Rasa core的训练
类似nlu，也有两种方式：
- 直接调用系统train方法

```
python -m rasa_core.train -s stories.md -d domain.yml -o 模型保存路径 --epochs 500
```
- python接口的方式
```python
def train_dialogue(domain_file="core_data/domain.yml",
                   model_path="models/dialogue_all",
                   training_data_file="core_data/story.md",
                   max_history=2):
    from rasa_core.policies.fallback import FallbackPolicy
    #初始化agent
    agent = Agent(domain_file, policies=[
        KerasPolicy(MaxHistoryTrackerFeaturizer(BinarySingleStateFeaturizer(),max_history=max_history)),
        FallbackPolicy(fallback_action_name='action_default_fallback',
                       core_threshold=0.1,
                       nlu_threshold=0.3)])
    #用agent训练
    agent.train(
        training_data_file,
        epochs=200,
        batch_size=16,
        augmentation_factor=50,
        validation_split=0.2
    )

    agent.persist(model_path)
```

## 4.2 自定义action
action用于处理具体的业务逻辑。
自定义步骤：
- 首先在domain.yml文件中的actins中新增自定义的action，需要在类名前加上模块名（py文件名）
```yaml
actions:
  - yue_action.Banlihow
```
- 实现该自定义action。继承Action类，实现name和run方法。系统会对name方法返回的str到本Action对象做一个映射，系统就是靠这个映射，使用名字来找具体的Action类的。
映射可以查看rasa core模型保存目录下的domain.yml文件里的`action_names`和`actions`两栏。
## 4.3 FallbackPolicy
FallbackPolicy会在意图识别的置信度和action预测的置信度分别低于设定的阈值时触发，进行一些自定义的操作。但是完成的功能在最新版rasa core里才有，我们使用的rasa core版本是0.9.8，fallback触发后默认调用ActtionListen，而这个action什么也没有做。
FallbackPolicy源码
```python
def __init__(self,
                 nlu_threshold=0.3,  # type: float
                 core_threshold=0.3,  # type: float
                 fallback_action_name="action_listen"  # type: Text
                 ):
        # type: (...) -> None

        super(FallbackPolicy, self).__init__()

        self.nlu_threshold = nlu_threshold
        self.core_threshold = core_threshold
        self.fallback_action_name = fallback_action_name

```
ActionListen源码:
```python
class ActionListen(Action):
    """The first action in any turn - bot waits for a user message.
    The bot should stop taking further actions and wait for the user to say
    something."""

    def name(self):
        return ACTION_LISTEN_NAME

    def run(self, dispatcher, tracker, domain):
        return []
```

新版FallbackPolicy默认调用action_default_fallback，然后调用的是utter_default这个模板。
ActionDefaultFallback源码：
```python
class ActionDefaultFallback(Action):
    """Executes the fallback action and goes back to the previous state
     of the dialogue"""
    def name(self):
        return ACTION_DEFAULT_FALLBACK_NAME
    def run(self, dispatcher, tracker, domain):
        from rasa_core.events import UserUtteranceReverted
        dispatcher.utter_template("utter_default", tracker, silent_fail=True)
        return [UserUtteranceReverted()]
```
但这还是不是我们想要的，我们是想在触发Fallback的时候根据识别的实体或者置信度最高的意图（即便置信度小于阈值，但还是能得到最新度最高的意图）返回一些相关的问题，所以进行了改造。
步骤：
- 首先在domain.yml文件中增加actin：yue_action.ActionDefaultFallback
- 在yue_action.py文件中实现ActionDefaultFallback，其中name方法返回的名字一定要和传给FallbackPolicy的fallback_action_name参数保持一致。

## 4.4 消息返回
整体来说，有两种方式返回消息
- 如果rasa core判定直接使用某个模板来返回，则会从该模板中选择一条语句进行返回
- 如果rasa core判定是调用某个action来返回，则又有两种方式:在action的run()方法中分别调用dispatcher的utter_message()和utter_template()方法
```python
dispatcher.utter_message(text)#直接返回text
dispatcher.utter_template(template_name)#调用模板，返回该模板中的某一句
```
