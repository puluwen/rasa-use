# 2.slot槽
分类:
- text：仅告诉core这个slot是否有值
- categorical：
- boolean：
- float：
- list：
- unfeaturized：仅仅想存数据，不想影响决策，可以用这个slot
## 2.1 How rasa uses slot
`rasa_core.policies.Policy`只是获得槽值，它收到的是特征化后的表示。
## 2.2 How slots get set
可以设置一个初始值
```
slots:
  name:
    type: text
    initial_value: "human"
```
在会话过程中设置slot的方法有几种
### 2.2.1 从NLU中获得
如果nlu识别出了一个实体，并且当前领域包含了一个同名的slot，则slot会自动被赋值。例如:
```
# story_01
* greet{"name": "Ali"}
  - slot{"name": "Ali"}
  - utter_greet</pre>
```
上面这个例子中，- slot就不用写了{}，因为会被自动赋值
### 2.2.2 Slots set by clicking buttons

```
utter_ask_color:
- text: "what  color  would  you  like?"
  buttons:
  - title: "blue"
    payload: '/choose{"color":  "blue"}'
  - title: "red"
    payload: '/choose{"color":  "red"}'
```
# 3 Fallback Actions
当nlu的意图识的置信度小于`nlu_threshold`或者所有对话策略预测出的action的置信度都小于`core_threadhold`的时候，会被调用，返回如“对不起，我没有理解！”。这时要将`FallbackPolicy`加入到策略组合中
```python
from rasa_core.policies.fallback import FallbackPolicy
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.agent import Agent

fallback = FallbackPolicy(fallback_action_name="action_default_fallback",
                          core_threshold=0.3,
                          nlu_threshold=0.3)

agent = Agent("domain.yml", policies=[KerasPolicy(), fallback])
```
`actioni_fallback`是rasa_core默认的action，它将会发送`utter_default`模板给user，所以必须要保证这个模板在domain文件里有。并且它还会将会话状态回退到引起回退的用户句子之前的状态（即上次用户说的话），以避免对将来action预测的影响。
源码为：
```python
class  ActionDefaultFallback(Action):  
    """Executes the fallback action and goes back to the previous state  of the dialogue"""  
    def  name(self):  
        return  ACTION_DEFAULT_FALLBACK_NAME  
    def  run(self,  dispatcher,  tracker,  domain):  
        from  rasa_core.events  import  UserUtteranceReverted      
        dispatcher.utter_template("utter_default",  tracker,  silent_fail=True)  
        return  [UserUtteranceReverted()]
```
# 4 训练
训练命令
```
python -m rasa_core.train -d domain.yml -s data/stories.md -o models/current/dialogue --epochs 200</pre>
```
或者通过创建agent并运行run方法
```python
from rasa_core.agent import Agent

agent = Agent()
data = agent.load_data("stories.md")
agent.train(data)</pre>
```
## 4.1 数据扩充
默认情况下，Rasa Core会通过随机粘贴故事文件中的故事来创建更长的故事。这会使得policy在不相关的情况下，**忽略**对话历史记录，无论前面发生了什么，只需回应相同的操作。
可以使用`--augmentation`标志更改此行为，`--augmentation 0`禁用。
在python中，可以将`augmentation_factor`参数传递给`Agent.load_data`方法。
## 4.2 max_history
rasa core一个重要参数就是`max_history`,控制可以查看的对话历史记录，以决定接下来采取的操作。
>注意： `MaxHistoryTrackerFeaturizer`使用最大历史记录，而`FullDialogueTrackerFeaturizer`始终查看完整的对话历史记录。

增加`max_history`会使增大模型，训练实验也会增大。
## 4.3 训练脚本选项
```
usage: train.py [-h]
                (-s STORIES | --url URL | --core CORE)
                [-o OUT]
                [-d DOMAIN]
                [-u NLU]
                [--history HISTORY]
                [--epochs EPOCHS]
                [--validation_split VALIDATION_SPLIT]
                [--batch_size BATCH_SIZE]
                [--online]
                [--finetune]
                [--augmentation AUGMENTATION]
                [--debug_plots]
                [--dump_stories]
                [--endpoints ENDPOINTS]
                [--nlu_threshold NLU_THRESHOLD]
                [--core_threshold CORE_THRESHOLD]
                [--fallback_action_name FALLBACK_ACTION_NAME]
                [-v]
                [-vv]
                [--quiet]

trains a dialogue model

optional arguments:
  -h, --help            显示帮助信息并退出
  -s STORIES, --stories STORIES
                        story文件或者文件夹
  --url URL             story文件下载地址，下载后以此为训练story文件。通过GET请求获取数据
  --core CORE           预训练的模型地址（仅在线模型）
  -o OUT, --out OUT     directory to persist the trained model in
  -d DOMAIN, --domain DOMAIN
                        领域制定的yaml文件
  -u NLU, --nlu NLU     训练的NLU模型
  --history HISTORY     使用最大历史对话
  --epochs EPOCHS       训练的epoch数量
  --validation_split VALIDATION_SPLIT
                        分给验证集的数据集的比例,默认是0.1
  --batch_size BATCH_SIZE
                        batch-size大小
  --online              是否在线训练
  --finetune            retrain the model immediately based on feedback.
  --augmentation AUGMENTATION
                        how much data augmentation to use during training
  --debug_plots         If enabled, will create plots showing checkpoints and
                        their connections between story blocks in a file
                        called `story_blocks_connections.pdf`.
  --dump_stories        If enabled, save flattened stories to a file
  --endpoints ENDPOINTS
                        Configuration file for the connectors as a yml file
  --nlu_threshold NLU_THRESHOLD
                        如果nlu预测的执行度低于这个值，则调用fallback
  --core_threshold CORE_THRESHOLD
                        如果core中action预测的执行度低于这个值将调用fallback
  --fallback_action_name FALLBACK_ACTION_NAME
                        When a fallback is triggered (e.g. because the ML
                        prediction is of low confidence) this is the name of
                        tje action that will get triggered instead.
  -v, --verbose         Be verbose. Sets logging level to INFO
  -vv, --debug          Print lots of debugging statements. Sets logging level
                        to DEBUG
  --quiet               Be quiet! Sets logging level to WARNING</pre>
```
# 5 Policy策略
该`rasa_core.policies.Policy`课程决定在对话的每一步中采取的操作。
在一个agent中可以包含多个policy。在每一轮对话中都选择执行度最高的动作。
```
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.agent import Agent

agent = Agent("domain.yml",
               policies=[MemoizationPolicy(), KerasPolicy()])
```
>默认情况下，rasa core使用`KerasPolicy`和`MemorizationPolicy`

## 5.1 Memoization Policy
这个`Memoization Policy`只是会记住训练数据，如果对话在数据中存在，就以执行度**1.0**返回下一个action，否则以执行度**0**返回**None**。
## 5.2 KerasPolicy
使用神经网络Keras选择下一个动作，默认基于LSTM，以max_history对话为输入，下一个action为输出。可以覆写`KerasPolicy.model_architecture`方法以实现自己的体系。
```
def model_architecture(
            self,
            input_shape,  # type: Tuple[int, int]
            output_shape  # type: Tuple[int, Optional[int]]
    ):
        # type: (...) -> tf.keras.models.Sequential
        """Build a keras model and return a compiled model."""

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import \
            Masking, LSTM, Dense, TimeDistributed, Activation

        # Build Model
        model = Sequential()

        # the shape of the y vector of the labels,
        # determines which output from rnn will be used
        # to calculate the loss
        if len(output_shape) == 1:
            # y is (num examples, num features) so
            # only the last output from the rnn is used to
            # calculate the loss
            model.add(Masking(mask_value=-1, input_shape=input_shape))
            model.add(LSTM(self.rnn_size, dropout=0.2))
            model.add(Dense(input_dim=self.rnn_size, units=output_shape[-1]))
        elif len(output_shape) == 2:
            # y is (num examples, max_dialogue_len, num features) so
            # all the outputs from the rnn are used to
            # calculate the loss, therefore a sequence is returned and
            # time distributed layer is used

            # the first value in input_shape is max dialogue_len,
            # it is set to None, to allow dynamic_rnn creation
            # during prediction
            model.add(Masking(mask_value=-1,
                              input_shape=(None, input_shape[1])))
            model.add(LSTM(self.rnn_size, return_sequences=True, dropout=0.2))
            model.add(TimeDistributed(Dense(units=output_shape[-1])))
        else:
            raise ValueError("Cannot construct the model because"
                             "length of output_shape = {} "
                             "should be 1 or 2."
                             "".format(len(output_shape)))

        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        logger.debug(model.summary())

        return model</pre>
```
训练在这里run
```
def train(self,
              training_trackers,  # type: List[DialogueStateTracker]
              domain,  # type: Domain
              **kwargs  # type: Any
              ):
        # type: (...) -> Dict[Text: Any]

        if kwargs.get('rnn_size') is not None:
            logger.debug("Parameter `rnn_size` is updated with {}"
                         "".format(kwargs.get('rnn_size')))
            self.rnn_size = kwargs.get('rnn_size')

        training_data = self.featurize_for_training(training_trackers,
                                                    domain,
                                                    **kwargs)

        # noinspection PyPep8Naming
        shuffled_X, shuffled_y = training_data.shuffled_X_y()

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            with self.session.as_default():
                if self.model is None:
                    self.model = self.model_architecture(shuffled_X.shape[1:],
                                                         shuffled_y.shape[1:])

                validation_split = kwargs.get("validation_split", 0.0)
                logger.info("Fitting model with {} total samples and a validation "
                            "split of {}".format(training_data.num_examples(),
                                                 validation_split))
                # filter out kwargs that cannot be passed to fit
                params = self._get_valid_params(self.model.fit, **kwargs)

                self.model.fit(shuffled_X, shuffled_y, **params)
                # the default parameter for epochs in keras fit is 1
                self.current_epoch = kwargs.get("epochs", 1)
                logger.info("Done fitting keras policy model")</pre>
```
# 6 调试
可以在命令行加上`--debug`进入调试
```
python -m rasa_core.run -d models/dialogue -u models/nlu/current --debug</pre>
```
会打印出很多信息,如
```text
 Bot loaded. Type a message and press enter:
 /greet
 rasa_core.tracker_store - Creating a new tracker for id 'default'.
 rasa_core.processor - Received user message '/greet' with intent '{'confidence': 1.0, 'name': 'greet'}' and entities '[]'
 rasa_core.processor - Logged UserUtterance - tracker now has 2 events
 rasa_core.processor - Current slot values:

 rasa_core.policies.memoization - Current tracker state [None, {}, {'prev_action_listen': 1.0, 'intent_greet': 1.0}]
 rasa_core.policies.memoization - There is a memorised next action '2'
 rasa_core.policies.ensemble - Predicted next action using policy_0_MemoizationPolicy
 rasa_core.policies.ensemble - Predicted next action 'utter_greet' with prob 1.00.
 Hey! How are you?</pre>
```
# 7 story文件格式
```
## story_07715946    <!-- name of the story - just for debugging -->
* greet
   - action_ask_howcanhelp
* inform{"location": "rome", "price": "cheap"}  <!-- user utterance, in format intent{entities} -->
   - action_on_it
   - action_ask_cuisine
* inform{"cuisine": "spanish"}
   - action_ask_numpeople        <!-- action that the bot should execute -->
* inform{"people": "six"}
   - action_ack_dosearch</pre>
```
- 一个故事开头前面有##，可以随便命名，可以为调试提供信息
- 故事结尾用一个空行隔开，然后开始新的故事
- 用户发送的消息以\*开头
- 机器执行的action以- 开头，并包含操作名称
- action返回的时间要马上接在action之后。如例如：如果某个操作范湖我一个`SetSlot`事件，则会显示为该行`- slot{"slot_name":"value"}`
