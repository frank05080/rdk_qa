# RDK QA

## 节点介绍

本仓库让开发者在RDK X5开发板上体验完整的一套云边端结合的语音方案。本方案基于TROS，使用的算法如下：

1. 本地百度飞浆的语音唤醒KWS算法（MDTC）
2. 本地Huggingface中的wav2vec2语音识别算法
3. 云端基于火山引擎的doubao-128k-pro算法

## 节点使用

本项目包含两个节点，一个发布节点（publisher），以及一个订阅节点（subscriber）:

1. 发布节点（publisher）：发布节点会开启一个音频流，将唤醒词、需要语音识别句子的音频发送给订阅节点
2. 订阅节点（subscriber）：订阅节点负责接收唤醒词、需要语音识别句子的音频，并将这些音频发送给对应的模型进行推理

### 环境准备

#### 火山引擎大模型网关

用户需前往火山引擎大模型网关官网（https://www.volcengine.com/docs/6893/1263410），跟着步骤完成账号注册，且在`创建网关访问秘钥`这一步，勾选`Doubao-pro-128k`。

完成步骤1，即可获得网关访问秘钥，这个秘钥会在后续的api_key参数中使用。

#### 硬件准备

用户需准备一个3.5mm的耳机麦克风插入RDK X5耳机孔，用于音频的输入和输出

#### 软件准备

安装ros依赖：
```
sudo apt update
sudo apt install python3-colcon-ros
```

激活ros环境（注：每次新开一个terminal（终端），都建议重新激活一下环境，即执行以下代码）
对于RDK OS V3.0版本，执行：`source /opt/tros/humble/setup.bash`
对于RDK OS V2.1版本，执行：`source /opt/tros/setup.bash`

在RDK开发板上新建一个工作空间目录，并构建src子目录：

```
mkdir -p colcon_ws/src
cd colcon_ws/src
```

使用git clone命令，将本项目克隆至src目录下。

```
git clone https://github.com/frank05080/rdk_qa.git
```

进入项目，拉取对应模型，安装pip依赖。
```
cd rdk_qa
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/kws.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/wav2vec2_output_30000.bin
pip3 install -r requirements.txt
cd ..
```
注：该项目与RDK Model Zoo依赖保持一致，均需要bpu_infer_lib，如尚未安装bpu_infer_lib，请参考https://github.com/D-Robotics/rdk_model_zoo链接进行安装

回到工作空间目录（colcon_ws），构建项目

```
cd ..
colcon build
```


激活项目环境（注：在完成环境构建后，每次新开一个terminal（终端），都建议重新激活一下环境，即执行以下代码）
对于RDK OS V3.0版本，执行：`source ./install/setup.sh`
对于RDK OS V2.1版本，执行：`source ./install/setup.bash`


### 使用节点

#### 发布节点

我们首先在MobaXTerm打开一个终端，启动一个发布节点。

```
ros2 run rdk_qa pub
```

执行后，会弹出一系列和ALSA有关的警告，如Invalid card等，无视即可。

#### 订阅节点

随后，使用MobaXTerm打开另一个终端，启动一个订阅节点。

注意（**这一步必须替换，如果直接复制粘贴如下命令，程序会直接退出**）

1. 这里需要填入bin模型的地址做为参数，请根据实际情况填入kws.bin以及wav2vec2_outpu_30000.bin文件路径
2. 这里需要填入的llm_api_key为`环境准备`步骤中申请的api_key，将其复制到这里即可

```
ros2 run rdk_qa sub --ros-args -p llm_api_key:=None -p kws_bin_path:=None -p asr_bin_path:=None
```

启动订阅节点后，我们看到`[audio_sub]: FINISHED LOADING MODEL, START AWAKEN..`的回显，则表示所有本地模型已经完成加载，此刻可以进行唤醒。

**目前唤醒词固定为：hey, snips**

这里唤醒词的发音可以参考B站视频的发音方式。

我们将唤醒词的正确发音对准耳机的麦克风进行发音，即可让模型正确唤醒。当模型唤醒后，我们会在订阅节点的终端中看到`唤醒`二字。

此时当发布节点的终端会出现回显：`[audio_pub]: Please start speaking:`时，即可对耳机麦克风进行说话。

**注意：由于X5语音识别模型为静态模型，输入音频的长度有限，在本案例中，输入音频的长度限制为2s，即期待用户以较快的语速进行一些简短问题的提问。如希望有较长的音频输入，请联系地瓜工程师**

完成2s的语音输入后，看到回显：`[INFO] [1739192185.705863110] [audio_pub]: Speaking ends now..`，则表示语音成功输入。

稍等片刻，在订阅节点的终端中会出现回显：`[INFO] [1739192189.928730583] [audio_sub]: clean_text: 你在哪`。这里文字即使语音转文字（ASR）的本地模型推理结果。

随后，订阅节点会将语音转文字（ASR）的识别结果喂入云端豆包大模型，豆包大模型会根据您问的问题给出对应的答案。