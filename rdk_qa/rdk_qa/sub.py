import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from std_msgs.msg import Int32, ByteMultiArray
import numpy as np
import bpu_infer_lib
import paddle
from paddleaudio.compliance.kaldi import fbank
import pyaudio
import scipy.signal as sps
import onnxruntime as ort
import os
from openai import OpenAI
import sys


class LLMInfer:
    def __init__(self, api_key: str):
        self.ans_hist = []
        self.client = OpenAI(
            base_url="https://ai-gateway.vei.volces.com/v1",
            api_key=api_key,
        )
        self.cnt = 0

    def llm_infer(self, text):
        messages = [{"role": "system", "content": "你的名字叫地瓜机器人。"},]
        for entry in self.ans_hist:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["assistant"]})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text},
            ]
        })
        self.cnt += 1

        completion = self.client.chat.completions.create(
            model="doubao-pro-128k",
            messages=messages,
        )
        response_message = completion.choices[0].message.content
        self.ans_hist.append({
            "user": text,
            "assistant": response_message
        })
        print("Answer:", response_message)
    

class AudioSub(Node):
    def __init__(self):
        super().__init__('audio_sub')
        
        self.api_key = self.get_ros_param("llm_api_key", "None")
        self.kws_bin_path = self.get_ros_param("kws_bin_path", "None")
        self.asr_bin_path = self.get_ros_param("asr_bin_path", "None")
        self.vocab_path = self.get_ros_param("vocab_path", "None")
        
        self.check_params()
        
        self.llm_inf = LLMInfer(self.api_key)

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.subscription = self.create_subscription(
            ByteMultiArray,
            'kws_topic',
            self.callback,
            qos_profile
        )
        self.asr_sub = self.create_subscription(
            ByteMultiArray,
            'asr_topic',
            self.asr_cb,
            qos_profile
        )
        self.subscription
        self.mode_publisher = self.create_publisher(Int32, 'mode_topic', qos_profile)
        
        self.inf = bpu_infer_lib.Infer(False)
        self.inf.load_model(self.kws_bin_path)
        
        self.asr_inf = bpu_infer_lib.Infer(False)
        self.asr_inf.load_model(self.asr_bin_path)
        
        self.samplerate = 16000
        self.channels = 1
        self.bitdepth = pyaudio.paInt16
        self.bufsize = 16000
        self.feat_func = lambda waveform, sr: fbank(
            waveform=paddle.to_tensor(waveform),
            sr=sr,
            frame_shift=10,
            frame_length=25,
            n_mels=80,
        )
        
        with open(self.vocab_path, "r", encoding="utf-8-sig") as f:
            d = eval(f.read())
        self.asr_dict = dict((v, k) for k, v in d.items())
        self.asr_dict[69] = "[PAD]"
        self.asr_dict[68] = "[UNK]"
        self.cur_txt = ""
        self.get_logger().info("FINISHED LOADING MODEL, START AWAKEN..")
        
    def get_ros_param(self, param_name, default_value):
        """Helper function to declare and retrieve a ROS 2 parameter."""
        self.declare_parameter(param_name, default_value)
        return self.get_parameter(param_name).value
    
    def check_params(self):
        """Check if any required parameter is missing and exit if so."""
        missing_params = {
            "LLM API key": self.api_key,
            "KWS bin path": self.kws_bin_path,
            "ASR bin path": self.asr_bin_path,
            "VOCAB json path": self.vocab_path
        }
        for name, value in missing_params.items():
            if value == "None":
                self.get_logger().error(f"{name} not set. Exiting...")
                rclpy.shutdown()
                sys.exit(1)
        
    def switch_mode(self, mode):
        mode_msg = Int32()
        mode_msg.data = mode
        self.mode_publisher.publish(mode_msg)
        if mode == 0:
            self.get_logger().debug("Published mode: 0 (kws)")
        elif mode == 1:
            self.get_logger().debug("Published mode: 1 (asr)")

    def callback(self, msg):
        def audio_trunc(audio_arr):
            thres = 60000
            length = audio_arr.shape[1]
            if length > thres:
                audio_arr = audio_arr[:, :thres]
                return audio_arr
            elif length < thres:
                pad_zero = paddle.zeros((1,thres), dtype=audio_arr.dtype)
                pad_zero[:, :length] = audio_arr
                return pad_zero
            
        data = b''.join(msg.data)
        audio_array = np.frombuffer(data, dtype=np.int16)
        paddle_audio_arr = (audio_array / np.iinfo(np.int16).max).reshape(1,-1).astype(np.float32)
        key_test_load = (audio_trunc(paddle_audio_arr), self.samplerate)
        keyword_feat = self.feat_func(*key_test_load)
        key_input = keyword_feat.unsqueeze(0).numpy()
        random_array = key_input.astype(np.float64).flatten()
                
        received_array = random_array.astype(np.float32).reshape(1, 373, 80)
        self.inf.read_input(received_array, 0)
        self.inf.forward(more=True)
        self.inf.get_output()
        out = self.inf.outputs[0].data
        keyword_score = np.max(out).item()
        self.get_logger().debug(f'keyword_score is: {keyword_score}')
        if keyword_score > 0.00001 and keyword_score < 1.1:
            print("唤醒")
            self.switch_mode(1)

    def asr_cb(self, msg):
        def _normalize(x):
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            return np.squeeze((x - mean) / np.sqrt(var + 1e-5))
        
        new_rate = 16000
        AUDIO_MAXLEN = 30000
        
        data = b''.join(msg.data)
        audio_array = np.frombuffer(data, dtype=np.int16)
        
        samples = round(len(audio_array) * float(new_rate) / 16000)
        new_data = sps.resample(audio_array, samples)
        speech = np.array(new_data, dtype=np.float32)
        speech = _normalize(speech)[None]

        self.asr_inf.read_input(speech, 0)
        self.asr_inf.forward(more=True)
        self.asr_inf.get_output()
        output_arr = self.asr_inf.outputs[0].data.squeeze(-1)
        
        prediction = np.argmax(output_arr, axis=-1)
        _t1 = "".join([self.asr_dict[i] for i in list(prediction[0])])
        cleaned_text = _t1.replace("<pad>", "").replace("|", "")
        self.get_logger().info(f"clean_text: {cleaned_text}")
        self.llm_inf.llm_infer(cleaned_text)
        self.switch_mode(0)
        

def main(args=None):
    rclpy.init(args=args)
    node = AudioSub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
