import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from std_msgs.msg import String, Int32, ByteMultiArray
import numpy as np
import pyaudio
import threading


ASR_AUDIO_MAXLEN = 30000


class AudioPub(Node):
    def __init__(self):
        super().__init__('audio_pub')

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.publisher_ = self.create_publisher(ByteMultiArray, 'kws_topic', qos_profile)
        self.asr_pub = self.create_publisher(ByteMultiArray, 'asr_topic', qos_profile)

        self.mode = 0  
        self.subscription = self.create_subscription(
            Int32,
            'mode_topic',
            self.mode_callback,
            qos_profile
        )
        self.msg = ByteMultiArray()

        self.thread = threading.Thread(target=self.publish_loop, daemon=True)
        self.thread.start()
        
    def _initialize_audio_stream(self, samplerate=16000, channels=1, bitdepth=pyaudio.paInt16):
        """Initialize the PyAudio object and the stream."""
        re = pyaudio.PyAudio()
        stream = re.open(format=bitdepth, channels=channels, rate=samplerate, input=True, frames_per_buffer=16000)
        return re, stream

    def mode_callback(self, msg: String):
        self.mode = msg.data
        self.get_logger().info(f"Mode changed to: {self.mode}")
        
    def _publish(self, stream, bufsize, pub):
        if bufsize == ASR_AUDIO_MAXLEN:
            self.get_logger().info("Please start speaking:")
        data = stream.read(bufsize, exception_on_overflow=False)
        if bufsize == ASR_AUDIO_MAXLEN:
            self.get_logger().info("Speaking ends now..")
        self.msg.data = [bytes([b]) for b in data]
        pub.publish(self.msg)

    def publish_loop(self):
        samplerate = 16000
        channels = 1
        bitdepth = pyaudio.paInt16
        last_asr = False
        self.get_logger().info("开始唤醒中..")
        re, stream = self._initialize_audio_stream(samplerate, channels, bitdepth)
        
        try:
            while rclpy.ok():
                if self.mode == 0: 
                    bufsize = 16000
                    self._publish(stream, bufsize, self.publisher_)
                    last_asr = False
                elif self.mode == 1 and not last_asr: 
                    bufsize = ASR_AUDIO_MAXLEN
                    self._publish(stream, bufsize, self.asr_pub)
                    last_asr = True
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            re.terminate()


def main(args=None):
    rclpy.init(args=args)
    node = AudioPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
