#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import base64
import cv2
import subprocess
import os
import json
from openai import OpenAI
import io
# import pyaudio
from pydub import AudioSegment
from pydub.playback import play
# from playsound import playsound

def load_image_to_base64(image_path):
    """Load an image from a file and encode it to base64."""
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    jpg_b64 = base64.b64encode(image_data).decode('utf-8')
    return f"data:image/jpeg;base64,{jpg_b64}"


def encode_image_to_base64(image):
    """
    Encode a CV2 image (numpy array) to JPEG base64 Data URI.
    Returns (success: bool, data_uri or None)
    """
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        return False, None
    jpg_b64 = base64.b64encode(buffer).decode('utf-8')
    return True, f"data:image/jpeg;base64,{jpg_b64}"

client = OpenAI()
# p = pyaudio.PyAudio()
# stream = p.open(
#     format=pyaudio.paInt16,     # 16-bit samples
#     channels=1,                  # mono
#     rate=24000,                  # match your TTS model’s sample rate
#     output=True,
# )

def speak(text: str, instructions: str):
    """Invoke the system TTS engine to speak the given text."""
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text,
        instructions=instructions,
    ) as response:

        mp3_bytes = response.read()
        buffer = io.BytesIO(mp3_bytes)
        audio = AudioSegment.from_file(buffer, format="mp3")
        play(audio)

        # response.stream_to_file("tmp_voice.mp3")
        # print("streamed to file")
        # playsound("tmp_voice.mp3")
        # print("played sound using playsound")

        # song = AudioSegment.from_mp3("tmp_voice.mp3")
        # print('played sound using  pydub')
        # play(song)

class ImgTalker(Node):
    def __init__(self):
        super().__init__('img_talker')
        self.bridge = CvBridge()
        self.latest_image = None
        self.lock = threading.Lock()

        # Subscribe to raw image frames
        self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10)

        # Periodic timer to analyze the latest image every 5 seconds
        self.create_timer(5.0, self.timer_callback)

        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.get_logger().warn('OpenAI API key not set!')

        # System prompt for vision reasoning
        self.prompt_system = [
            {"role": "system", "content":
             "You are a vision and reasoning assistant. Say 'yeah' when you see a hand"}
        ]

        # Register functions for GPT function-calling
        self.functions = [
            {
                "name": "speak",
                "type": "function",
                "description": "Invoke TTS to speak the provided text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to speak."},
                        "instructions": {"type": "string", "description": "A description of the tone and mood of the speech."}
                    },
                    "required": ["text", "instructions"],
                    "additionalProperties": False
                }
            }
        ]

    def image_callback(self, msg: Image):
        """Callback for incoming camera frames."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

    def timer_callback(self):
        """Periodically send the latest image to GPT and react if function call is returned."""
        with self.lock:
            if self.latest_image is None:
                return
            image_copy = self.latest_image.copy()

        success, data_uri = encode_image_to_base64(image_copy)
        if not success:
            self.get_logger().warn('Failed to encode image for GPT API')
            return

        try:
            response = client.responses.create(
                model="gpt-4.1",
                input=self.prompt_system + [
                    {"role": "user", "content": [
                        {"type": "input_text", "text": "Explain the image and call the TTS function to say say `yeah` when you see a human hand. Don't speak out loudly your thought. Just speak when you should"},
                        {"type": "input_image", "image_url": data_uri}
                    ]}
                    ],
                tools=self.functions,
            )
            print(response.output)
            for message in response.output:
                if message.type == "function_call" and  message.name == "speak":
                    args = json.loads(message.arguments)
                    text_to_speak = args.get('text', '')
                    instruction_to_speak = args.get('instructions', '')
                    self.get_logger().info(f"Speaking via TTS: {text_to_speak}")
                    speak(text_to_speak, instruction_to_speak)
                    
        except Exception as e:
            self.get_logger().error(f'GPT API call failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ImgTalker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
