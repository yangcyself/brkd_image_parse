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
import random
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
        self.count = 0

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
"""
You are a vision and reasoning assistant. You help a kid to build the mega blocks. 
You are witty, slightly sassy, and always impeccably polite
You calls the tool `speak` so that you can talk to the kid. The kid can not hear you unless you call the `speak` function. 
The responses without `speak` are just for you and the parent to understand.
You have a robot arm that helps the kid to build legos. You assist the kid but the kid might has a different idea. So you need to `speak` and communitcate.
Keep the speak sentence short, but also witty and clever. Each response should take about 5 seconds to read aloud unless the user is asking for something more complex.
While the child does use legos, you can pretend they're working on a major project just happen to be using legos.
Don't be excessively verbose, and don't be too \"nice\" or \"sweet\". Use comebacks and sarcasm, sometimes quite excessively so.
Don't start with building with legos or anything like that. Just greet broadly and ask how you can help.
In conversation, try not to say Legos or Lego bricks too often just to keep the pretense that this is a major project, but it is okay to say it once in a while.
The thing is the user has a  robot arm with which to build, and this arm will pick up and place the bricks, so you can say that the user is using a robot arm to build with.
But for now, we can only build 2D mosaic-like vertical but flat structures, so don't say anything about building 3D structures.
Here are some examples of simple designs we can build: cat, up arrow, down arrow, house, human, heart, arc
But of course, keep the pretense that this is a major project, and don't say anything about building 2D structures.
"""}
        ]
        self.history = []

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
        self.count += 1

        success, data_uri = encode_image_to_base64(image_copy)
        if not success:
            self.get_logger().warn('Failed to encode image for GPT API')
            return

        try:
            if self.count % 5 == 0:
                self.history += [
                        {"role": "user", "content": [
                            {"type": "input_text", "text": 
                             "please invoke the speak function to say something, maybe guess what the kid is trying to build. "
                             },
                            {"type": "input_image", "image_url": data_uri}
                        ]}
                ]
            elif self.count % 5 == 1:
                target = random.choice(["elephant", "cat", "house", "mountail", "car"])
                self.history += [
                        {"role": "user", "content": [
                            {"type": "input_text", 
                             "text": 
                              (f"The goal is to build an {target}, does it look like? ")
                             },
                            {"type": "input_image", "image_url": data_uri}
                        ]}
                ]
            elif self.count % 5 == 2:
                self.history += [
                        {"role": "user", "content": [
                            {"type": "input_text", 
                             "text": 
                              ("What do you think will be possible to improve the building?")
                             },
                            {"type": "input_image", "image_url": data_uri}
                        ]}
                ]
            elif self.count % 5 == 3:
                self.history += [
                        {"role": "user", "content": [
                            {"type": "input_text", 
                             "text": 
                              ("Analyze the current status, which part is which? No need to speak out loudly your thought. But if there is anything interesting, please `speak` it out. ")
                             },
                            {"type": "input_image", "image_url": data_uri}
                        ]}
                ]
            else:
                self.history += [
                        {"role": "user", "content": [
                            {"type": "input_text", 
                             "text": 
                              ("How good is the mega block building. What's the difference compared to the previous image. Don't speak out loudly your thought. "
                              "Just `speak` when you should. Don't be silent all the time but don't be too repetitive. ")
                             },
                            {"type": "input_image", "image_url": data_uri}
                        ]}
                ]
            response = client.responses.create(
                model="gpt-4.1",
                input=self.prompt_system + self.history,
                tools=self.functions,
            )
            print(response)
            for message in response.output:
                if message.type == "function_call" and  message.name == "speak":
                    args = json.loads(message.arguments)
                    text_to_speak = args.get('text', '')
                    instruction_to_speak = args.get('instructions', '')
                    self.get_logger().info(f"Speaking via TTS: {text_to_speak}")
                    speak(text_to_speak, instruction_to_speak)
                elif message.type != "function_call":
                    self.history.append(message)
            self.history = self.history[-5:]  # Keep the last 5 messages
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
