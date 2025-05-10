#!/usr/bin/env python3
"""
ROS2 node acting as an Action Server 'img_to_table'.
- Subscribes to '/image_raw' and stores the latest sensor_msgs/Image
- When a goal is received, encodes the current image as base64 JPEG
  and sends it to the OpenAI Chat Completion API for analysis.
- Expects a JSON-formatted 2D array of mega-block colors/types as the response.
- Returns the JSON string in the action result.

Dependencies:
  - rclpy
  - sensor_msgs
  - cv_bridge
  - opencv-python (cv2)
  - openai
  - your custom action definition in my_robot_interfaces/action/ImgToTable.action

Ensure you define the ImgToTable action as follows:

# ImgToTable.action
# --- (Goal section - empty) ---
# --- (Result section) ---
string table_json
# --- (Feedback section - optional) ---
string feedback
"""
import os
import threading
import json
import base64

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from brkd_image_parse.action import ImgToTable
import openai
import cv2


def load_image_to_base64(image_path):
    """Load an image from a file and encode it to base64."""
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    jpg_b64 = base64.b64encode(image_data).decode('utf-8')
    data_uri = f"data:image/jpeg;base64,{jpg_b64}"
    return data_uri

def encode_image_to_base64(image):
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        return False, None

    jpg_b64 = base64.b64encode(buffer).decode('utf-8')
    data_uri = f"data:image/jpeg;base64,{jpg_b64}"
    return True, data_uri

class ImgToTableServer(Node):
    def __init__(self):
        super().__init__('img_to_table_server')

        # Subscribe to image topic
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.bridge = CvBridge()
        self.latest_image = None
        self.lock = threading.Lock()

        # Initialize Action Server
        self.action_server = ActionServer(
            self,
            ImgToTable,
            'img_to_table',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

        # Set your OpenAI API key in the environment or here directly
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            self.get_logger().warn('OpenAI API key not set!')

        # with open("/home/chenyu/workspace/brkdai_ws/src/brkd_image_parse/data/example1.png", "rb") as exampple_1_f:
        #     example_1 = exampple_1_f.read()
        # with open("/home/chenyu/workspace/brkdai_ws/src/brkd_image_parse/data/example2.png", "rb") as exampple_2_f:
        #     example_2 = exampple_2_f.read()

        self.prompt_system = [
            {"type": "input_text", "text": 
"""
You are a vision and reasoning assistant.  Given an image of Mega Blocks, return a JSON object with the following fields.
The user teach you how to analyze the image with examples and you understand and follow the examples
width: should be the maximum from left to right
height: the total number of layers
table: a 2D table where each element is the color or type of each block cell. 

"""
             },
        ]
        self.user_example1 = [
            {"type": "input_text", "text": 
"""
For example in the following image the two blocks are stacked together. Note that in your answer connection knob is not counted as a block. And note the unit of the height is half of the unit of the height.
So you should have the following answer for the image:
{"width": 2, "height": 3, "table": [["empty", "blue"], ["empty", "blue"], ["purple", "purple"]]}
"""
             },
            {
                "type": "input_image",
                "image_url": load_image_to_base64(
                    "/home/chenyu/workspace/brkdai_ws/src/brkd_image_parse/data/example1.png"
                ),
            },
        ]
        self.user_example2 = [
            {"type": "input_text", "text": 
"""
For example in the following image the two blocks are stacked together. Note that in your answer connection knob is not counted as a block. And note the unit of the height is half of the unit of the height.
So you should have the following answer for the image:
{"width": 3, "height": 3, "table": [["empty", "red", "red"], ["green", "green", "empty"], ["green", "green", "empty"]]}
"""
             },
            {
                "type": "input_image",
                "image_url": load_image_to_base64(
                    "/home/chenyu/workspace/brkdai_ws/src/brkd_image_parse/data/example2.png"
                ),
            },
        ]
        self.user_example3 = [
            {"type": "input_text", "text": 
"""
For example for the following image you should have the following answer:
{"width": 6, "height": 7, "table": [["empty", "empty", "empty", "blue", "blue", "blue"], 
                                    ["empty", "empty", "empty", "blue", "blue", "blue"], 
                                    ["blue", "blue", "blue", "blue", "empty", "empty"], 
                                    ["empty", "red", "empty", "red", "empty", "empty"], 
                                    ["blue", "blue", "blue", "blue", "empty", "empty"], 
                                    ["darkblue", "darkblue", "darkblue", "darkblue", "empty", "empty"]]}
"""
             },
            {
                "type": "input_image",
                "image_url": load_image_to_base64(
                    "/home/chenyu/workspace/brkdai_ws/src/brkd_image_parse/data/example3.png"
                ),
            },
        ]

    def image_callback(self, msg: Image):
        """Store the latest image frame"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

    def goal_callback(self, goal_request):
        self.get_logger().info('Received img_to_table goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received img_to_table cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing img_to_table action...')

        # Acquire the latest image safely
        with self.lock:
            if self.latest_image is None:
                self.get_logger().error('No image received yet! Aborting.')
                goal_handle.abort()
                result = ImgToTable.Result()
                result.table_json = ''
                result.feedback = 'No image available'
                return result
            image_to_process = self.latest_image.copy()

        # Encode as JPEG and then base64
        success, data_uri = encode_image_to_base64(image_to_process)
        if not success:
            self.get_logger().error('Failed to encode image to base64')
            goal_handle.abort()
            result = ImgToTable.Result()
            result.table_json = ''
            result.feedback = 'Image encoding failed'
            return result

        # Function to encode the image
        user_prompt = (
            f"Please analyze the following image. "
        )

        try:
            client = openai.OpenAI()
            response = client.responses.create(
                model="gpt-4.1",
                input=[
                    {
                        "role": "system",
                        "content": self.prompt_system,
                    },
                    {
                        "role": "user",
                        "content": 
                            self.user_example1+self.user_example2+[
                            {"type": "input_text", "text": user_prompt},
                            {
                                "type": "input_image",
                                "image_url": data_uri,
                            },
                        ],
                    }
                ]
            )
        except Exception as e:
            self.get_logger().error(f'OpenAI API error: {e}')
            goal_handle.abort()
            result = ImgToTable.Result()
            result.table_json = ''
            result.feedback = f'OpenAI error: {e}'
            return result

        # Extract content
        content = response.output_text

        # Validate JSON
        try:
            table = json.loads(content)
            table_json = json.dumps(table)
        except json.JSONDecodeError:
            self.get_logger().warn('Response was not valid JSON; returning raw text')
            table_json = content

        # Succeed and return
        goal_handle.succeed()

        result = ImgToTable.Result()
        result.table_json = table_json
        result.feedback = 'Success'
        return result


from rclpy.action import ActionClient
def main(args=None):
    rclpy.init(args=args)
    node = ImgToTableServer()

    # Launch the server's spin in a daemon thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Create an action client on the same node
    client = ActionClient(node, ImgToTable, 'img_to_table')

    try:
        while True:
            input("\nPress ENTER to send ImgToTable goal (Ctrl+C to exit)…")

            # Wait for server
            if not client.wait_for_server(timeout_sec=2.0):
                node.get_logger().error('Action server not available, retrying…')
                continue

            # Build and send an empty goal
            goal_msg = ImgToTable.Goal()
            # send_goal_future = client.send_goal_async(
            #     goal_msg,
            #     feedback_callback=lambda fb: node.get_logger().info(f"Feedback: {fb.feedback}")
            # )
            goal_handle = client.send_goal(
                goal_msg,
                feedback_callback=lambda fb: node.get_logger().info(f"Feedback: {fb.feedback}")
            )
            # Wait for the goal to be accepted
            # rclpy.spin_until_future_complete(node, send_goal_future)
            # goal_handle = send_goal_future.result()
            # if not goal_handle.accepted:
            #     node.get_logger().warn('Goal was rejected by server')
            #     continue

            # Wait for the result
            # result_future = goal_handle.get_result_async()
            # rclpy.spin_until_future_complete(node, result_future)
            # result = result_future.result().result

            result = client._get_result(goal_handle)

            # Print out the 2D table returned
            print("Resulting 2D table:")
            for row in result.table:
                print("  ", row)

    except KeyboardInterrupt:
        node.get_logger().info('Shutting down…')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
