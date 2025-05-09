#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from brkd_image_parse.action import ImgToTable

def main(args=None):
    # 1) Init ROS 2
    rclpy.init(args=args)
    node = rclpy.create_node('img_to_table_client')

    # 2) Create the ActionClient
    client = ActionClient(node, ImgToTable, 'img_to_table')

    try:
        while True:
            # 3) Wait for user to hit ENTER
            input('Press ENTER to send ImgToTable goal (Ctrl+C to exit)…')

            # 4) Wait for the server to be available
            if not client.wait_for_server(timeout_sec=5.0):
                node.get_logger().error('Action server not available, retrying…')
                continue

            # 5) Send an empty goal, with a feedback callback
            goal_msg = ImgToTable.Goal()
            send_goal_future = client.send_goal_async(
                goal_msg,
                feedback_callback=lambda fb: node.get_logger().info(
                    f'🛈 Feedback received: {fb.feedback}'
                )
            )
            # Block until goal is accepted/rejected
            rclpy.spin_until_future_complete(node, send_goal_future)
            goal_handle = send_goal_future.result()
            if not goal_handle.accepted:
                node.get_logger().warn('❗ Goal was rejected by server')
                continue

            node.get_logger().info('✅ Goal accepted, awaiting result…')
            # 6) Wait for the result
            get_result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(node, get_result_future)
            result = get_result_future.result().result

            # 7) Print the returned 2D table
            print('\n🌟 Resulting 2D table:')
            print(result.table_json)

    except KeyboardInterrupt:
        node.get_logger().info('Shutting down client…')
    finally:
        # 8) Clean up
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
