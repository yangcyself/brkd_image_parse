cmake_minimum_required(VERSION 3.11)
project(brkd_image_parse)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# runtime dependencies
find_package(rclpy REQUIRED)
# for generating the Python action interface
find_package(rosidl_default_generators REQUIRED)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ImgToTable.action"
  DEPENDENCIES builtin_interfaces  # if you need timestamps, etc.
)

# If you have any standalone scripts (e.g. your node),
# install them here. Adjust paths as needed:
install(PROGRAMS
  scripts/brkd_image_parse_node.py
  scripts/brkd_image_parse_client.py
  scripts/mega_block_state_estimator.py
  DESTINATION lib/${PROJECT_NAME}
)

# Also install the raw .action file so it ends up in the share directory
install(DIRECTORY action
  DESTINATION share/${PROJECT_NAME}
)


ament_package()
