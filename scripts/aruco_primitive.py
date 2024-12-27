#!/usr/bin/env python3

"""
Copyright (C) 2024 Harsh Davda

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

For inquiries or further information, you may contact:
Harsh Davda
Email: info@opensciencestack.org
"""

from ras_bt_framework.ras_bt_framework.behavior_template.instruction import FunctionalInstruction
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty

@FunctionalInstruction
def aruco_primitive():
    rclpy.init(args=None)
    node = Node('aruco_client')
    client = node.create_client(Empty, 'logging_server')
    request = Empty.Request()
    client.call_async(request)
    node.destroy_node()
    rclpy.shutdown()
