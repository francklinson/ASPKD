"""
统一接口版本的Web GUI应用程序
使用新的统一算法接口 - 模块化重构版本

原文件已拆分为 gradio_gui/ 目录下的多个模块：
- managers.py: 算法管理器
- monitor.py: 目录监控器
- export.py: 导出功能
- utils.py: 工具函数
- components.py: UI组件
- offline_handlers.py: 离线模式处理
- online_handlers.py: 在线模式处理
- app.py: 应用组装
"""

import os

from gradio_gui import create_demo
from gradio_gui.app import cleanup_resources

if __name__ == "__main__":
    demo, config, js_autoscroll = create_demo()

    server_config = config.config.get('server', {})
    try:
        demo.launch(
            server_name=server_config.get('server_name', '0.0.0.0'),
            server_port=server_config.get('port', 8002),
            share=server_config.get('share', False),
            inbrowser=server_config.get('inbrowser', True),
            show_error=True,
            head=js_autoscroll
        )
    finally:
        cleanup_resources()
