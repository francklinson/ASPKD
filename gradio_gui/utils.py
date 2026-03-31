"""
工具函数 - 设备检测、日志管理、临时文件清理
"""
import os
import gc
import torch
from typing import List, Tuple, Dict

# 可选依赖：内存监控
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def get_available_devices() -> List[Tuple[str, str]]:
    """获取可用的计算设备列表"""
    devices = [("auto", "自动选择 (GPU优先)"), ("cpu", "CPU (纯CPU运行)")]

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            short_name = gpu_name[:30] + "..." if len(gpu_name) > 30 else gpu_name
            devices.append((f"cuda:{i}", f"GPU {i}: {short_name}"))

    return devices


def get_memory_stats(device: str = "auto") -> str:
    """获取内存和显存使用情况"""
    device_info = f"设备: {device}"

    # 解析设备ID
    gpu_id = 0
    if isinstance(device, str) and device.startswith("cuda:"):
        try:
            gpu_id = int(device.split(":")[1])
        except:
            gpu_id = 0

    if not PSUTIL_AVAILABLE:
        gpu_mem_str = "未使用"
        if torch.cuda.is_available():
            try:
                gpu_mem = torch.cuda.memory_allocated(gpu_id) / 1024 / 1024
                gpu_mem_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024 / 1024
                gpu_percent = (gpu_mem / gpu_mem_total) * 100
                gpu_mem_str = f"{gpu_mem:.0f}MB/{gpu_mem_total:.0f}MB ({gpu_percent:.1f}%)"
            except:
                gpu_mem_str = "获取失败"
        return f"{device_info} | CPU: 需安装psutil | GPU: {gpu_mem_str}"

    # CPU内存
    mem = psutil.virtual_memory()
    cpu_mem_str = f"{mem.used/1024/1024/1024:.1f}GB/{mem.total/1024/1024/1024:.1f}GB ({mem.percent}%)"

    # GPU显存
    gpu_mem_str = "未使用"
    if torch.cuda.is_available():
        try:
            gpu_mem = torch.cuda.memory_allocated(gpu_id) / 1024 / 1024
            gpu_mem_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024 / 1024
            gpu_percent = (gpu_mem / gpu_mem_total) * 100
            gpu_mem_str = f"{gpu_mem:.0f}MB/{gpu_mem_total:.0f}MB ({gpu_percent:.1f}%)"
        except:
            gpu_mem_str = "获取失败"

    return f"{device_info} | CPU: {cpu_mem_str} | GPU: {gpu_mem_str}"


def cleanup_directory_recursive(dir_path: str) -> Tuple[int, int, int]:
    """递归清理目录中的所有文件和子目录，返回 (文件数, 目录数, 释放字节数)"""
    file_count, dir_count, total_size = 0, 0, 0

    if not os.path.exists(dir_path):
        return file_count, dir_count, total_size

    try:
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)

            if os.path.isfile(item_path):
                try:
                    file_size = os.path.getsize(item_path)
                    os.remove(item_path)
                    file_count += 1
                    total_size += file_size
                except Exception as e:
                    print(f"[Cleanup] 删除文件失败 {item_path}: {e}")

            elif os.path.isdir(item_path):
                sub_file_count, sub_dir_count, sub_size = cleanup_directory_recursive(item_path)
                file_count += sub_file_count
                dir_count += sub_dir_count
                total_size += sub_size

                try:
                    os.rmdir(item_path)
                    dir_count += 1
                except Exception as e:
                    print(f"[Cleanup] 删除目录失败 {item_path}: {e}")
    except Exception as e:
        print(f"[Cleanup] 清理目录失败 {dir_path}: {e}")

    return file_count, dir_count, total_size


def cleanup_all_temp_files() -> Dict[str, tuple]:
    """清理所有临时文件（slice/、exports/ 和 visualize/ 目录）"""
    result = {'slice': (0, 0, 0), 'exports': (0, 0, 0), 'visualize': (0, 0, 0)}

    for dir_name in ['slice', 'exports', 'visualize']:
        if os.path.exists(dir_name):
            file_count, dir_count, size = cleanup_directory_recursive(dir_name)
            result[dir_name] = (file_count, dir_count, size)
            print(f"[Cleanup] {dir_name}目录: 清理 {file_count} 个文件, {dir_count} 个目录, 释放 {size/1024/1024:.2f} MB")

    total_files = sum(r[0] for r in result.values())
    total_dirs = sum(r[1] for r in result.values())
    total_size = sum(r[2] for r in result.values())
    print(f"[Cleanup] 总计: 清理 {total_files} 个文件, {total_dirs} 个目录, 释放 {total_size/1024/1024:.2f} MB")

    return result


def generate_auto_download_html(zip_path: str) -> str:
    """生成自动下载的HTML脚本"""
    zip_filename = os.path.basename(zip_path)
    return f"""<script>
setTimeout(function() {{
    var link = document.createElement('a');
    link.href = '/file={zip_path}';
    link.download = '{zip_filename}';
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}}, 500);
</script>"""
