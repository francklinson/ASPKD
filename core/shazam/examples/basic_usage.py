# -*- coding: utf-8 -*-
"""
Shazam 模块使用示例

演示如何使用 AudioFingerprinter 进行音频指纹识别和定位
"""

from core.shazam import AudioFingerprinter, RecognitionResult, LocationResult
from core.shazam import create_fingerprint_db, batch_recognize, batch_locate


def example_1_basic_usage():
    """示例1: 基础用法 - 添加参考音频并识别"""
    print("=" * 50)
    print("示例1: 基础用法")
    print("=" * 50)

    # 创建指纹识别器（自动加载默认配置）
    fingerprinter = AudioFingerprinter()

    # 初始化数据库
    fingerprinter.init_database()

    # 添加参考音频到指纹库
    ref_path = "dataset/key/song1.wav"
    music_id = fingerprinter.add_reference(ref_path, name="歌曲1")
    print(f"已添加参考音频，ID: {music_id}")

    # 识别查询音频
    query_path = "dataset/query/clip.wav"
    result = fingerprinter.recognize(query_path)
    print(f"识别结果: {result}")

    # 关闭连接
    fingerprinter.close()


def example_2_locate_audio():
    """示例2: 在长音频中定位片段"""
    print("\n" + "=" * 50)
    print("示例2: 音频定位")
    print("=" * 50)

    with AudioFingerprinter() as fp:
        # 添加参考音频
        fp.add_reference("ref/渡口片段10s.wav", name="渡口片段")

        # 在长音频中定位
        location = fp.locate(
            long_audio_path="原始数据/long_audio.wav",
            reference_name="渡口片段"
        )

        if location.found:
            print(f"找到片段位置: {location.start_time:.2f}s ~ {location.end_time:.2f}s")
            print(f"置信度: {location.confidence}")
        else:
            print("未找到匹配片段")


def example_3_batch_processing():
    """示例3: 批量处理"""
    print("\n" + "=" * 50)
    print("示例3: 批量处理")
    print("=" * 50)

    # 批量创建指纹库
    ids = create_fingerprint_db("dataset/key/", pattern="*.wav")
    print(f"已添加 {len(ids)} 首歌曲到指纹库")

    # 批量识别
    query_files = ["dataset/query/q1.wav", "dataset/query/q2.wav"]
    results = batch_recognize(query_files, threshold=10)

    for path, result in zip(query_files, results):
        status = "✓" if result.matched else "✗"
        print(f"{status} {path}: {result.name} (置信度: {result.confidence})")


def example_4_4asd_integration():
    """示例4: ASD项目集成 - 替换原有的MFCC+DTW定位"""
    print("\n" + "=" * 50)
    print("示例4: ASD项目集成")
    print("=" * 50)

    def process_audio_with_shazam(audio_path: str, ref_audio_path: str) -> dict:
        """
        使用 Shazam 替代 MFCC+DTW 进行音频定位

        Args:
            audio_path: 待处理音频
            ref_audio_path: 参考音频

        Returns:
            包含 offset 和 duration 的字典
        """
        with AudioFingerprinter() as fp:
            # 确保参考音频在指纹库中
            fp.add_reference(ref_audio_path, name="ref")

            # 定位
            location = fp.locate(audio_path, reference_path=ref_audio_path)

            if location.found:
                return {
                    "offset": location.start_time,
                    "duration": location.end_time - location.start_time,
                    "confidence": location.confidence,
                    "success": True
                }
            else:
                return {
                    "offset": 0,
                    "duration": 0,
                    "confidence": 0,
                    "success": False
                }

    # 使用示例
    result = process_audio_with_shazam(
        "原始数据/test.wav",
        "ref/渡口片段10s.wav"
    )
    print(f"定位结果: {result}")


def example_5_database_management():
    """示例5: 数据库管理"""
    print("\n" + "=" * 50)
    print("示例5: 数据库管理")
    print("=" * 50)

    with AudioFingerprinter() as fp:
        # 列出所有参考音频
        refs = fp.get_all_references()
        print(f"指纹库中共有 {len(refs)} 首参考音频:")
        for ref in refs:
            print(f"  ID={ref['music_id']}, 名称={ref['name']}, 哈希数={ref['hash_count']}")

        # 删除指定音频（谨慎操作）
        # fp.delete_reference(music_id=1)

        # 清空数据库（极度谨慎）
        # fp.clear_database()


def example_6_custom_config():
    """示例6: 使用自定义配置"""
    print("\n" + "=" * 50)
    print("示例6: 自定义配置")
    print("=" * 50)

    # 使用指定配置文件
    fingerprinter = AudioFingerprinter(config_path="Shazam/config/config.yaml")

    # 或使用上下文管理器
    with AudioFingerprinter() as fp:
        result = fp.recognize("query.wav")
        print(result)


if __name__ == "__main__":
    # 运行示例
    try:
        example_1_basic_usage()
    except Exception as e:
        print(f"示例1失败（可能是缺少测试文件）: {e}")

    try:
        example_2_locate_audio()
    except Exception as e:
        print(f"示例2失败: {e}")

    try:
        example_3_batch_processing()
    except Exception as e:
        print(f"示例3失败: {e}")

    try:
        example_4_4asd_integration()
    except Exception as e:
        print(f"示例4失败: {e}")

    try:
        example_5_database_management()
    except Exception as e:
        print(f"示例5失败: {e}")

    try:
        example_6_custom_config()
    except Exception as e:
        print(f"示例6失败: {e}")
