#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shazam 音频指纹识别 - 快速启动脚本

提供命令行接口，方便直接使用 Shazam 功能

用法:
    # 初始化数据库
    python quickstart.py init

    # 添加参考音频
    python quickstart.py add ref/audio.wav --name "参考音频1"
    python quickstart.py add ref/ --pattern "*.wav"

    # 识别音频
    python quickstart.py recognize query.wav

    # 定位音频片段
    python quickstart.py locate long_audio.wav --ref ref.wav

    # 列出所有参考音频
    python quickstart.py list

    # 批量处理
    python quickstart.py batch query/ --output results.json
"""

import argparse
import json
import sys
import os
from pathlib import Path

# 确保能导入 Shazam 模块
sys.path.insert(0, str(Path(__file__).parent))

from shazam import AudioFingerprinter, create_fingerprint_db, batch_recognize


def cmd_init(args):
    """初始化数据库"""
    print("正在初始化数据库...")
    with AudioFingerprinter() as fp:
        fp.init_database()
    print("✓ 数据库初始化完成")


def cmd_add(args):
    """添加参考音频"""
    with AudioFingerprinter() as fp:
        if os.path.isdir(args.path):
            # 批量添加目录
            import glob
            pattern = args.pattern or "*.wav"
            files = glob.glob(os.path.join(args.path, pattern))
            print(f"找到 {len(files)} 个文件，正在添加...")
            ids = fp.add_references(files)
            print(f"✓ 成功添加 {len(ids)} 首音频")
        else:
            # 添加单个文件
            music_id = fp.add_reference(args.path, name=args.name)
            print(f"✓ 已添加: {args.path} (ID: {music_id})")


def cmd_recognize(args):
    """识别音频"""
    with AudioFingerprinter() as fp:
        result = fp.recognize(args.path, threshold=args.threshold)

        if result.matched:
            print(f"✓ 匹配成功!")
            print(f"  名称: {result.name}")
            print(f"  偏移: {result.offset:.2f}s")
            print(f"  置信度: {result.confidence}")
        else:
            print(f"✗ 未匹配 (置信度: {result.confidence})")


def cmd_locate(args):
    """定位音频片段"""
    with AudioFingerprinter() as fp:
        # 如果使用参考名称，需要先确认在库中
        if args.ref_path:
            fp.add_reference(args.ref_path, name="temp_ref")

        location = fp.locate(
            args.path,
            reference_path=args.ref_path,
            reference_name=args.ref_name,
            threshold=args.threshold
        )

        if location.found:
            print(f"✓ 找到片段!")
            print(f"  位置: {location.start_time:.2f}s ~ {location.end_time:.2f}s")
            print(f"  时长: {location.end_time - location.start_time:.2f}s")
            print(f"  置信度: {location.confidence}")
        else:
            print(f"✗ 未找到匹配片段")


def cmd_list(args):
    """列出所有参考音频"""
    with AudioFingerprinter() as fp:
        refs = fp.get_all_references()

        if not refs:
            print("指纹库为空")
            return

        print(f"共 {len(refs)} 首参考音频:")
        print("-" * 60)
        print(f"{'ID':<6} {'名称':<30} {'哈希数':<10} {'路径':<20}")
        print("-" * 60)
        for ref in refs:
            name = ref['name'][:28] + '..' if len(ref['name']) > 30 else ref['name']
            path = ref['path'][:18] + '..' if len(ref['path']) > 20 else ref['path']
            print(f"{ref['music_id']:<6} {name:<30} {ref['hash_count']:<10} {path:<20}")


def cmd_delete(args):
    """删除参考音频"""
    with AudioFingerprinter() as fp:
        if fp.delete_reference(args.id):
            print(f"✓ 已删除 ID={args.id}")
        else:
            print(f"✗ 删除失败")


def cmd_batch(args):
    """批量处理"""
    import glob

    # 获取所有查询文件
    if os.path.isdir(args.input):
        pattern = args.pattern or "*.wav"
        query_files = glob.glob(os.path.join(args.input, pattern))
    else:
        query_files = [args.input]

    print(f"批量识别 {len(query_files)} 个文件...")

    results = batch_recognize(query_files, threshold=args.threshold)

    # 整理结果
    output = []
    for path, result in zip(query_files, results):
        output.append({
            'file': path,
            'matched': result.matched,
            'name': result.name,
            'offset': result.offset,
            'confidence': result.confidence
        })

    # 输出结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"✓ 结果已保存到: {args.output}")
    else:
        # 打印到控制台
        for item in output:
            status = "✓" if item['matched'] else "✗"
            print(f"{status} {item['file']}: {item['name']} (置信度: {item['confidence']})")


def main():
    parser = argparse.ArgumentParser(
        description="Shazam 音频指纹识别工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    %(prog)s init                                    # 初始化数据库
    %(prog)s add ref.wav --name "参考音频"           # 添加单个音频
    %(prog)s add ref/ --pattern "*.wav"              # 批量添加
    %(prog)s recognize query.wav                     # 识别音频
    %(prog)s locate long.wav --ref ref.wav           # 定位片段
    %(prog)s list                                    # 列出参考音频
    %(prog)s batch query/ --output results.json      # 批量处理
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # init 命令
    parser_init = subparsers.add_parser('init', help='初始化数据库')

    # add 命令
    parser_add = subparsers.add_parser('add', help='添加参考音频')
    parser_add.add_argument('path', help='音频文件或目录路径')
    parser_add.add_argument('--name', help='音频名称（单个文件时使用）')
    parser_add.add_argument('--pattern', help='文件匹配模式（目录时使用，默认*.wav）')

    # recognize 命令
    parser_rec = subparsers.add_parser('recognize', help='识别音频')
    parser_rec.add_argument('path', help='查询音频路径')
    parser_rec.add_argument('--threshold', type=int, default=10, help='匹配阈值')

    # locate 命令
    parser_loc = subparsers.add_parser('locate', help='定位音频片段')
    parser_loc.add_argument('path', help='长音频路径')
    parser_loc.add_argument('--ref-path', help='参考音频路径')
    parser_loc.add_argument('--ref-name', help='参考音频名称（已在库中）')
    parser_loc.add_argument('--threshold', type=int, default=10, help='匹配阈值')

    # list 命令
    parser_list = subparsers.add_parser('list', help='列出参考音频')

    # delete 命令
    parser_del = subparsers.add_parser('delete', help='删除参考音频')
    parser_del.add_argument('id', type=int, help='音乐ID')

    # batch 命令
    parser_batch = subparsers.add_parser('batch', help='批量识别')
    parser_batch.add_argument('input', help='输入文件或目录')
    parser_batch.add_argument('--pattern', help='文件匹配模式')
    parser_batch.add_argument('--output', help='输出JSON文件路径')
    parser_batch.add_argument('--threshold', type=int, default=10, help='匹配阈值')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # 执行对应命令
    commands = {
        'init': cmd_init,
        'add': cmd_add,
        'recognize': cmd_recognize,
        'locate': cmd_locate,
        'list': cmd_list,
        'delete': cmd_delete,
        'batch': cmd_batch,
    }

    if args.command in commands:
        try:
            commands[args.command](args)
        except Exception as e:
            print(f"错误: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
