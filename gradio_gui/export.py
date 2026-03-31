"""
导出功能类 - 处理结果导出和文件打包
"""
import os
import zipfile
from datetime import datetime
from typing import List, Dict

import pandas as pd


class ExportManager:
    """文件导出管理器"""

    @staticmethod
    def create_excel_report(results: List[Dict], save_dir: str) -> str:
        """创建Excel报告"""
        df = pd.DataFrame(results)
        df_for_excel = df[['filename', 'anomaly_score', 'is_anomaly']]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(save_dir, f"anomaly_detection_results_{timestamp}.xlsx")
        df_for_excel.to_excel(excel_path, index=False)
        return excel_path

    @classmethod
    def create_zip_with_results(cls, zip_path: str, excel_path: str, images: List[str]) -> str:
        """将Excel和图像打包成ZIP文件"""
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(excel_path, os.path.basename(excel_path))
            for img in images:
                if os.path.exists(img):
                    zipf.write(img, os.path.basename(img))
        return zip_path

    @classmethod
    def export_monitor_results(cls, results: List[Dict], exports_dir: str = "exports") -> str:
        """导出监控结果为Excel"""
        os.makedirs(exports_dir, exist_ok=True)

        export_data = [{
            'filename': r['filename'],
            'anomaly_score': r['anomaly_score'],
            'is_anomaly': r['is_anomaly'],
            'timestamp': r['timestamp'],
            'filepath': r['filepath']
        } for r in results]

        return cls.create_excel_report(export_data, exports_dir)

    @classmethod
    def create_monitor_zip(cls, results: List[Dict], algorithm_name: str,
                          exports_dir: str = "exports") -> tuple:
        """打包监控结果为ZIP，返回 (zip_path, file_count, image_count)"""
        os.makedirs(exports_dir, exist_ok=True)

        export_data = []
        heatmap_paths = []

        for r in results:
            export_data.append({
                'filename': r['filename'],
                'anomaly_score': r['anomaly_score'],
                'is_anomaly': r['is_anomaly'],
                'timestamp': r['timestamp'],
                'filepath': r['filepath']
            })
            if r.get('heatmap_path') and os.path.exists(r['heatmap_path']):
                heatmap_paths.append(r['heatmap_path'])
            if r.get('processed_images'):
                for img_path in r['processed_images']:
                    if os.path.exists(img_path) and img_path not in heatmap_paths:
                        heatmap_paths.append(img_path)

        excel_path = cls.create_excel_report(export_data, exports_dir)

        zip_filename = f"monitor_results_{algorithm_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = os.path.join(exports_dir, zip_filename)
        cls.create_zip_with_results(zip_path, excel_path, heatmap_paths)

        return zip_path, len(export_data), len(heatmap_paths)
