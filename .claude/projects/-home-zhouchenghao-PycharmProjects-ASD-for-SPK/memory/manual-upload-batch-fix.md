---
name: manual-upload-batch-fix
description: 2026-06-21 数据集构建页面手动上传失败修复及批量上传支持
metadata:
  type: reference
---

V1 `dataset_builder.py` 的 `upload-manual` 端点从单文件 `file: UploadFile` 改为多文件 `files: List[UploadFile]`。前端 `dataset.html` 同步修改：file input 加 `multiple`、拖放/选择事件传递全部文件、字段名统一为 `files`、进度提示支持多文件。

**根因**: 前端字段名 `file` 不匹配后端 `upload-and-split` 期望的 `files`，导致 422。

**Why:** 用户需要同时上传多个音频文件进行数据集构建，原实现只支持单文件上传，且传参名不匹配导致上传失败。

**How to apply:** 已修复并验证（curl 批量上传返回 success: true）。参见项目变更记录 2026-06-21 条目及工作记录 `records/20260621_手动上传批量支持修复.md`。
