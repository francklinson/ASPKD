"""
参考音频库管理 API
用于管理 Shazam 音频指纹库的参考音频
"""
import os
import shutil
import time
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from core.shazam import AudioFingerprinter

router = APIRouter()


def log_operation(operation: str, details: str = "", status: str = "INFO"):
    """记录操作日志"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [ReferenceAudio] [{status}] {operation} | {details}")


class ReferenceAudioInfo(BaseModel):
    """参考音频信息"""
    music_id: int
    name: str
    path: str
    hash_count: int


class ReferenceAudioListResponse(BaseModel):
    """参考音频列表响应"""
    references: List[ReferenceAudioInfo]
    total: int


class AddReferenceResponse(BaseModel):
    """添加参考音频响应"""
    success: bool
    music_id: Optional[int] = None
    message: str


class DeleteReferenceResponse(BaseModel):
    """删除参考音频响应"""
    success: bool
    message: str


# 参考音频存储目录
REF_AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ref")
os.makedirs(REF_AUDIO_DIR, exist_ok=True)


def get_fingerprinter():
    """获取指纹识别器实例"""
    try:
        return AudioFingerprinter()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"无法连接指纹数据库: {str(e)}")


@router.get("/list", response_model=ReferenceAudioListResponse)
async def get_reference_list():
    """
    获取所有参考音频列表
    
    返回指纹库中所有参考音频的详细信息
    使用单次查询优化性能，避免N+1查询问题
    """
    start_time = time.time()
    log_operation("LIST_START", "开始获取参考音频列表")
    
    try:
        with get_fingerprinter() as fp:
            connector = fp._get_connector()
            hp = fp.hp
            
            # 使用单次查询获取所有参考音频及其哈希数量
            sql = f"""
                SELECT 
                    m.{hp.fingerprint.database.tables.music.column.music_id},
                    m.{hp.fingerprint.database.tables.music.column.music_name},
                    m.{hp.fingerprint.database.tables.music.column.music_path},
                    COUNT(fp.{hp.fingerprint.database.tables.finger_prints.column.id_fp}) as hash_count
                FROM {hp.fingerprint.database.tables.music.name} m
                LEFT JOIN {hp.fingerprint.database.tables.finger_prints.name} fp
                    ON m.{hp.fingerprint.database.tables.music.column.music_id} = 
                       fp.{hp.fingerprint.database.tables.finger_prints.column.music_id_fk}
                GROUP BY m.{hp.fingerprint.database.tables.music.column.music_id}
                ORDER BY m.{hp.fingerprint.database.tables.music.column.music_id}
            """
            
            connector.cursor.execute(sql)
            results = connector.cursor.fetchall()
            
            # 转换为响应格式
            ref_list = [
                ReferenceAudioInfo(
                    music_id=music_id,
                    name=name,
                    path=path,
                    hash_count=hash_count or 0
                )
                for music_id, name, path, hash_count in results
            ]
            
            elapsed_time = (time.time() - start_time) * 1000
            log_operation(
                "LIST_SUCCESS", 
                f"成功获取 {len(ref_list)} 条参考音频记录，耗时 {elapsed_time:.2f}ms"
            )
            
            return ReferenceAudioListResponse(
                references=ref_list,
                total=len(ref_list)
            )
    except Exception as e:
        elapsed_time = (time.time() - start_time) * 1000
        log_operation(
            "LIST_ERROR", 
            f"获取失败: {str(e)}，耗时 {elapsed_time:.2f}ms",
            "ERROR"
        )
        raise HTTPException(status_code=500, detail=f"获取参考音频列表失败: {str(e)}")


@router.post("/upload", response_model=AddReferenceResponse)
async def upload_reference_audio(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None)
):
    """
    上传并添加参考音频到指纹库
    
    注意：文件不会保存到 ref/ 文件夹，只生成指纹并存储到数据库

    - **file**: 音频文件 (支持 wav, mp3, flac 等格式)，时长不得超过30秒
    - **name**: 音频名称（可选，默认使用文件名）
    """
    import librosa
    import tempfile

    start_time = time.time()
    log_operation(
        "UPLOAD_START",
        f"开始上传文件: {file.filename}, 自定义名称: {name or '无'}"
    )

    # 验证文件格式
    allowed_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        log_operation(
            "UPLOAD_ERROR",
            f"不支持的文件格式: {ext}",
            "ERROR"
        )
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式，仅支持 {allowed_extensions}"
        )

    # 确定显示名称和文件名
    if name:
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        if not safe_name:
            safe_name = "reference"
        filename = f"{safe_name}{ext}"
        display_name = name
    else:
        filename = file.filename
        display_name = os.path.splitext(filename)[0]

    # 构建目标路径（用于数据库记录，不实际保存文件）
    file_path = os.path.join(REF_AUDIO_DIR, filename)
    
    temp_file_path = None
    
    try:
        with get_fingerprinter() as fp:
            connector = fp._get_connector()
            hp = fp.hp
            
            # 检查数据库中是否已存在同名参考音频
            existing_id = connector.find_music_by_music_path(file_path)
            if existing_id is not None:
                log_operation(
                    "UPLOAD_DUPLICATE",
                    f"数据库中已存在同名参考音频: {filename}, ID={existing_id}",
                    "WARNING"
                )
                raise HTTPException(
                    status_code=409,
                    detail=f"参考音频 '{display_name}' 已存在于数据库中，请勿重复添加"
                )
            
            # 保存到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_file_path = temp_file.name
            
            log_operation(
                "UPLOAD_TEMP_SAVED",
                f"文件已保存到临时位置: {temp_file_path}"
            )

            # 检查音频时长
            try:
                duration = librosa.get_duration(path=temp_file_path)
                log_operation(
                    "UPLOAD_DURATION_CHECK",
                    f"音频时长: {duration:.2f}秒"
                )

                if duration > 30:
                    log_operation(
                        "UPLOAD_REJECTED",
                        f"音频时长超过30秒限制: {duration:.2f}秒",
                        "WARNING"
                    )
                    # 删除临时文件
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    raise HTTPException(
                        status_code=400,
                        detail=f"音频时长为 {duration:.1f} 秒，超过30秒限制，请裁剪后重新上传"
                    )
            except HTTPException:
                raise
            except Exception as e:
                log_operation(
                    "UPLOAD_DURATION_ERROR",
                    f"时长检测失败: {str(e)}",
                    "ERROR"
                )
                # 时长检测失败，继续处理
                pass

            # 生成指纹并添加到数据库
            log_operation(
                "UPLOAD_FINGERPRINT",
                f"开始生成指纹: {display_name}"
            )

            # 使用临时文件路径生成指纹
            creator = fp._get_creator()
            creator.create_finger_prints_and_save_database(
                music_path=temp_file_path,
                connector=connector
            )
            
            # 获取音乐ID
            music_id = connector.find_music_by_music_path(temp_file_path)
            
            # 更新音乐名称为用户指定的名称，并更新路径为虚拟路径（不实际保存）
            if music_id:
                sql_update = f"""
                    UPDATE {hp.fingerprint.database.tables.music.name}
                    SET {hp.fingerprint.database.tables.music.column.music_name} = %s,
                        {hp.fingerprint.database.tables.music.column.music_path} = %s
                    WHERE {hp.fingerprint.database.tables.music.column.music_id} = %s
                """
                connector.cursor.execute(sql_update, (display_name, file_path, music_id))
                connector.conn.commit()

            # 删除临时文件
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                log_operation(
                    "UPLOAD_TEMP_CLEANUP",
                    f"临时文件已删除: {temp_file_path}"
                )

            elapsed_time = (time.time() - start_time) * 1000
            log_operation(
                "UPLOAD_SUCCESS",
                f"参考音频添加成功: ID={music_id}, 名称='{display_name}', "
                f"耗时={elapsed_time:.2f}ms"
            )

            return AddReferenceResponse(
                success=True,
                music_id=music_id,
                message=f"参考音频 '{display_name}' 添加成功"
            )

    except HTTPException:
        # 删除临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise
    except Exception as e:
        elapsed_time = (time.time() - start_time) * 1000
        log_operation(
            "UPLOAD_ERROR",
            f"添加失败: {str(e)}, 耗时={elapsed_time:.2f}ms",
            "ERROR"
        )
        # 删除临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            log_operation(
                "UPLOAD_TEMP_CLEANUP",
                f"清理临时文件: {temp_file_path}"
            )
        raise HTTPException(status_code=500, detail=f"添加参考音频失败: {str(e)}")


@router.post("/add-existing")
async def add_existing_reference(
    file_path: str = Form(...),
    name: Optional[str] = Form(None)
):
    """
    将已存在的音频文件路径添加到指纹库（仅添加记录，不处理文件）
    
    - **file_path**: 音频文件的路径（只需要路径信息用于数据库记录，不需要实际文件存在）
    - **name**: 音频名称（可选，默认从文件名提取）
    
    注意：此接口仅添加数据库记录，不会读取或验证文件内容。
    适用于指纹已经通过其他方式添加到数据库的场景。
    """
    import time
    start_time = time.time()
    
    log_operation(
        "ADD_EXISTING_START",
        f"开始添加已有参考音频记录: path={file_path}, name={name or '默认'}"
    )
    
    # 验证文件格式
    allowed_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in allowed_extensions:
        log_operation(
            "ADD_EXISTING_ERROR",
            f"不支持的文件格式: {ext}",
            "ERROR"
        )
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式，仅支持 {allowed_extensions}"
        )
    
    try:
        with get_fingerprinter() as fp:
            connector = fp._get_connector()
            hp = fp.hp
            
            # 确定显示名称
            display_name = name if name else os.path.splitext(os.path.basename(file_path))[0]
            
            # 检查是否已存在
            existing_id = connector.find_music_by_music_path(file_path)
            if existing_id is not None:
                elapsed_time = (time.time() - start_time) * 1000
                log_operation(
                    "ADD_EXISTING_DUPLICATE",
                    f"参考音频已存在: ID={existing_id}, path={file_path}, 耗时={elapsed_time:.2f}ms",
                    "WARNING"
                )
                return AddReferenceResponse(
                    success=True,
                    music_id=existing_id,
                    message=f"参考音频 '{display_name}' 已存在 (ID={existing_id})"
                )
            
            # 直接添加音乐记录到数据库（不验证文件是否存在）
            sql = f"""
                INSERT INTO {hp.fingerprint.database.tables.music.name}
                ({hp.fingerprint.database.tables.music.column.music_name}, 
                 {hp.fingerprint.database.tables.music.column.music_path})
                VALUES (%s, %s)
            """
            connector.cursor.execute(sql, (display_name, file_path))
            connector.conn.commit()
            
            # 获取新添加的音乐ID
            music_id = connector.find_music_by_music_path(file_path)
            
            elapsed_time = (time.time() - start_time) * 1000
            log_operation(
                "ADD_EXISTING_SUCCESS",
                f"参考音频记录添加成功: ID={music_id}, name='{display_name}', path={file_path}, 耗时={elapsed_time:.2f}ms"
            )
            
            return AddReferenceResponse(
                success=True,
                music_id=music_id,
                message=f"参考音频 '{display_name}' 添加成功 (ID={music_id})"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        elapsed_time = (time.time() - start_time) * 1000
        log_operation(
            "ADD_EXISTING_ERROR",
            f"添加失败: {str(e)}, 耗时={elapsed_time:.2f}ms",
            "ERROR"
        )
        raise HTTPException(status_code=500, detail=f"添加参考音频失败: {str(e)}")


@router.delete("/{music_id}", response_model=DeleteReferenceResponse)
async def delete_reference_audio(music_id: int):
    """
    删除参考音频及其指纹（仅删除数据库记录，不删除文件）

    - **music_id**: 音乐ID
    """
    start_time = time.time()
    log_operation(
        "DELETE_START",
        f"开始删除参考音频: ID={music_id}"
    )

    try:
        with get_fingerprinter() as fp:
            # 先获取音频信息用于日志记录
            connector = fp._get_connector()
            hp = fp.hp

            # 查询要删除的音频信息
            sql = f"""
                SELECT {hp.fingerprint.database.tables.music.column.music_name}
                FROM {hp.fingerprint.database.tables.music.name}
                WHERE {hp.fingerprint.database.tables.music.column.music_id} = %s
            """
            connector.cursor.execute(sql, (music_id,))
            result = connector.cursor.fetchone()

            if result:
                music_name = result[0]
                log_operation(
                    "DELETE_INFO",
                    f"找到要删除的音频: ID={music_id}, 名称='{music_name}'"
                )
            else:
                log_operation(
                    "DELETE_WARNING",
                    f"未找到音频记录: ID={music_id}",
                    "WARNING"
                )

            # 仅删除指纹库记录，不操作文件
            success = fp.delete_reference(music_id)

            if success:
                elapsed_time = (time.time() - start_time) * 1000
                log_operation(
                    "DELETE_SUCCESS",
                    f"参考音频删除成功: ID={music_id}, "
                    f"名称='{music_name if result else '未知'}', 耗时={elapsed_time:.2f}ms"
                )
                return DeleteReferenceResponse(
                    success=True,
                    message=f"参考音频 (ID={music_id}) 已删除"
                )
            else:
                log_operation(
                    "DELETE_ERROR",
                    f"删除失败: ID={music_id}",
                    "ERROR"
                )
                raise HTTPException(status_code=400, detail="删除失败")

    except HTTPException:
        raise
    except Exception as e:
        elapsed_time = (time.time() - start_time) * 1000
        log_operation(
            "DELETE_ERROR",
            f"删除异常: ID={music_id}, 错误={str(e)}, 耗时={elapsed_time:.2f}ms",
            "ERROR"
        )
        raise HTTPException(status_code=500, detail=f"删除参考音频失败: {str(e)}")


@router.get("/stats")
async def get_reference_stats():
    """
    获取参考音频库统计信息
    """
    start_time = time.time()
    log_operation("STATS_START", "开始获取参考音频库统计信息")

    try:
        with get_fingerprinter() as fp:
            references = fp.get_all_references()

            total_hashes = sum(ref["hash_count"] for ref in references)
            total_references = len(references)
            average_hashes = total_hashes / total_references if references else 0

            elapsed_time = (time.time() - start_time) * 1000
            log_operation(
                "STATS_SUCCESS",
                f"统计信息: 总数={total_references}, 总哈希={total_hashes}, "
                f"平均哈希={average_hashes:.2f}, 耗时={elapsed_time:.2f}ms"
            )

            return {
                "total_references": total_references,
                "total_hashes": total_hashes,
                "average_hashes": average_hashes
            }
    except Exception as e:
        elapsed_time = (time.time() - start_time) * 1000
        log_operation(
            "STATS_ERROR",
            f"获取统计信息失败: {str(e)}, 耗时={elapsed_time:.2f}ms",
            "ERROR"
        )
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")
