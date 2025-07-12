import os
import sys
import time
import tempfile
import base64
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import json
import torch
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
except ImportError:
    # 兼容旧版本的transformers
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from transformers import DetrImageProcessor, DetrForObjectDetection
    except ImportError:
        # 如果还是无法导入，使用替代方案
        BlipProcessor = None
        BlipForConditionalGeneration = None
        DetrImageProcessor = None
        DetrForObjectDetection = None
        print("警告: 无法导入transformers模型，将使用替代方案")

# 导入自定义工具（可选）
try:
    from tools import ImageCaptionTool, ObjectDetectionTool
    CUSTOM_TOOLS_AVAILABLE = True
    print("自定义工具加载成功")
except ImportError as e:
    print(f"自定义工具加载失败: {e}")
    print("将使用文心一言进行图像分析")
    CUSTOM_TOOLS_AVAILABLE = False

    # 创建占位符类
    class ImageCaptionTool:
        def _run(self, image_path):
            return "自定义工具不可用"

    class ObjectDetectionTool:
        def _run(self, image_path):
            return "自定义工具不可用"

# 导入向量数据库相关库
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.schema import Document

# 多模态检索相关导入
try:
    from paddlenlp.embeddings import PaddleEmbeddings
    from paddlenlp.embeddings import PaddleImageEmbeddings
    PADDLE_AVAILABLE = True
    print("PaddleNLP嵌入模块加载成功")
except ImportError as e:
    print(f"PaddleNLP嵌入模块加载失败: {e}")
    print("将使用HuggingFace嵌入作为替代")
    PADDLE_AVAILABLE = False

    # 创建替代的嵌入类
    class PaddleEmbeddings(Embeddings):
        def __init__(self):
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings()

        def embed_documents(self, texts):
            return self.embeddings.embed_documents(texts)

        def embed_query(self, text):
            return self.embeddings.embed_query(text)

    class PaddleImageEmbeddings(Embeddings):
        def __init__(self):
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings()

        def embed_documents(self, image_paths):
            # 简单的图像路径嵌入（使用路径字符串）
            return self.embeddings.embed_documents([str(path) for path in image_paths])

        def embed_query(self, text):
            return self.embeddings.embed_query(text)

# 设置页面标题和配置
st.set_page_config(page_title="视频内容理解与RAG系统", layout="wide")
st.title("视频内容理解与RAG系统")

# 简化CSS样式
st.markdown("""
<style>
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    
    .query-box {
        background-color: #F3E5F5;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 默认API密钥
DEFAULT_API_KEY = "1bc3aca311f155f00ad7a33d2eb5b86c472e558b"

# 从OpenAI导入API客户端
try:
    from openai import OpenAI
except ImportError:
    st.error("请先安装OpenAI客户端：pip install openai")
    st.stop()

# 创建数据目录
datapath = Path("./data/visual").resolve()
datapath.mkdir(parents=True, exist_ok=True)

# 创建目录来存储帧和缓存
frame_dir = str(datapath / "frames_from_clips")
cache_dir = str(datapath / "cache")
os.makedirs(frame_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# 文本嵌入类，用于RAG功能
class SimpleEmbeddings(Embeddings):
    """使用FixedDimensionImageEmbeddings作为文本嵌入类"""
    
    def __init__(self):
        """初始化嵌入模型"""
        from tools import FixedDimensionImageEmbeddings
        self.embedder = FixedDimensionImageEmbeddings(dimension=768)
    
    def embed_documents(self, texts: list) -> list:
        """将文本列表转换为嵌入向量"""
        try:
            # 使用tools.py中的嵌入模型
            return self.embedder.embed_documents(texts)
        except Exception as e:
            print(f"嵌入文档时出错: {str(e)}")
            # 降级方案，返回简单的哈希向量
            return [[hash(text) % 1000 / 1000 for _ in range(768)] for text in texts]
    
    def embed_query(self, text: str) -> list:
        """将查询文本转换为嵌入向量"""
        try:
            return self.embedder.embed_query(text)
        except Exception as e:
            print(f"嵌入查询时出错: {str(e)}")
            # 降级方案
            return self.embed_documents([text])[0]

# 核心分析功能封装
class VideoAnalyzer:
    """视频内容理解系统类"""
    
    def __init__(self, api_key=None):
        """初始化视频内容理解系统"""
        self.api_key = api_key or os.environ.get("AI_STUDIO_API_KEY") or DEFAULT_API_KEY
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://aistudio.baidu.com/llm/lmapi/v3"
        )
        self.frame_results = []  # 存储每个关键帧的分析结果
        self.event_summary = ""  # 存储事件总结
        self.vector_store = None  # 存储向量数据库
        self.video_hash = ""  # 存储视频文件的哈希值

        # 多模态检索相关
        self.text_db = None  # 文本向量数据库
        self.image_db = None  # 图像向量数据库
        self.video_metadata_list = []  # 视频元数据列表
        self.frame_metadata_list = []  # 帧元数据列表
        self.uris = []  # 存储帧路径
        self.text_content = []  # 存储文本内容
        
        # 创建固定的帧存储目录
        self.temp_dir = os.path.join("data", "visual", "frames_from_clips")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 初始化图像分析工具（如果可用）
        if CUSTOM_TOOLS_AVAILABLE:
            self.caption_tool = ImageCaptionTool()
            self.detection_tool = ObjectDetectionTool()
        else:
            self.caption_tool = None
            self.detection_tool = None
    
    def encode_image(self, image_path):
        """将图片文件编码为base64格式"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            st.error(f"图片编码错误: {str(e)}")
            return None

    def initialize_multimodal_stores(self):
        """初始化多模态向量存储"""
        print("初始化多模态向量存储...")

        try:
            # 初始化文本向量存储
            text_embedder = PaddleEmbeddings()
            self.text_db = FAISS.from_texts(
                texts=["示例文本"],
                embedding=text_embedder,
                metadatas=[{"source": "初始化"}]
            )

            # 创建一个空白图像并确保保存成功
            sample_image_path = os.path.join(self.temp_dir, "sample.jpg")
            blank_image = np.zeros((224, 224, 3), dtype=np.uint8)
            success = cv2.imwrite(sample_image_path, blank_image)
            print(f"创建样本图像{'成功' if success else '失败'}: {sample_image_path}")

            # 初始化图像向量存储
            image_embedder = PaddleImageEmbeddings()
            self.image_db = FAISS.from_texts(
                texts=[sample_image_path],
                embedding=image_embedder,
                metadatas=[{"source": "初始化", "path": sample_image_path}]
            )

            print("多模态向量存储初始化成功")
            return True

        except Exception as e:
            print(f"初始化多模态向量存储失败: {str(e)}")
            return False

    def process_video_frames_multimodal(self, video_path, video_name=None, number_of_frames_per_second=1):
        """处理视频并提取帧用于多模态检索"""
        print("处理视频帧用于多模态检索...")

        if not video_name:
            video_name = os.path.basename(video_path)

        # 清空之前的数据
        self.text_content = []
        self.video_metadata_list = []
        self.uris = []
        self.frame_metadata_list = []

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"无法打开视频: {video_path}")
                return False

            # 获取视频元数据
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            # 添加视频描述到文本内容
            video_description = f"视频 {video_name} 的内容分析"
            self.text_content.append(video_description)

            metadata = {
                "video": video_name,
                "fps": fps,
                "total_frames": total_frames
            }
            self.video_metadata_list.append(metadata)

            # 提取帧
            mod = int(fps // number_of_frames_per_second) if fps > 0 else 1
            if mod == 0:
                mod = 1

            frame_count = 0
            extracted_frames = 0

            while cap.isOpened() and extracted_frames < 20:  # 限制最多20帧
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % mod == 0:
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # 转换为秒
                    frame_path = os.path.join(self.temp_dir, f"{self.video_hash}_multimodal_frame_{frame_count}.jpg")

                    # 保存帧
                    success = cv2.imwrite(frame_path, frame)
                    if success:
                        frame_metadata = {
                            "timestamp": timestamp,
                            "frame_path": frame_path,
                            "video": video_name,
                            "frame_num": frame_count,
                        }
                        self.uris.append(frame_path)
                        self.frame_metadata_list.append(frame_metadata)
                        extracted_frames += 1

            cap.release()
            print(f"成功提取 {len(self.uris)} 个帧用于多模态检索")
            return True

        except Exception as e:
            print(f"处理视频帧时出错: {str(e)}")
            return False

    def build_multimodal_index(self):
        """构建多模态向量索引"""
        print("构建多模态向量索引...")

        try:
            # 构建文本索引
            if self.text_content and self.video_metadata_list:
                documents = [
                    Document(page_content=text, metadata=metadata)
                    for text, metadata in zip(self.text_content, self.video_metadata_list)
                ]
                self.text_db = FAISS.from_documents(documents, PaddleEmbeddings())
                print(f"成功建立 {len(self.text_content)} 个文本向量")
            else:
                print("警告: 没有文本内容用于构建索引")

            # 构建图像索引
            if self.uris and self.frame_metadata_list:
                # 验证图像路径存在
                valid_uris = []
                valid_metadata = []
                for uri, metadata in zip(self.uris, self.frame_metadata_list):
                    if os.path.exists(uri):
                        valid_uris.append(uri)
                        valid_metadata.append(metadata)

                if valid_uris:
                    documents = [
                        Document(page_content=uri, metadata=metadata)
                        for uri, metadata in zip(valid_uris, valid_metadata)
                    ]
                    self.image_db = FAISS.from_documents(documents, PaddleImageEmbeddings())
                    print(f"成功建立 {len(valid_uris)} 个图像向量")
                else:
                    print("警告: 没有有效的图像路径")
            else:
                print("警告: 没有图像内容用于构建索引")

            return True

        except Exception as e:
            print(f"构建多模态向量索引时出错: {str(e)}")
            return False

    def multimodal_query(self, query, top_k=3):
        """多模态查询，结合文本和图像检索"""
        print(f"执行多模态查询: {query}")

        try:
            # 优先使用文本检索
            text_results = []
            if self.text_db:
                try:
                    text_results = self.text_db.similarity_search(query, k=top_k)
                    print(f"文本检索返回 {len(text_results)} 个结果")
                except Exception as e:
                    print(f"文本检索失败: {str(e)}")

            # 如果文本检索有结果，优先返回文本结果
            if text_results:
                print("使用文本检索结果")
                return self._process_text_results(text_results, query)

            # 如果文本检索失败，使用图像检索
            image_results = []
            if self.image_db:
                try:
                    image_results = self.image_db.similarity_search(query, k=top_k)
                    print(f"图像检索返回 {len(image_results)} 个结果")
                except Exception as e:
                    print(f"图像检索失败: {str(e)}")

            if image_results:
                print("使用图像检索结果")
                return self._process_image_results(image_results, query)

            return "未找到相关内容", None, []

        except Exception as e:
            print(f"多模态查询时出错: {str(e)}")
            return f"查询出错: {str(e)}", None, []

    def _process_text_results(self, results, query):
        """处理文本检索结果"""
        try:
            # 分析结果找最佳视频
            video_scores = {}
            for result in results:
                if "video" in result.metadata:
                    video_name = result.metadata["video"]
                    if video_name not in video_scores:
                        video_scores[video_name] = 0
                    video_scores[video_name] += 1

            if not video_scores:
                return "无法确定相关视频", None, []

            # 获取最高分的视频
            best_video = max(video_scores.items(), key=lambda x: x[1])[0]

            # 构建场景描述
            scene_descriptions = []
            for result in results:
                content = result.page_content
                metadata = result.metadata
                video_name = metadata.get("video", "未知视频")
                scene_descriptions.append(f"视频 {video_name}: {content}")

            scene_des = "\n".join(scene_descriptions)

            # 调用AI生成回答
            ai_response = self._generate_ai_response(query, scene_des)

            # 转换结果格式
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result.page_content,
                    "metadata": result.metadata
                })

            return ai_response, best_video, formatted_results

        except Exception as e:
            print(f"处理文本结果时出错: {str(e)}")
            return f"处理结果出错: {str(e)}", None, []

    def _process_image_results(self, results, query):
        """处理图像检索结果"""
        try:
            # 分析结果找最佳视频
            video_scores = {}
            for result in results:
                if "video" in result.metadata:
                    video_name = result.metadata["video"]
                    if video_name not in video_scores:
                        video_scores[video_name] = 0
                    video_scores[video_name] += 1

            if not video_scores:
                return "无法确定相关视频", None, []

            # 获取最高分的视频
            best_video = max(video_scores.items(), key=lambda x: x[1])[0]

            # 构建场景描述（基于图像路径和元数据）
            scene_descriptions = []
            for result in results:
                metadata = result.metadata
                video_name = metadata.get("video", "未知视频")
                timestamp = metadata.get("timestamp", 0)
                frame_path = result.page_content

                # 尝试从已有的分析结果中获取描述
                frame_description = self._get_frame_description(frame_path)
                if frame_description:
                    scene_descriptions.append(f"时间点 {timestamp:.1f}s: {frame_description}")
                else:
                    scene_descriptions.append(f"时间点 {timestamp:.1f}s: 视频帧 {frame_path}")

            scene_des = "\n".join(scene_descriptions)

            # 调用AI生成回答
            ai_response = self._generate_ai_response(query, scene_des)

            # 转换结果格式
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result.page_content,
                    "metadata": result.metadata
                })

            return ai_response, best_video, formatted_results

        except Exception as e:
            print(f"处理图像结果时出错: {str(e)}")
            return f"处理结果出错: {str(e)}", None, []

    def _get_frame_description(self, frame_path):
        """从已有的分析结果中获取帧描述"""
        for result in self.frame_results:
            if result.get("frame_path") == frame_path:
                return result.get("analysis", "")
        return ""

    def _generate_ai_response(self, query, scene_des):
        """生成AI回答"""
        try:
            completion = self.client.chat.completions.create(
                model="ernie-4.5-turbo-32k",
                messages=[
                    {
                        "role": "system",
                        "content": "你是百度研发的知识增强大语言模型文心一言，专门用于视频内容分析和问答。"
                    },
                    {
                        "role": "user",
                        "content": f"""基于以下视频场景描述回答问题：

场景描述：
{scene_des}

问题：{query}

请按照以下格式详细回答：

1. 首先确定最相关的视频片段或时间点
2. 如果问题涉及寻找特定的人物，请：
   - 明确列出找到的人物数量
   - 详细描述每个人的外观特征（衣着、位置、动作等）
   - 用Shopper 1、Shopper 2等方式区分不同的人物
   - 说明每个人正在做什么具体的行为

3. 如果问题涉及其他内容，请基于场景描述提供详细、具体的回答

请确保回答具体、详细，避免模糊的描述。"""
                    }
                ],
                temperature=0.7
            )

            return completion.choices[0].message.content

        except Exception as e:
            print(f"生成AI回答时出错: {str(e)}")
            return f"基于场景描述，找到相关内容但AI分析不可用。场景描述：{scene_des[:200]}..."
    
    def calculate_video_hash(self, video_path):
        """计算视频文件的哈希值作为唯一标识符"""
        try:
            with open(video_path, "rb") as f:
                # 读取前10MB的内容计算哈希值，避免大文件读取过慢
                content = f.read(10 * 1024 * 1024)
                import hashlib
                self.video_hash = hashlib.md5(content).hexdigest()
                return self.video_hash
        except Exception as e:
            st.error(f"计算视频哈希值时出错: {str(e)}")
            self.video_hash = f"error_{int(time.time())}"
            return self.video_hash
    
    def get_cache_path(self):
        """获取缓存文件路径"""
        if not self.video_hash:
            return None
        return os.path.join(cache_dir, f"{self.video_hash}.json")
    
    def save_to_cache(self):
        """将分析结果保存到缓存"""
        if not self.video_hash or not self.frame_results:
            return False
        
        try:
            # 准备缓存数据
            cache_data = {
                "video_hash": self.video_hash,
                "frame_results": [],
                "event_summary": self.event_summary,
                "timestamp": time.time()
            }
            
            # 处理帧结果，确保图像文件存在
            for result in self.frame_results:
                # 直接使用原始路径，因为现在使用固定目录
                frame_path = result["frame_path"]

                # 确保文件存在
                if os.path.exists(frame_path):
                    # 直接保存原始路径
                    cache_data["frame_results"].append(result.copy())
                else:
                    print(f"警告: 帧文件不存在: {frame_path}")
                    # 仍然保存结果，但标记文件缺失
                    result_copy = result.copy()
                    result_copy["file_missing"] = True
                    cache_data["frame_results"].append(result_copy)
            
            # 保存到文件
            with open(self.get_cache_path(), "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            st.error(f"保存缓存时出错: {str(e)}")
            return False
    
    def load_from_cache(self):
        """从缓存加载分析结果"""
        if not self.video_hash:
            return False
        
        cache_path = self.get_cache_path()
        if not cache_path or not os.path.exists(cache_path):
            return False
        
        try:
            # 从文件加载缓存
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            # 验证缓存数据
            if cache_data["video_hash"] != self.video_hash:
                return False
            
            # 加载数据
            self.frame_results = cache_data["frame_results"]
            self.event_summary = cache_data["event_summary"]

            # 注意：视频数据不从缓存恢复，需要用户重新上传

            # 验证所有图像文件是否存在
            missing_files = []
            for result in self.frame_results:
                if not os.path.exists(result["frame_path"]):
                    missing_files.append(result["frame_path"])

            if missing_files:
                print(f"警告: 以下帧文件缺失: {missing_files[:3]}...")  # 只显示前3个
                # 不返回False，继续使用缓存，但标记文件缺失
                for result in self.frame_results:
                    if not os.path.exists(result["frame_path"]):
                        result["file_missing"] = True

            # 重建向量存储
            self.build_vector_store()
            
            return True
        except Exception as e:
            st.error(f"加载缓存时出错: {str(e)}")
            return False
    
    def extract_frames(self, video_path, interval=5, scene_change_threshold=30):
        """从视频中提取关键帧"""
        if not os.path.exists(video_path):
            st.error(f"错误: 视频文件'{video_path}'不存在")
            return [], []
        
        # 计算视频哈希值
        self.calculate_video_hash(video_path)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("错误: 无法打开视频文件")
            return [], []
        
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # 存储关键帧和时间戳
        frame_paths = []
        timestamps = []
        
        # 上一帧的时间和图像
        last_frame_time = -interval  # 确保第一帧被保存
        last_frame = None
        
        frame_count = 0
        
        # 更新进度
        progress_text = st.empty()
        progress_text.text("正在提取视频关键帧...")
        progress_bar = st.progress(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            frame_count += 1
            
            # 每50帧更新进度
            if frame_count % 50 == 0:
                progress = (frame_count / total_frames)
                progress_bar.progress(progress)
            
            # 检查是否与上一帧相差了足够的时间
            if current_time - last_frame_time < interval:
                continue
            
            # 检查是否有场景变化
            if last_frame is not None:
                # 转换为灰度图像
                gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_last = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
                
                # 计算两帧之间的差异
                frame_diff = cv2.absdiff(gray_current, gray_last)
                
                # 平均差异值越大，变化越显著
                mean_diff = np.mean(frame_diff)
                
                # 如果变化不够明显，并且不是强制关键帧，则跳过
                if mean_diff < scene_change_threshold and current_time - last_frame_time < interval * 2:
                    continue
            
            # 保存关键帧，使用视频哈希值作为前缀
            frame_filename = os.path.join(self.temp_dir, f"{self.video_hash}_frame_{int(current_time)}.jpg")
            cv2.imwrite(frame_filename, frame)
            
            frame_paths.append(frame_filename)
            timestamps.append(current_time)
            
            last_frame_time = current_time
            last_frame = frame.copy()
        
        cap.release()
        progress_bar.progress(1.0)
        progress_text.text(f"成功提取 {len(frame_paths)} 个关键帧")
        
        return frame_paths, timestamps
    
    def analyze_frame(self, frame_path, timestamp, prompt=None, use_tools=True):
        """分析单个关键帧，可选择使用自定义工具或文心一言"""
        if not prompt:
            prompt = """请详细分析这张视频画面，特别关注以下方面：

1. **人物描述**：
   - 详细描述每个人的外观特征（衣着、发型、配饰等）
   - 描述每个人的位置和姿态
   - 说明每个人正在做什么动作
   - 如果有多个人，请用Shopper 1、Shopper 2等方式区分

2. **场景环境**：
   - 描述场所类型和环境特征
   - 描述货架、商品、装饰等物品
   - 注意空间布局和位置关系

3. **行为分析**：
   - 分析人物的具体行为和动作
   - 识别是否有购物、挑选、查看等行为
   - 注意人物之间的互动

4. **重要细节**：
   - 注意任何特殊的物品或行为
   - 识别可能的重要事件或活动

请用清晰、具体的语言描述，避免模糊的表述。"""
        
        # 格式化时间
        time_str = str(timedelta(seconds=int(timestamp)))
        
        try:
            # 使用自定义工具进行分析（如果可用且被选择）
            if use_tools and CUSTOM_TOOLS_AVAILABLE and self.caption_tool and self.detection_tool:
                # 使用图像描述工具
                caption = self.caption_tool._run(frame_path)
                
                # 使用对象检测工具
                detections = self.detection_tool._run(frame_path)
                
                # 组合结果
                result = f"图像描述: {caption}\n\n检测到的对象:\n{detections}"
            else:
                # 使用文心一言进行分析
                base64_image = self.encode_image(frame_path)
                if not base64_image:
                    return None
                
                # 创建带图片的对话请求
                completion = self.client.chat.completions.create(
                    model="ernie-4.5-turbo-vl-32k",
                    messages=[
                        {
                            "role": "system",
                            "content": "你是视频内容理解专家，请详细分析视频中的关键画面。识别画面中的活动、人物、物体及场景情况。"
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.3
                )
                
                # 获取分析结果
                result = completion.choices[0].message.content
            
            # 构建结果
            frame_result = {
                "timestamp": timestamp,
                "time_str": time_str,
                "frame_path": frame_path,
                "analysis": result
            }
            
            return frame_result
            
        except Exception as e:
            st.error(f"分析过程中发生错误: {str(e)}")
            return None
    
    def build_vector_store(self):
        """构建向量存储，用于RAG功能"""
        if not self.frame_results:
            st.warning("没有关键帧分析结果，无法构建向量存储")
            return None
        
        # 准备文档
        documents = []
        for result in self.frame_results:
            # 创建文档，包含分析结果和时间信息
            doc = Document(
                page_content=result["analysis"],
                metadata={
                    "timestamp": result["timestamp"],
                    "time_str": result["time_str"],
                    "frame_path": result["frame_path"]
                }
            )
            documents.append(doc)
        
        # 创建向量存储
        embeddings = SimpleEmbeddings()
        vector_store = FAISS.from_documents(documents, embeddings)
        
        self.vector_store = vector_store
        return vector_store
    
    def query_video(self, query, top_k=3):
        """使用RAG查询视频内容"""
        try:
            if not self.vector_store:
                st.warning("请先构建向量存储")
                print("错误: 向量存储未初始化")
                return []
            
            print(f"开始查询: {query}")
            print(f"向量存储状态: {self.vector_store is not None}")
            
            # 直接使用基本的相似性搜索
            print("执行相似性搜索...")
            try:
                docs = self.vector_store.similarity_search(query, k=top_k)
                print(f"搜索到 {len(docs)} 个文档")
            except Exception as e:
                print(f"相似性搜索出错: {str(e)}")
                return []
            
            # 准备结果
            results = []
            for i, doc in enumerate(docs):
                try:
                    print(f"处理文档 {i+1}: {doc.page_content[:100]}...")
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
                except Exception as e:
                    print(f"处理文档 {i+1} 时出错: {str(e)}")
                    continue
            
            print(f"查询完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            print(f"查询视频内容时出错: {str(e)}")
            st.error(f"查询视频内容时出错: {str(e)}")
            return []

    def intelligent_video_query(self, query, top_k=3):
        """智能视频查询，支持多模态检索"""
        try:
            print(f"查询: {query}")

            # 优先使用多模态查询（如果可用）
            if self.text_db or self.image_db:
                print("使用多模态查询")
                return self.multimodal_query(query, top_k)

            # 回退到基础向量搜索
            print("使用基础向量搜索")
            if not self.vector_store:
                return "请先分析视频以构建向量存储", None, []

            # 执行向量搜索
            results = self.vector_store.similarity_search(query, k=top_k)

            if not results:
                return "未找到相关视频内容", None, []

            print(f"找到 {len(results)} 个相关结果")

            # 构建场景描述，类似main.ipynb中的scene_des
            scene_descriptions = []
            for result in results:
                content = result.page_content
                metadata = result.metadata
                time_str = metadata.get("time_str", "未知时间")
                scene_descriptions.append(f"时间点 {time_str}: {content}")

            scene_des = "\n".join(scene_descriptions)

            # 调用文心一言API，类似main.ipynb中的call_ernie_api_fixed
            try:
                completion = self.client.chat.completions.create(
                    model="ernie-4.5-turbo-32k",
                    messages=[
                        {
                            "role": "system",
                            "content": "你是百度研发的知识增强大语言模型文心一言，专门用于视频内容分析和问答。"
                        },
                        {
                            "role": "user",
                            "content": f"""基于以下视频场景描述回答问题：

场景描述：
{scene_des}

问题：{query}

请按照以下格式详细回答：

1. 首先确定最相关的视频片段或时间点
2. 如果问题涉及寻找特定的人物，请：
   - 明确列出找到的人物数量
   - 详细描述每个人的外观特征（衣着、位置、动作等）
   - 用Shopper 1、Shopper 2等方式区分不同的人物
   - 说明每个人正在做什么具体的行为

3. 如果问题涉及其他内容，请基于场景描述提供详细、具体的回答

请确保回答具体、详细，避免模糊的描述。"""
                        }
                    ],
                    temperature=0.7
                )

                ai_response = completion.choices[0].message.content

                # 转换结果格式以便显示
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "content": result.page_content,
                        "metadata": result.metadata
                    })

                return ai_response, "当前分析的视频", formatted_results

            except Exception as e:
                print(f"调用文心一言API时出错: {str(e)}")
                # 如果API调用失败，返回基础查询结果
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "content": result.page_content,
                        "metadata": result.metadata
                    })

                basic_response = f"找到 {len(results)} 个相关内容片段，但AI分析不可用"
                return basic_response, "当前分析的视频", formatted_results

        except Exception as e:
            print(f"智能查询过程中发生错误: {str(e)}")
            return f"查询过程中出现错误: {str(e)}", None, []

    def summarize_events(self, custom_prompt=None):
        """根据所有关键帧分析结果，生成事件总结"""
        if not self.frame_results:
            st.warning("没有关键帧分析结果，无法生成事件总结")
            return "无法生成总结：缺少关键帧分析结果"
        
        # 准备输入数据
        frame_analyses = []
        for result in self.frame_results:
            frame_analyses.append({
                "时间": result["time_str"],
                "分析结果": result["analysis"]
            })
        
        # 转为JSON字符串
        analyses_json = json.dumps(frame_analyses, ensure_ascii=False)
        
        # 保存分析结果
        with open(os.path.join(self.temp_dir, "frame_analyses.json"), "w", encoding="utf-8") as f:
            f.write(analyses_json)
        
        # 设置提示词
        prompt = custom_prompt or "基于这些视频关键帧的分析结果，总结视频中发生的主要事件，按照时间顺序列出。为视频内容提供一个完整的概述。"
        
        try:
            # 创建总结请求
            completion = self.client.chat.completions.create(
                model="ernie-4.5-turbo-128k-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "你是视频内容理解专家，请基于关键帧分析结果对整段视频内容进行总结。重点关注事件发生的时间线和内容连贯性。"
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\n关键帧分析结果：{analyses_json}"
                    }
                ],
                temperature=0.3
            )
            
            # 获取总结结果
            summary = completion.choices[0].message.content
            self.event_summary = summary
            
            return summary
            
        except Exception as e:
            st.error(f"总结生成过程中发生错误: {str(e)}")
            return "生成总结时发生错误"

# 简介
st.markdown("""
<div class='info-box'>
    此应用使用大模型对视频内容进行智能分析，通过关键帧提取与RAG技术实现对视频完整内容的理解和查询。
    <br><br>
    <b>功能特点：</b>
    <ul>
        <li>自动提取视频关键帧</li>
        <li>智能分析画面内容</li>
        <li>时序建模与整合</li>
        <li>生成事件时间线</li>
        <li>基于RAG的视频内容查询</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# 创建侧边栏参数设置
st.sidebar.header("参数设置")

# API密钥输入
api_key = st.sidebar.text_input(
    "API密钥（选填）",
    value="",
    help="输入文心一言API密钥，留空则使用默认密钥"
)

# 视频处理参数
st.sidebar.subheader("视频处理参数")

interval = st.sidebar.slider(
    "关键帧提取间隔（秒）",
    min_value=1,
    max_value=30,
    value=5,
    help="两个关键帧之间的最小时间间隔"
)

threshold = st.sidebar.slider(
    "场景变化检测阈值",
    min_value=5,
    max_value=100,
    value=30,
    help="值越高，需要更明显的场景变化才会提取关键帧"
)

# 分析工具选择
use_custom_tools = st.sidebar.checkbox(
    "使用自定义分析工具", 
    value=False,
    help="选择是否使用自定义图像分析工具而非文心一言API"
)

# 分析提示语
st.sidebar.subheader("分析提示语")

frame_prompt = st.sidebar.text_area(
    "关键帧分析提示",
    value="请详细分析这张视频画面，描述画面中的主要对象、场景和动作。注意识别任何重要事件或活动。",
    help="指导模型如何分析每个关键帧的提示语"
)

summary_prompt = st.sidebar.text_area(
    "内容总结提示",
    value="基于这些视频关键帧的分析结果，总结视频中发生的主要事件，按照时间顺序列出。为视频内容提供一个完整的概述。",
    help="指导模型如何生成整体内容总结的提示语"
)

# 关于部分
st.sidebar.markdown("---")
st.sidebar.info(
    "视频内容理解与RAG系统 v1.0\n\n"
    "基于大模型技术\n\n"
    "© 2023 版权所有"
)

# 主界面
st.header("上传视频文件")

# 上传视频文件
uploaded_file = st.file_uploader("选择视频文件", type=["mp4", "avi", "mov", "mkv", "wmv"])

# 如果上传了视频文件
if uploaded_file is not None:
    # 显示上传的视频信息
    st.subheader("视频信息")
    st.write(f"文件名: {uploaded_file.name}")
    st.write(f"文件大小: {uploaded_file.size / (1024 * 1024):.2f} MB")
    st.write(f"文件类型: {uploaded_file.type}")
    
    # 分析视频按钮
    analyze_btn = st.button("开始分析视频")
    
    if analyze_btn:
        # 保存上传的视频到临时文件（仅用于分析）
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_video_path = tmp_file.name

        # 保存视频数据到session_state以便后续播放
        st.session_state.uploaded_video_data = uploaded_file.getvalue()
        st.session_state.uploaded_video_name = uploaded_file.name
        
        try:
            # 设置API密钥
            if not api_key:
                api_key = DEFAULT_API_KEY
                
            # 开始分析
            st.markdown("<div class='info-box'>开始分析视频...</div>", unsafe_allow_html=True)
            
            # 创建分析器并提取关键帧
            analyzer = VideoAnalyzer(api_key)
            
            # 保存analyzer到session_state
            st.session_state.analyzer = analyzer
            
            # 计算视频哈希值
            analyzer.calculate_video_hash(temp_video_path)
            
            # 尝试从缓存加载
            if analyzer.load_from_cache():
                st.markdown("<div class='success-box'>✅ 从缓存加载分析结果成功！</div>", unsafe_allow_html=True)
                # 显示缓存的第一帧作为预览
                if analyzer.frame_results:
                    preview_img = Image.open(analyzer.frame_results[0]["frame_path"])
                    st.image(preview_img, caption=f"关键帧预览 (共{len(analyzer.frame_results)}帧)")
            else:
                # 提取关键帧
                frame_paths, timestamps = analyzer.extract_frames(
                    temp_video_path,
                    interval=interval,
                    scene_change_threshold=threshold
                )
                
                if not frame_paths:
                    st.markdown("<div class='warning-box'>❌ 未能提取关键帧，请检查视频文件</div>", unsafe_allow_html=True)
                    st.stop()
                    
                # 显示提取的第一帧作为预览
                if frame_paths:
                    preview_img = Image.open(frame_paths[0])
                    st.image(preview_img, caption=f"关键帧预览 (共{len(frame_paths)}帧)")
                    
                # 分析关键帧
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                for i, (frame_path, timestamp) in enumerate(zip(frame_paths, timestamps)):
                    # 更新进度
                    progress = (i + 1) / len(frame_paths)
                    progress_bar.progress(progress)
                    progress_text.text(f"正在分析第 {i+1}/{len(frame_paths)} 帧...")
                    
                    # 分析当前帧
                    result = analyzer.analyze_frame(
                        frame_path, 
                        timestamp, 
                        prompt=frame_prompt,
                        use_tools=use_custom_tools
                    )
                    if result:
                        analyzer.frame_results.append(result)
                
                progress_text.text("关键帧分析完成，正在生成内容总结...")
                
                # 构建向量存储用于RAG
                analyzer.build_vector_store()

                # 构建多模态检索索引
                progress_text.text("正在构建多模态检索索引...")
                try:
                    # 初始化多模态向量存储
                    if analyzer.initialize_multimodal_stores():
                        # 处理视频帧用于多模态检索
                        if analyzer.process_video_frames_multimodal(temp_video_path, uploaded_file.name):
                            # 构建多模态索引
                            if analyzer.build_multimodal_index():
                                st.markdown("<div class='success-box'>✅ 多模态检索索引构建成功！</div>", unsafe_allow_html=True)
                            else:
                                st.warning("多模态索引构建失败，但基础RAG功能仍可用")
                        else:
                            st.warning("多模态帧处理失败，但基础RAG功能仍可用")
                    else:
                        st.warning("多模态向量存储初始化失败，但基础RAG功能仍可用")
                except Exception as e:
                    print(f"多模态检索构建失败: {str(e)}")
                    st.warning("多模态检索功能不可用，但基础RAG功能仍可用")

                # 生成事件总结
                summary = analyzer.summarize_events(custom_prompt=summary_prompt)
                
                # 保存到缓存
                if analyzer.save_to_cache():
                    st.markdown("<div class='success-box'>✅ 分析结果已缓存，下次分析相同视频将更快！</div>", unsafe_allow_html=True)
                
                # 完成分析
                progress_text.text("✅ 分析完成！")
                st.success("视频分析已完成")
                
            # 保存分析结果到session_state
            st.session_state.analysis_completed = True
            st.session_state.analysis_results = {
                'event_summary': analyzer.event_summary,
                'frame_results': analyzer.frame_results
            }
          
        except Exception as e:
            # 处理错误
            st.error(f"分析过程中发生错误: {str(e)}")
            st.code(str(e))
            
        finally:
            # 清理临时文件
            if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except:
                    pass

else:
    # 显示使用说明
    st.markdown("""
    <div class='warning-box'>
        请上传视频文件以开始分析。支持的格式：MP4, AVI, MOV, MKV, WMV
    </div>
    """, unsafe_allow_html=True)
    
    # 展示使用流程
    st.subheader("使用流程")
    st.markdown("""
    1. 在侧边栏调整分析参数（或使用默认值）
    2. 上传视频文件
    3. 点击"开始分析视频"按钮
    4. 等待系统提取关键帧并进行分析
    5. 查看分析结果和总结
    6. 使用RAG功能查询视频内容
    """)
    
    # 展示视频理解方法
    st.subheader("视频理解方法")
    
    st.markdown("""
    本系统采用"离散采样 + RAG"的方法让大模型理解视频内容：
    
    1. **离散采样**：将视频拆分为关键帧序列
    2. **时序保留**：为每帧保存时间信息
    3. **单帧分析**：分析每帧画面内容
    4. **向量化存储**：将分析结果存入向量数据库
    5. **RAG查询**：基于用户问题检索相关视频内容
    6. **内容总结**：生成视频事件和内容概述
    """)

# 初始化会话状态变量
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'query_results' not in st.session_state:
    st.session_state.query_results = []
if 'analysis_completed' not in st.session_state:
    st.session_state.analysis_completed = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = None
if 'query_video_name' not in st.session_state:
    st.session_state.query_video_name = None
if 'uploaded_video_data' not in st.session_state:
    st.session_state.uploaded_video_data = None
if 'uploaded_video_name' not in st.session_state:
    st.session_state.uploaded_video_name = None

# 显示保存的分析结果（如果存在且不在上传文件的条件块中）
if st.session_state.analysis_completed and st.session_state.analysis_results:
    st.header("分析结果")

    # 显示内容总结
    st.subheader("内容总结")
    st.markdown(f"<div class='info-box'>{st.session_state.analysis_results['event_summary'].replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)

    # 显示关键帧时间线
    st.subheader("关键帧时间线")

    # 使用基础布局展示关键帧和分析结果
    for i, result in enumerate(st.session_state.analysis_results['frame_results']):
        st.markdown(f"### 时间点 {result['time_str']}")
        # 显示关键帧图像
        try:
            frame_path = result['frame_path']
            if os.path.exists(frame_path):
                img = Image.open(frame_path)
                st.image(img, caption=f"关键帧 {i+1}")
            else:
                st.warning(f"图像文件不存在: {os.path.basename(frame_path)}")
                st.info("提示: 可能需要重新分析视频以生成帧文件")
        except Exception as e:
            st.error(f"加载图像时出错: {str(e)}")
        # 显示分析结果
        st.markdown(f"**分析结果**:")
        st.markdown(f"<div class='info-box'>{result['analysis'].replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
        st.markdown("---")

# 添加RAG查询功能 - 放在最后，确保不会影响前面的内容显示
st.header("视频内容查询 (RAG)")
st.markdown("<div class='info-box'>在下方输入问题，系统将基于视频内容为您提供答案</div>", unsafe_allow_html=True)



# 在侧边栏添加查询设置
with st.sidebar:
    st.subheader("查询设置")
    top_k = st.slider(
        "返回结果数量",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="设置查询返回的最相关结果数量"
    )
    st.markdown(f"当前设置: **{top_k}** 个结果")

    # 多模态检索设置
    st.subheader("多模态检索")
    multimodal_enabled = st.checkbox(
        "启用多模态检索",
        value=True,
        help="启用后将同时使用文本和图像进行检索，提高查询准确性"
    )

    if multimodal_enabled:
        st.info("🔍 多模态检索已启用\n\n系统将优先使用文本检索，如果失败则使用图像检索")
    else:
        st.info("📝 仅使用基础文本检索")

    # 缓存管理
    st.subheader("缓存管理")
    cache_dir = os.path.join("data", "visual", "cache")
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
        st.info(f"当前有 {len(cache_files)} 个缓存文件")

        if st.button("清理所有缓存", key="clear_cache"):
            try:
                import shutil
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                    os.makedirs(cache_dir, exist_ok=True)
                st.success("缓存已清理")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"清理缓存失败: {str(e)}")
    else:
        st.info("暂无缓存文件")

    # 测试查询
    st.subheader("快速测试")

    # 测试查询选项
    test_queries = [
        "找三个正在挑选货物的人",
        "视频中有几个人？",
        "他们在做什么？",
        "描述视频中的场景",
        "有什么商品？"
    ]

    selected_query = st.selectbox("选择测试查询", test_queries)

    if st.button("执行测试查询", key="test_query"):
        if 'analyzer' in st.session_state:
            st.session_state.query = selected_query
            print(f"测试查询: {selected_query}")

            # 执行测试查询
            try:
                analyzer = st.session_state.analyzer
                ai_response, video_name, results = analyzer.intelligent_video_query(selected_query, top_k=top_k)

                st.session_state.ai_response = ai_response
                st.session_state.query_video_name = video_name
                st.session_state.query_results = results

                print(f"测试查询完成，结果数量: {len(results) if results else 0}")
                st.success("测试查询已执行，请查看下方结果")

            except Exception as e:
                st.error(f"测试查询失败: {str(e)}")
        else:
            st.warning("请先分析视频")

    # 显示多模态检索状态
    if 'analyzer' in st.session_state:
        analyzer = st.session_state.analyzer
        if hasattr(analyzer, 'text_db') and hasattr(analyzer, 'image_db'):
            text_status = "✅" if analyzer.text_db else "❌"
            image_status = "✅" if analyzer.image_db else "❌"
            st.markdown(f"""
            **多模态检索状态:**
            - 文本索引: {text_status}
            - 图像索引: {image_status}
            """)
        else:
            st.markdown("**检索状态:** 仅基础RAG可用")

# 查询输入框
query = st.text_input("输入您的问题，查询视频内容：",
                        placeholder="例如：视频中有什么人物？",
                        key="query_input",
                        on_change=None)

# 查询按钮
if st.button("提交查询", key="query_button"):
    if query:
        st.session_state.query = query
        print('开始处理查询请求')
        # 显示加载状态
        with st.spinner("正在查询视频内容..."):
            try:
                print(f"开始查询: {query}")

                # 从session_state获取analyzer
                if 'analyzer' in st.session_state:
                    analyzer = st.session_state.analyzer
                    print(f"analyzer对象状态: {analyzer is not None}")
                    print(f"frame_results数量: {len(analyzer.frame_results) if analyzer.frame_results else 0}")
                    print(f"vector_store状态: {analyzer.vector_store is not None}")

                    try:
                        # 执行智能RAG查询
                        print("开始执行智能查询...")
                        ai_response, video_name, results = analyzer.intelligent_video_query(query, top_k=top_k)
                        print(f"智能查询完成，结果数量: {len(results) if results else 0}")

                        # 保存AI回答到session_state
                        st.session_state.ai_response = ai_response
                        st.session_state.query_video_name = video_name

                    except Exception as e:
                        print(f"查询执行出错: {str(e)}")
                        st.error(f"查询执行出错: {str(e)}")
                        results = []
                        st.session_state.ai_response = None
                        st.session_state.query_video_name = None
                else:
                    st.error("未找到分析器，请先分析视频")
                    print("未找到analyzer对象")
                    st.session_state.query_results = []
                    results = []

                print(f"查询返回结果数量: {len(results) if results else 0}")
                st.session_state.query_results = results

                # 如果没有结果，显示提示
                if not results or len(results) == 0:
                    st.warning("未找到相关内容，请尝试其他问题")
                    print("查询未返回任何结果")
                else:
                    print("查询成功，结果已保存到session_state")
            except Exception as e:
                st.error(f"查询过程中发生错误: {str(e)}")
                print(f"查询过程中发生错误: {str(e)}")
                st.session_state.query_results = []
    else:
        st.warning("请输入查询内容")



# 显示AI智能回答
if hasattr(st.session_state, 'ai_response') and st.session_state.ai_response:
    st.header("🤖 完整回复")

    # 显示查询信息
    if hasattr(st.session_state, 'query') and st.session_state.query:
        st.markdown(f"**测试查询:** {st.session_state.query}")

    # 显示AI回答
    st.markdown("**完整回复:**")
    st.markdown(st.session_state.ai_response)

    if hasattr(st.session_state, 'query_video_name') and st.session_state.query_video_name:
        st.markdown(f"**相关视频：** {st.session_state.query_video_name}")

        # 如果有analyzer，尝试显示原始视频
        if 'analyzer' in st.session_state and hasattr(st.session_state, 'uploaded_video_data'):
            try:
                # 显示视频播放器
                st.subheader("📹 相关视频片段")
                video_data = st.session_state.uploaded_video_data
                video_name = getattr(st.session_state, 'uploaded_video_name', 'uploaded_video.mp4')

                if video_data:
                    st.video(video_data)
                    st.caption(f"视频文件: {video_name}")
                else:
                    st.warning("原始视频数据不可用")
            except Exception as e:
                st.warning(f"无法显示视频: {str(e)}")

# 显示查询结果 - 使用容器来避免影响其他内容
if st.session_state.query:
    with st.container():
        st.markdown(f"<div class='query-box'>查询: {st.session_state.query}</div>", unsafe_allow_html=True)

        if st.session_state.query_results and len(st.session_state.query_results) > 0:
            # 直接显示向量搜索结果，不进行额外过滤
            results = st.session_state.query_results

            st.success(f"找到 {len(results)} 个相关内容")

            st.subheader("🔍 详细查询结果")
            st.markdown("以下是与您的查询最相关的视频片段：")

            # 显示查询结果
            for i, result in enumerate(results):
                    try:
                        # 检查结果格式，适配不同的返回格式
                        if isinstance(result, dict) and "metadata" in result:
                            # 标准格式
                            metadata = result["metadata"]
                            content = result.get("content", "")
                            relevance_score = result.get("relevance_score", 0)

                            # 获取对应的帧路径和时间
                            frame_path = metadata.get("frame_path", "")
                            time_str = metadata.get("time_str", "")

                            # 创建两列布局
                            col1, col2 = st.columns([1, 2])

                            with col1:
                                # 显示关键帧
                                if frame_path and os.path.exists(frame_path):
                                    img = Image.open(frame_path)
                                    st.image(img, caption=f"时间点: {time_str}")
                                else:
                                    st.warning(f"图像文件不存在: {frame_path}")

                            with col2:
                                # 显示分析结果
                                st.markdown(f"**相关内容 {i+1}:**")
                                st.markdown(f"**时间点: {time_str}**")
                                st.markdown(f"{content}")

                        # tools.py中可能返回其他格式的结果
                        elif isinstance(result, dict):
                            # 尝试提取关键信息
                            st.markdown(f"**相关内容 {i+1}:**")

                            # 显示所有可能的有用信息
                            for key, value in result.items():
                                if key.lower() not in ["score", "index"]:
                                    st.markdown(f"**{key}**: {value}")

                            # 如果有图像路径，尝试显示
                            if "path" in result and os.path.exists(result["path"]):
                                st.image(result["path"], caption="相关图像")

                        else:
                            # 未知格式，直接显示
                            st.markdown(f"**相关内容 {i+1}:**")
                            st.write(result)

                        st.markdown("---")
                    except Exception as e:
                        st.error(f"显示结果 {i+1} 时出错: {str(e)}")
                        print(f"显示结果 {i+1} 时出错: {str(e)}")
                        continue
        else:
            st.warning("未找到相关内容")
            print("查询返回了空结果")

# 添加页脚
st.markdown("---")
st.markdown("基于大模型技术 | Visual-Vision-RAG项目")