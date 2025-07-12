from langchain.tools import BaseTool
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
from PIL import Image
import torch
import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path
from threading import Thread
from typing import List, Optional, Dict, Any
from zipfile import ZipFile
from openai import OpenAI
from IPython.display import Video

# PaddlePaddle 相关导入
import paddle
from paddlenlp.transformers import ErnieModel, ErnieTokenizer
# 使用ResNet模型替代ViT
from paddle.vision.models import resnet50
from paddle.vision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from paddle.io import Dataset

# 向量数据库
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
# CFG
# 设置路径
datapath = Path("./data/visual").resolve()
datapath.mkdir(parents=True, exist_ok=True)

# 创建目录来存储帧
frame_dir = str(datapath / "frames_from_clips")
os.makedirs(frame_dir, exist_ok=True)

# 每秒提取的帧数
number_of_frames_per_second = 2


class ImageCaptionTool(BaseTool):
    name = "Image captioner"
    description = "Use this tool when given the path to an image that you would like to be described. " \
                  "It will return a simple caption describing the image."

    def _run(self, img_path):
        if BlipProcessor is None or BlipForConditionalGeneration is None:
            return "无法使用图像描述功能：transformers模型未正确加载"
        
        try:
            image = Image.open(img_path).convert('RGB')

            model_name = "Salesforce/blip-image-captioning-large"
            device = "cpu"  # cuda

            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

            inputs = processor(image, return_tensors='pt').to(device)
            output = model.generate(**inputs, max_new_tokens=20)

            caption = processor.decode(output[0], skip_special_tokens=True)

            return caption
        except Exception as e:
            return f"图像描述失败: {str(e)}"

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


class ObjectDetectionTool(BaseTool):
    name = "Object detector"
    description = "Use this tool when given the path to an image that you would like to detect objects. " \
                  "It will return a list of all detected objects. Each element in the list in the format: " \
                  "[x1, y1, x2, y2] class_name confidence_score."

    def _run(self, img_path):
        if DetrImageProcessor is None or DetrForObjectDetection is None:
            return "无法使用物体检测功能：transformers模型未正确加载"
        
        try:
            image = Image.open(img_path).convert('RGB')

            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)

            # convert outputs (bounding boxes and class logits) to COCO API
            # let's only keep detections with score > 0.9
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            detections = ""
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                detections += ' {}'.format(model.config.id2label[int(label)])
                detections += ' {}\n'.format(float(score))

            return detections
        except Exception as e:
            return f"物体检测失败: {str(e)}"

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
class PaddleEmbeddings(Embeddings):
    """使用PaddleNLP的词嵌入模型"""
    
    def __init__(self, model_name="ernie-3.0-medium-zh"):
        """初始化PaddleEmbeddings"""
        super().__init__()
        self.tokenizer = ErnieTokenizer.from_pretrained(model_name)
        self.model = ErnieModel.from_pretrained(model_name)
        self.model.eval()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """生成文本的嵌入向量"""
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pd", padding=True, truncation=True, max_length=512)
            with paddle.no_grad():
                outputs = self.model(**inputs)
            # 使用[CLS]向量作为文档嵌入
            embedding = outputs[1].numpy().tolist()[0]  # 使用pooled output
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """生成查询的嵌入向量"""
        return self.embed_documents([text])[0]
# 图像嵌入类
class PaddleImageEmbeddings(Embeddings):
    """使用PaddleVision的图像嵌入模型"""
    
    def __init__(self):
        """初始化PaddleImageEmbeddings"""
        super().__init__()
        self.model = resnet50(pretrained=True)
        self.model.eval()
    
    def _preprocess_image(self, image_path):
        """预处理图像"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 预处理图像
            img = cv2.resize(img, (256, 256))
            img = img.astype('float32') / 255.0
            img -= np.array([0.485, 0.456, 0.406])
            img /= np.array([0.229, 0.224, 0.225])
            img = img.transpose((2, 0, 1))  # 转换为CHW格式
            img = np.expand_dims(img, axis=0)  # 增加batch维度
            return paddle.to_tensor(img)
        except Exception as e:
            print(f"图像预处理失败: {str(e)}")
            # 返回全零张量作为替代
            return paddle.zeros([1, 3, 224, 224])
    
    def embed_documents(self, image_paths: List[str]) -> List[List[float]]:
        """生成图像的嵌入向量"""
        embeddings = []
        for image_path in image_paths:
            try:
                img = self._preprocess_image(image_path)
                with paddle.no_grad():
                    # 获取特征向量 (使用ResNet的最后一个池化层输出)
                    feat = self.model(img)
                    # 将特征向量转换为一维数组
                    embedding = paddle.squeeze(feat).numpy().tolist()
                embeddings.append(embedding)
            except Exception as e:
                print(f"处理图像时出错 {image_path}: {str(e)}")
                # 返回零向量作为备选
                embeddings.append([0.0] * 1000)  # ResNet50的嵌入维度是1000
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """对查询文本进行嵌入 - 这里需要特殊处理，因为查询是文本"""
        # 这里使用文本嵌入来代替，实际应用可能需要多模态模型
        text_embedder = PaddleEmbeddings()
        return text_embedder.embed_query(text)
def initialize_vector_stores():
    """初始化文本和图像的向量存储"""
    print("初始化向量存储...")
    
    # 初始化文本向量存储
    text_embedder = PaddleEmbeddings()
    text_db = FAISS.from_texts(
        texts=["示例文本"],
        embedding=text_embedder,
        metadatas=[{"source": "初始化"}]
    )
    
    # 创建一个空白图像并确保保存成功
    sample_image_path = os.path.join(frame_dir, "sample.jpg")
    blank_image = np.zeros((224, 224, 3), dtype=np.uint8)
    success = cv2.imwrite(sample_image_path, blank_image)
    print(f"创建样本图像{'成功' if success else '失败'}: {sample_image_path}")
    print(f"文件存在: {os.path.exists(sample_image_path)}")
    image_embedder = PaddleImageEmbeddings()
    image_db = FAISS.from_texts(
        texts=[sample_image_path],  # 使用实际的文件路径
        embedding=image_embedder,
        metadatas=[{"source": "初始化", "path": sample_image_path}]
    )
    
    return text_db, image_db


class ModifiedPaddleImageEmbeddings(PaddleImageEmbeddings):
    def embed_documents(self, image_paths: List[str]) -> List[List[float]]:
        """生成图像的嵌入向量"""
        embeddings = []
        for image_path in image_paths:
            try:
                # 确认文件存在
                if not os.path.exists(image_path):
                    print(f"图像文件不存在: {image_path}")
                    # 提供零向量
                    embeddings.append([0.0] * 1000)
                    continue
                    
                img = self._preprocess_image(image_path)
                with paddle.no_grad():
                    # 获取特征向量
                    feat = self.model(img)
                    # 将特征向量转换为一维数组
                    embedding = paddle.squeeze(feat).numpy().tolist()
                embeddings.append(embedding)
            except Exception as e:
                print(f"处理图像时出错 {image_path}: {str(e)}")
                # 返回零向量作为备选
                embeddings.append([0.0] * 1000)
        return embeddings

# 替换原始的图像嵌入类
image_embedder = ModifiedPaddleImageEmbeddings()

# 修改构建索引函数，更好地处理图像路径
def build_vector_index(text_db, image_db, text_content, video_metadata_list, uris, frame_metadata_list):
    """构建文本和图像的向量索引"""
    print("构建向量索引...")
    
    # 添加文本到向量存储
    if text_content and video_metadata_list:
        try:
            documents = [
                Document(page_content=text, metadata=metadata)
                for text, metadata in zip(text_content, video_metadata_list)
            ]
            text_db = FAISS.from_documents(documents, PaddleEmbeddings())
            print(f"成功建立 {len(text_content)} 个文本向量")
        except Exception as e:
            print(f"建立文本向量时出错: {str(e)}")
    else:
        print("警告: 没有文本内容用于构建索引")
    
    # 检查图像路径并创建新图像
    if not uris or not frame_metadata_list:
        print("警告: 没有图像路径可用")
        # 创建一些简单的测试图像
        test_images = []
        test_metadata = []
        
        # 创建10个测试图像
        for i in range(10):
            img_path = os.path.join(frame_dir, f"test_image_{i}.jpg")
            # 创建随机图像
            test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(img_path, test_img)
            
            if os.path.exists(img_path):
                test_images.append(img_path)
                test_metadata.append({
                    "source": "测试图像",
                    "index": i,
                    "path": img_path
                })
        
        if test_images:
            try:
                documents = [
                    Document(page_content=path, metadata=meta)
                    for path, meta in zip(test_images, test_metadata)
                ]
                image_db = FAISS.from_documents(documents, ModifiedPaddleImageEmbeddings())
                print(f"成功建立 {len(test_images)} 个测试图像向量")
            except Exception as e:
                print(f"建立测试图像向量时出错: {str(e)}")
    else:
        # 使用现有图像路径
        valid_paths = []
        valid_metadata = []
        
        for uri, metadata in zip(uris, frame_metadata_list):
            if os.path.exists(uri):
                valid_paths.append(uri)
                valid_metadata.append(metadata)
                
        if valid_paths:
            try:
                documents = [
                    Document(page_content=path, metadata=meta)
                    for path, meta in zip(valid_paths, valid_metadata)
                ]
                image_db = FAISS.from_documents(documents, ModifiedPaddleImageEmbeddings())
                print(f"成功建立 {len(valid_paths)} 个图像向量")
            except Exception as e:
                print(f"建立图像向量时出错: {str(e)}")
        else:
            print("警告: 所有提供的图像路径都无效")
    
    return text_db, image_db
# 这里提供一段测试数据集
def download_and_extract_data():
    """下载并提取视频数据集"""
    print("下载并提取数据...")
    
    # 使用paddle下载数据
    download_path = os.path.join(str(datapath), "VideoSumForRetailData.zip")
    if not os.path.exists(download_path):
        try:
            # 使用paddle的下载工具
            paddle.utils.download.get_weights_path_from_url(
                "https://huggingface.co/datasets/Intel/Video_Summarization_For_Retail/resolve/main/VideoSumForRetailData.zip",
                os.path.join(str(datapath)),
                decompress=False
            )
        except:
            print(f"自动下载失败，请手动下载数据集并放置在 {download_path}")
            print("数据集地址: https://huggingface.co/datasets/Intel/Video_Summarization_For_Retail")
            return False
    
    # 解压数据
    zip_path = str(datapath / "VideoSumForRetailData.zip")
    if os.path.exists(zip_path):
        with ZipFile(zip_path, "r") as z:
            z.extractall(path=datapath)
    else:
        print(f"找不到数据文件: {zip_path}")
        return False
    
    # 加载视频描述
    anno_path = str(datapath / "VideoSumForRetailData/clips_anno.json")
    if not os.path.exists(anno_path):
        print(f"找不到标注文件: {anno_path}")
        return False
        
    with open(anno_path, "r") as f:
        scene_info = json.load(f)
    
    # 创建视频名称和场景描述的映射
    video_list = {}
    for scene in scene_info:
        video_list[scene["video"].split("/")[-1]] = scene["conversations"][1]["value"]
    
    return video_list

def process_video_frames(video_list):
    """处理视频并提取帧"""
    print("处理视频帧...")
    
    # 确保video_list是字典类型
    if not isinstance(video_list, dict) or not video_list:
        print("错误: video_list不是有效的字典或为空")
        return [], [], [], []
    
    text_content = []
    video_metadata_list = []
    uris = []
    frame_metadata_list = []
    
    video_dir = str(datapath / "VideoSumForRetailData/clips/")
    
    for video_name, description in tqdm(video_list.items()):
        video_path = os.path.join(video_dir, video_name)
        
        # 如果视频不存在，则跳过
        if not os.path.exists(video_path):
            print(f"视频不存在: {video_path}")
            continue
        
        # 获取描述和视频元数据
        text_content.append(description)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        metadata = {"video": video_name, "fps": fps, "total_frames": total_frames}
        video_metadata_list.append(metadata)
        
        # 获取每个提取帧的元数据
        mod = int(fps // number_of_frames_per_second)
        if mod == 0:
            mod = 1
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % mod == 0:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # 将毫秒转换为秒
                frame_path = os.path.join(frame_dir, f"{video_name}_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)  # 将帧保存为图像
                frame_metadata = {
                    "timestamp": timestamp,
                    "frame_path": frame_path,
                    "video": video_name,
                    "frame_num": frame_count,
                }
                uris.append(frame_path)
                frame_metadata_list.append(frame_metadata)
        cap.release()
    
    return text_content, video_metadata_list, uris, frame_metadata_list
def build_vector_index(text_db, image_db, text_content, video_metadata_list, uris, frame_metadata_list):
    """构建文本和图像的向量索引"""
    print("构建向量索引...")
    
    # 检查输入
    if not text_content or not video_metadata_list:
        print("警告: 没有文本内容用于构建索引")
    else:
        # 添加文本到向量存储
        documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(text_content, video_metadata_list)
        ]
        text_db = FAISS.from_documents(documents, PaddleEmbeddings())
        print(f"成功建立 {len(text_content)} 个文本向量")
    
    # 检查提取的帧
    if not uris or not frame_metadata_list:
        print("警告: 没有图像内容用于构建索引")
    else:
        # 验证图像路径存在
        valid_uris = []
        valid_metadata = []
        for uri, metadata in zip(uris, frame_metadata_list):
            if os.path.exists(uri):
                valid_uris.append(uri)
                valid_metadata.append(metadata)
        
        if not valid_uris:
            print("警告: 没有有效的图像路径")
        else:
            # 添加图像到向量存储
            documents = [
                Document(page_content=uri, metadata=metadata)
                for uri, metadata in zip(valid_uris, valid_metadata)
            ]
            image_db = FAISS.from_documents(documents, PaddleImageEmbeddings())
            print(f"成功建立 {len(valid_uris)} 个图像向量")
    
    return text_db, image_db
def multimodal_retrieval(text_db, image_db, query, n_texts=1, n_images=3):
    """执行多模态检索，分开处理文本和图像"""
    print(f"正在查询: {query}")
    print(f"\t检索 {n_texts} 个文本文档和 {n_images} 个图像文档")
    
    # 检索文本
    try:
        text_results = text_db.similarity_search(query, k=n_texts)
        print(f"文本检索成功，找到 {len(text_results)} 个结果")
    except Exception as e:
        print(f"文本检索出错: {str(e)}")
        text_results = []
    
    # 检索图像 - 捕获可能的错误
    try:
        image_results = image_db.similarity_search(query, k=n_images)
        print(f"图像检索成功，找到 {len(image_results)} 个结果")
    except Exception as e:
        print(f"图像检索出错: {str(e)}")
        image_results = []
    
    # 合并结果
    return text_results + image_results

def retrieve_top_results(text_db, image_db, prompt, qcnt=0, print_text_content=False):
    """检索顶部结果"""
    print("正在查询数据库...")
    
    # 只使用文本检索结果
    try:
        text_results = text_db.similarity_search(prompt, k=3)
        print(f"文本检索成功，找到 {len(text_results)} 个结果")
        results = text_results
    except Exception as e:
        print(f"文本检索出错: {str(e)}")
        # 尝试使用图像检索
        try:
            image_results = image_db.similarity_search(prompt, k=3)
            print(f"图像检索成功，找到 {len(image_results)} 个结果")
            results = image_results
        except Exception as e:
            print(f"图像检索也出错: {str(e)}")
            return None, None
    
    print("检索到最匹配的视频!\n")
    
    if print_text_content and results:
        print(f"\t内容:\n\t\t{results[0].page_content}\n")
        print(f"\t元数据:\n\t\t{results[0].metadata}\n")
    
    top_doc = get_top_doc(results, qcnt)
    if top_doc is None:
        return None, None
    
    return top_doc["video"], top_doc

def simple_chatbot(text_db, image_db, video_list, user_query):
    """简单聊天机器人处理用户查询"""
    try:
        video_name, top_doc = retrieve_top_results(text_db, image_db, user_query, print_text_content=True)
        
        if not video_name or video_name not in video_list:
            return "未找到相关视频，或检索过程中出现错误。", None
        
        scene_des = video_list[video_name]
        response = call_ernie_api(user_query, scene_des)
        
        full_response = f"最相关的视频是 **{video_name}** \n\n{response}"
        
        video_dir = str(datapath / "VideoSumForRetailData/clips/")
        video_path = os.path.join(video_dir, video_name)
        
        return full_response, video_path
    except Exception as e:
        print(f"聊天机器人处理出错: {str(e)}")
        return f"处理查询时出现错误: {str(e)}", None

# 创建一个统一维度的图像嵌入类
class FixedDimensionImageEmbeddings(Embeddings):
    """使用固定维度的图像嵌入"""
    
    def __init__(self, dimension=768):
        """初始化为固定维度"""
        self.dimension = dimension
        # 确保使用动态图模式
        paddle.disable_static()
        # 使用ResNet50
        self.model = resnet50(pretrained=True)
        self.model.eval()
    
    def _preprocess_image(self, image_path):
        """预处理图像"""
        try:
            if not os.path.exists(image_path):
                print(f"图像文件不存在: {image_path}")
                return paddle.zeros([1, 3, 224, 224])
                
            img = cv2.imread(image_path)
            if img is None:
                print(f"无法读取图像: {image_path}")
                return paddle.zeros([1, 3, 224, 224])
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype('float32') / 255.0
            img -= np.array([0.485, 0.456, 0.406])
            img /= np.array([0.229, 0.224, 0.225])
            img = img.transpose((2, 0, 1))  # 转换为CHW格式
            img = np.expand_dims(img, axis=0)  # 增加batch维度
            return paddle.to_tensor(img)
        except Exception as e:
            print(f"图像预处理失败: {str(e)}")
            return paddle.zeros([1, 3, 224, 224])
    
    def embed_documents(self, image_paths: List[str]) -> List[List[float]]:
        """生成图像的嵌入向量"""
        embeddings = []
        for image_path in image_paths:
            try:
                # 使用简单的哈希方法生成固定维度的向量，避免模型加载问题
                import hashlib
                hash_obj = hashlib.md5(image_path.encode())
                hash_bytes = hash_obj.digest()
                
                # 将哈希值转换为固定维度的向量
                vector = []
                for i in range(self.dimension):
                    byte_index = i % len(hash_bytes)
                    vector.append(float(hash_bytes[byte_index]) / 255.0)
                
                embeddings.append(vector)
            except Exception as e:
                print(f"处理图像时出错 {image_path}: {str(e)}")
                # 返回零向量作为备选
                embeddings.append([0.0] * self.dimension)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """为文本查询提供一个占位符嵌入"""
        try:
            # 使用简单的哈希方法生成固定维度的向量，避免重新加载模型
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            
            # 将哈希值转换为固定维度的向量
            vector = []
            for i in range(self.dimension):
                byte_index = i % len(hash_bytes)
                vector.append(float(hash_bytes[byte_index]) / 255.0)
            
            return vector
        except Exception as e:
            print(f"文本嵌入出错: {str(e)}")
            # 返回零向量作为后备
            return [0.0] * self.dimension
from openai import OpenAI
def call_ernie_api_fixed(prompt, context):
    """调用文心一言API来回答问题 - 修复版本兼容性问题"""
    client = OpenAI(
        api_key="1bc3aca311f155f00ad7a33d2eb5b86c472e558b",  # Access Token属于个人账户的重要隐私信息，请谨慎管理，切忌随意对外公开,
        base_url="https://aistudio.baidu.com/llm/lmapi/v3",  # aistudio 大模型 api 服务域名
    )
    
    
    formatted_prompt = f"""
你是一个智能助手，能够理解视觉和文本内容。

你将获得两个信息: 场景描述和用户问题。你需要理解场景描述并回答用户问题。

作为助手，回答问题时需要遵循以下规则:
- 不要回答与提供的场景描述无关的问题
- 不要包含有害信息
- 如果能从提供的场景描述中回答，就回答；否则就说你没有足够信息来回答问题

场景描述: {context}

用户问题: {prompt}
"""
    
    completion = client.chat.completions.create(
        model="ernie-4.5-turbo-128k-preview",
        messages=[
            {
                "role": "system",
                "content": "你是百度研发的知识增强大语言模型文心一言，请帮助分析视频内容并回答用户问题。"
            },
            {
                "role": "user", 
                "content": [{"type": "text", "text": formatted_prompt}]
            }
        ],
        temperature=0.7
    )
    return completion
        

def simple_text_chatbot_fixed(text_db, video_list, user_query):
    """只基于文本的聊天机器人 - 带API调用修复"""
    print(f"查询: {user_query}")
    
    try:
        # 只使用文本检索
        text_results = text_db.similarity_search(user_query, k=3)
        
        if not text_results:
            return "未找到相关视频。", None
            
        # 分析结果找最佳视频
        video_scores = {}
        for result in text_results:
            if "video" in result.metadata:
                video_name = result.metadata["video"]
                if video_name not in video_scores:
                    video_scores[video_name] = 0
                video_scores[video_name] += 1
        
        if not video_scores:
            return "无法确定相关视频。", None
            
        # 获取最高分的视频
        video_name = max(video_scores.items(), key=lambda x: x[1])[0]
        
        if video_name not in video_list:
            return f"找到视频 {video_name}，但缺少描述信息。", None
        
        # 获取场景描述和调用修复后的API
        scene_des = video_list[video_name]
        response = call_ernie_api_fixed(user_query, scene_des)
        # print(f"API调用结果: {response}")
        response = response.choices[0].message.content
        
        full_response = f"最相关的视频是 **{video_name}** \n\n{response}"
        
        video_dir = str(datapath / "VideoSumForRetailData/clips/")
        video_path = os.path.join(video_dir, video_name)
        
        return full_response, video_path
    except Exception as e:
        print(f"处理查询时出错: {str(e)}")
        return f"处理查询时出现错误: {str(e)}", None

# # 测试使用修复后的聊天机器人
# test_query = "找穿卡其裤的男人"
# print(f"测试查询: {test_query}")
# response, video_path = simple_text_chatbot_fixed(text_db, video_list, test_query)
# print("\n完整回复:")
# print(response)