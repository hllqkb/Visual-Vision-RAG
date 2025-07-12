# 可视化与输出：呈现视频内容
# 此代码不在此处运行，而是在Streamlit应用中实现。以下是简化版的代码片段，供参考学习；


"""
文心一言4.5视频内容理解

"""

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

# 设置页面标题
st.title("文心一言4.5视频内容理解系统")

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
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
    
    def encode_image(self, image_path):
        """将图片文件编码为base64格式"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            st.error(f"图片编码错误: {str(e)}")
            return None
    
    def extract_frames(self, video_path, interval=5, scene_change_threshold=30):
        """从视频中提取关键帧"""
        if not os.path.exists(video_path):
            st.error(f"错误: 视频文件'{video_path}'不存在")
            return [], []
        
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
            
            # 保存关键帧
            frame_filename = os.path.join(self.temp_dir, f"frame_{int(current_time)}.jpg")
            cv2.imwrite(frame_filename, frame)
            
            frame_paths.append(frame_filename)
            timestamps.append(current_time)
            
            last_frame_time = current_time
            last_frame = frame.copy()
        
        cap.release()
        progress_bar.progress(1.0)
        progress_text.text(f"成功提取 {len(frame_paths)} 个关键帧")
        
        return frame_paths, timestamps
    
    def analyze_frame(self, frame_path, timestamp, prompt=None):
        """使用文心一言分析单个关键帧"""
        if not prompt:
            prompt = "请详细分析这张视频画面，描述画面中的主要对象、场景和动作。注意识别任何重要事件或活动。"
        
        # 编码图片
        base64_image = self.encode_image(frame_path)
        if not base64_image:
            return None
        
        try:
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
            
            # 格式化时间
            time_str = str(timedelta(seconds=int(timestamp)))
            
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
    此应用使用百度文心一言4.5大模型对视频内容进行智能分析，通过关键帧提取与时序整合的方法实现对视频完整内容的理解。
    <br><br>
    <b>功能特点：</b>
    <ul>
        <li>自动提取视频关键帧</li>
        <li>智能分析画面内容</li>
        <li>时序建模与整合</li>
        <li>生成事件时间线</li>
        <li>输出详细分析结果</li>
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
    "视频内容理解系统 v1.0\n\n"
    "基于文心一言4.5大模型\n\n"
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
    
    # 分析视频按钮 - 不使用任何现代参数
    analyze_btn = st.button("开始分析视频")
    
    if analyze_btn:
        # 保存上传的视频到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_video_path = tmp_file.name
        
        try:
            # 设置API密钥
            if not api_key:
                api_key = DEFAULT_API_KEY
                
            # 开始分析
            st.markdown("<div class='info-box'>开始分析视频...</div>", unsafe_allow_html=True)
            
            # 创建分析器并提取关键帧
            analyzer = VideoAnalyzer(api_key)
            
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
                result = analyzer.analyze_frame(frame_path, timestamp, prompt=frame_prompt)
                if result:
                    analyzer.frame_results.append(result)
            
            progress_text.text("关键帧分析完成，正在生成内容总结...")
            
            # 生成事件总结
            summary = analyzer.summarize_events(custom_prompt=summary_prompt)
            
            # 完成分析
            progress_text.text("✅ 分析完成！")
            st.success("视频分析已完成")
            
            # 显示分析结果
            st.header("分析结果")
            
            # 显示内容总结
            st.subheader("内容总结")
            st.markdown(f"<div class='info-box'>{summary.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
            
            # 显示关键帧时间线
            st.subheader("关键帧时间线")
            
            # 使用基础布局展示关键帧和分析结果
            for i, result in enumerate(analyzer.frame_results):
                st.markdown(f"### 时间点 {result['time_str']}")
                # 显示关键帧图像
                img = Image.open(result['frame_path'])
                st.image(img, caption=f"关键帧 {i+1}")
                # 显示分析结果
                st.markdown(f"**分析结果**:")
                st.markdown(f"<div class='info-box'>{result['analysis'].replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
                st.markdown("---")
            
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
    5. 查看分析结果
    """)
    
    # 展示视频理解方法
    st.subheader("视频理解方法")
    
    st.markdown("""
    本系统采用"离散采样 + 时序整合"的方法让多模态大模型理解视频内容：
    
    1. **离散采样**：将视频拆分为关键帧序列
    2. **时序保留**：为每帧保存时间信息
    3. **单帧分析**：分析每帧画面内容
    4. **时序整合**：结合时间信息理解整体视频
    5. **内容总结**：生成视频事件和内容概述
    """)

# 添加页脚
st.markdown("---")
st.markdown("基于百度文心一言4.5大模型 | 适用于AIStudio环境") 