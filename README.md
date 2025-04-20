<p align="center">
    <img src="https://github.com/user-attachments/assets/2cc030b4-87e1-40a0-b5bf-1b7d6b62820b" width="300">
</p>

# FramePack

Official implementation and desktop software for ["Packing Input Frame Context in Next-Frame Prediction Models for Video Generation"](https://lllyasviel.github.io/frame_pack_gitpage/).

Links: [**Paper**](https://lllyasviel.github.io/frame_pack_gitpage/pack.pdf), [**Project Page**](https://lllyasviel.github.io/frame_pack_gitpage/)

FramePack is a next-frame (next-frame-section) prediction neural network structure that generates videos progressively. 

FramePack compresses input contexts to a constant length so that the generation workload is invariant to video length.

FramePack can process a very large number of frames with 13B models even on laptop GPUs.

FramePack can be trained with a much larger batch size, similar to the batch size for image diffusion training.

**Video diffusion, but feels like image diffusion.**

# Requirements

Note that this repo is a functional desktop software with minimal standalone high-quality sampling system and memory management.

**Start with this repo before you try anything else!**

Requirements:

* Nvidia GPU in RTX 30XX, 40XX, 50XX series that supports fp16 and bf16. The GTX 10XX/20XX are not tested.
* Linux or Windows operating system.
* At least 6GB GPU memory.

To generate 1-minute video (60 seconds) at 30fps (1800 frames) using 13B model, the minimal required GPU memory is 6GB. (Yes 6 GB, not a typo. Laptop GPUs are okay.)

About speed, on my RTX 4090 desktop it generates at a speed of 2.5 seconds/frame (unoptimized) or 1.5 seconds/frame (teacache). On my laptops like 3070ti laptop or 3060 laptop, it is about 4x to 8x slower.

In any case, you will directly see the generated frames since it is next-frame(-section) prediction. So you will get lots of visual feedback before the entire video is generated.

# Installation

**Windows**:

One-click-package will be released soon. Please come back tomorrow.

**Linux**:

We recommend having an independent Python 3.10.

    #pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    pip uninstall torch torchvision torchaudio -y
    pip install -U torch torchvision torchaudio

    git clone https://github.com/svjack/FramePack && cd FramePack
    
    pip install -r requirements.txt
    pip install "httpx[socks]" gradio spaces 

To start the GUI, run:

    python demo_gradio.py --share

Note that it supports `--share`, `--port`, `--server`, and so on.

```python
from gradio_client import Client, handle_file

client = Client("http://localhost:7860/")
result = client.predict(
		input_image=handle_file('xiang0.jpg'),
		prompt="a young person with short, black hair and glasses, stands in front of a wooden wardrobe filled with various colored jackets. Xiang wears a light gray zip-up hoodie over a blue shirt, with a beige shoulder bag strap visible. capturing a casual and relaxed moment with a warm, natural lighting that highlights Xiang's friendly smile and gentle expression.",
		n_prompt="",
		seed=31337,
		total_second_length=10,
		latent_window_size=9,
		steps=25,
		cfg=1,
		gs=10,
		rs=0,
		gpu_memory_preservation=6,
		use_teacache=True,
		api_name="/process"
)
print(result)

from shutil import copy2
copy2(result[0]["video"], result[0]["video"].split("/")[-1])
```

#### XingQiu Demo 
```python
#### huggingface-cli download svjack/Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-SP --repo-type dataset --revision main --include "genshin_impact_XINGQIU_images_and_texts/*" --local-dir ./genshin_impact_XINGQIU_images_and_texts

import os
from tqdm import tqdm
from gradio_client import Client, handle_file
from shutil import copy2

# 设置路径
input_dir = "genshin_impact_XINGQIU_images_and_texts/genshin_impact_XINGQIU_images_and_texts"
output_dir = "genshin_impact_XINGQIU_FramePack"
os.makedirs(output_dir, exist_ok=True)

# 初始化Gradio客户端
client = Client("http://localhost:7860/")

# 获取所有png文件
png_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

# 处理每个文件对
for png_file in tqdm(png_files, desc="Processing files"):
    base_name = os.path.splitext(png_file)[0]
    txt_file = f"{base_name}.txt"

    # 检查对应的txt文件是否存在
    txt_path = os.path.join(input_dir, txt_file)
    if not os.path.exists(txt_path):
        print(f"Warning: Missing text file for {png_file}")
        continue

    # 读取提示文本
    with open(txt_path, 'r', encoding='utf-8') as f:
        prompt = f.read().strip()

    # 处理图像
    png_path = os.path.join(input_dir, png_file)
    try:
        result = client.predict(
            input_image=handle_file(png_path),
            prompt=prompt,
            n_prompt="",
            seed=31337,
            total_second_length=5,
            latent_window_size=9,
            steps=25,
            cfg=1,
            gs=10,
            rs=0,
            gpu_memory_preservation=6,
            use_teacache=True,
            api_name="/process"
        )

        # 复制视频文件
        video_path = result[0]["video"]
        output_video = os.path.join(output_dir, f"{base_name}.mp4")
        copy2(video_path, output_video)

        # 复制txt文件
        output_txt = os.path.join(output_dir, txt_file)
        copy2(txt_path, output_txt)

    except Exception as e:
        print(f"Error processing {png_file}: {str(e)}")

print("Processing completed!")
```

#### Aesthetics_X Demo
```python
#git clone https://huggingface.co/datasets/svjack/Aesthetics_X_Phone_Images_Rec_Captioned_5120x2880

import os
from tqdm import tqdm
from gradio_client import Client, handle_file
from shutil import copy2

# 设置路径
input_dir = "Aesthetics_X_Phone_Images_Rec_Captioned_5120x2880"
output_dir = "Aesthetics_X_FramePack"
os.makedirs(output_dir, exist_ok=True)

# 初始化Gradio客户端
client = Client("http://localhost:7860/")

# 获取所有png文件
png_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

# 处理每个文件对
for png_file in tqdm(png_files, desc="Processing files"):
    base_name = os.path.splitext(png_file)[0]
    txt_file = f"{base_name}.txt"

    output_txt = os.path.join(output_dir, txt_file)
    if os.path.exists(output_txt):
        print(f"Warning: Have text file for {png_file}")
        continue

    # 检查对应的txt文件是否存在
    txt_path = os.path.join(input_dir, txt_file)
    if not os.path.exists(txt_path):
        print(f"Warning: Missing text file for {png_file}")
        continue

    # 读取提示文本
    with open(txt_path, 'r', encoding='utf-8') as f:
        prompt = f.read().strip()

    # 处理图像
    png_path = os.path.join(input_dir, png_file)
    try:
        result = client.predict(
            input_image=handle_file(png_path),
            prompt=prompt,
            n_prompt="",
            seed=31337,
            total_second_length=5,
            latent_window_size=9,
            steps=25,
            cfg=1,
            gs=10,
            rs=0,
            gpu_memory_preservation=6,
            use_teacache=True,
            api_name="/process"
        )

        # 复制视频文件
        video_path = result[0]["video"]
        output_video = os.path.join(output_dir, f"{base_name}.mp4")
        copy2(video_path, output_video)

        # 复制txt文件
        output_txt = os.path.join(output_dir, txt_file)
        copy2(txt_path, output_txt)

    except Exception as e:
        print(f"Error processing {png_file}: {str(e)}")

print("Processing completed!")
```

#### Lelouch_Vi_Britanni Demo
```python
git clone https://huggingface.co/datasets/svjack/Lelouch_Vi_Britannia_Videos_Captioned

import os
import shutil
from moviepy.editor import VideoFileClip
import numpy as np
from skimage.metrics import structural_similarity as ssim

def extract_first_and_last_frame(video_path, output_dir='output_frames'):
    """
    提取视频的第一帧和最后一帧

    参数:
        video_path (str): 输入视频文件的路径
        output_dir (str): 输出目录，默认为'output_frames'

    返回:
        tuple: (第一帧路径, 最后一帧路径, 差异值)
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 获取视频文件名（不带扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 定义输出文件名
    first_frame_path = os.path.join(output_dir, f"{video_name}_first_frame.png")
    last_frame_path = os.path.join(output_dir, f"{video_name}_last_frame.png")
    diff_file_path = os.path.join(output_dir, f"{video_name}_frame_diff.txt")

    try:
        # 加载视频文件
        with VideoFileClip(video_path) as video:
            # 获取第一帧
            first_frame = video.get_frame(0)
            video.save_frame(first_frame_path, t=0)

            # 获取最后一帧
            duration = video.duration
            last_frame = video.get_frame(duration - 0.1)  # 稍微提前一点确保能获取到
            video.save_frame(last_frame_path, t=duration - 0.1)

            # 计算帧差异
            first_frame_gray = np.mean(first_frame, axis=2)
            last_frame_gray = np.mean(last_frame, axis=2)
            data_range = 1.0
            ssim_value, _ = ssim(first_frame_gray, last_frame_gray, full=True, data_range=data_range)
            difference = 1 - ssim_value

            # 保存差异值
            with open(diff_file_path, 'w') as f:
                f.write(str(difference))

        print(f"成功提取帧: {first_frame_path} 和 {last_frame_path}, 差异值: {difference}")
        return first_frame_path, last_frame_path, difference

    except Exception as e:
        print(f"处理视频时出错: {e}")
        return None, None, None

def process_all_videos(input_dir, output_dir):
    """
    处理输入目录中的所有MP4文件，并复制相关的文本文件

    参数:
        input_dir (str): 包含视频和文本文件的输入目录
        output_dir (str): 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有文件
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)

            # 处理MP4文件
            if file.lower().endswith('.mp4'):
                print(f"正在处理视频文件: {file_path}")
                extract_first_and_last_frame(file_path, output_dir)

            # 复制文本文件
            elif file.lower().endswith('.txt'):
                output_path = os.path.join(output_dir, file)
                shutil.copy2(file_path, output_path)
                print(f"已复制文本文件: {file_path} -> {output_path}")

if __name__ == "__main__":
    # 定义输入和输出目录
    input_dataset_dir = "Lelouch_Vi_Britannia_Videos_Captioned"
    output_dir = "Lelouch_Vi_Britannia_First_Last_Frame_Captioned_saved"

    # 处理所有视频和文本文件
    process_all_videos(input_dataset_dir, output_dir)

    print("所有文件处理完成!")

from datasets import Dataset, DatasetDict
import os
from PIL import Image

def create_huggingface_dataset(data_dir):
    """
    从指定目录创建HuggingFace数据集，每行包含:
    - first_frame: 第一帧图片
    - last_frame: 最后一帧图片
    - text: 关联的文本内容
    - frame_diff: 两帧差异值
    
    参数:
        data_dir (str): 包含图片和文本文件的目录路径
        
    返回:
        DatasetDict: 包含训练集和测试集的HuggingFace数据集
    """
    # 使用字典来存储匹配的文件，键为视频名前缀
    video_data = {}
    
    # 遍历目录收集文件并匹配
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            # 提取视频名前缀
            if '_first_frame.png' in file:
                prefix = file.split('_first_frame.png')[0]
                if prefix not in video_data:
                    video_data[prefix] = {'first_frame': file_path}
                else:
                    video_data[prefix]['first_frame'] = file_path
                    
            elif '_last_frame.png' in file:
                prefix = file.split('_last_frame.png')[0]
                if prefix not in video_data:
                    video_data[prefix] = {'last_frame': file_path}
                else:
                    video_data[prefix]['last_frame'] = file_path
                    
            elif '_frame_diff.txt' in file:
                prefix = file.split('_frame_diff.txt')[0]
                if prefix not in video_data:
                    video_data[prefix] = {}
                try:
                    with open(file_path, 'r') as f:
                        diff_value = float(f.read().strip())
                    video_data[prefix]['frame_diff'] = diff_value
                except:
                    video_data[prefix]['frame_diff'] = None
                    
            elif file.endswith('.txt') and '_frame_diff.txt' not in file:
                prefix = file.split('.txt')[0]
                if prefix not in video_data:
                    video_data[prefix] = {}
                video_data[prefix]['text_path'] = file_path
    
    # 准备最终数据集结构
    dataset_data = {
        'first_frame': [],
        'last_frame': [],
        'text': [],
        'frame_diff': []
    }
    
    # 处理每个视频的数据
    for prefix, data in video_data.items():
        # 确保所有必需字段都存在
        if 'first_frame' in data and 'last_frame' in data:
            # 读取文本内容
            text_content = ""
            if 'text_path' in data:
                try:
                    with open(data['text_path'], 'r', encoding='utf-8') as f:
                        text_content = f.read()
                except:
                    text_content = ""
            
            # 添加数据
            dataset_data['first_frame'].append(data['first_frame'])
            dataset_data['last_frame'].append(data['last_frame'])
            dataset_data['text'].append(text_content)
            dataset_data['frame_diff'].append(data.get('frame_diff', None))
    
    # 创建数据集
    dataset = Dataset.from_dict(dataset_data)
    
    # 添加图片处理功能
    def process_example(example):
        example['first_frame'] = Image.open(example['first_frame'])
        example['last_frame'] = Image.open(example['last_frame'])
        return example
    
    dataset = dataset.map(process_example)
    
    # 分割为训练集和测试集 (80-20分割)
    #dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    return dataset

# 使用示例
if __name__ == "__main__":
    data_dir = "Lelouch_Vi_Britannia_First_Last_Frame_Captioned_saved"
    dataset = create_huggingface_dataset(data_dir)
    
    # 保存数据集到磁盘
    save_path = "lelouch_dataset"
    dataset.save_to_disk(save_path)
    print(f"数据集已保存到: {save_path}")
    
    # 打印数据集信息
    print("\n数据集结构:")
    print(dataset)

    '''
    # 打印第一个示例
    print("\n第一个训练示例:")
    print({k: v if not isinstance(v, Image.Image) else "PIL Image" for k, v in dataset['train'][0].items()})
    '''

from datasets import Image as HfImage, Dataset
dataset = dataset.cast_column("first_frame", HfImage()).cast_column("last_frame", HfImage())
dataset.push_to_hub("svjack/Lelouch_Vi_Britannia_First_Last_Frame_Diff_Captioned")
```

```bash
git clone https://github.com/nirvash/FramePack && cd FramePack

python endframe.py --share --server "0.0.0.0" --port 7860
```

```python
import os
from datasets import load_dataset
from gradio_client import Client, handle_file
from shutil import copy2
from tqdm import tqdm
from PIL import Image
import tempfile

# Load dataset
ds = load_dataset("svjack/Lelouch_Vi_Britannia_First_Last_Frame_Diff_Captioned")
dataset = ds['train']  # Assuming we want the train split

# Create output directory
output_dir = "Lelouch_Vi_Britannia_FramePack_First_Last_Frame_Video_Captioned"
os.makedirs(output_dir, exist_ok=True)

# Initialize Gradio client
client = Client("http://localhost:7860/")

# Process each item with frame_diff < 0.9
for item in tqdm(dataset, desc="Processing videos"):
    if item['frame_diff'] >= 0.9:
        continue
        
    # Save frames to temporary files
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as first_frame_file, \
         tempfile.NamedTemporaryFile(suffix='.png', delete=False) as last_frame_file:
        
        # Save PIL images to temp files
        first_frame = item['first_frame']
        last_frame = item['last_frame']
        first_frame.save(first_frame_file.name)
        last_frame.save(last_frame_file.name)
        
        # Process with Gradio client
        result = client.predict(
            input_image=handle_file(first_frame_file.name),
            end_frame=handle_file(last_frame_file.name),
            prompt=item['text'],
            n_prompt="",
            seed=2119383695,
            total_second_length=3,
            latent_window_size=9,
            steps=25,
            cfg=1,
            gs=10,
            rs=0,
            gpu_memory_preservation=6,
            use_teacache=True,
            use_random_seed=True,
            save_section_frames=True,
            api_name="/process"
        )
        
        # Clean up temp files
        os.unlink(first_frame_file.name)
        os.unlink(last_frame_file.name)
    
    # Copy video to output directory
    video_path = result[0]["video"]
    video_filename = os.path.basename(video_path)
    output_video_path = os.path.join(output_dir, video_filename)
    copy2(video_path, output_video_path)
    
    # Save text to corresponding .txt file
    text_filename = os.path.splitext(video_filename)[0] + ".txt"
    text_path = os.path.join(output_dir, text_filename)
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(item['text'])
    
    tqdm.write(f"Processed: {video_filename}")

print("Processing complete!")
```

#### Yi Chen Frame 4
```python
#git clone https://huggingface.co/datasets/svjack/Yi_Chen_Dancing_Animation_Videos

import os
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image
from datasets import Dataset, Image as HfImage
from typing import Dict, List
from skimage.metrics import structural_similarity as ssim

def extract_video_frames(
    video_path: str,
    output_dir: str,
    num_frames: int = 4
) -> Dict[str, List[str]]:
    """
    从视频中提取均匀间隔的帧并保存
    
    参数:
        video_path (str): 视频文件路径
        output_dir (str): 输出目录
        num_frames (int): 要提取的帧数
        
    返回:
        Dict[str, List[str]]: 包含帧路径和差异值的字典
    """
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    result = {"frame_paths": [], "diffs": []}
    
    try:
        with VideoFileClip(video_path) as video:
            duration = video.duration
            frame_times = np.linspace(0, duration - 0.1, num_frames)
            
            # 提取并保存所有帧
            frames = []
            for i, t in enumerate(frame_times):
                frame_path = os.path.join(output_dir, f"{video_name}_frame_{i}.png")
                video.save_frame(frame_path, t=t)
                frames.append(video.get_frame(t))
                result["frame_paths"].append(frame_path)
            
            # 计算相邻帧差异
            for i in range(len(frames)-1):
                frame1 = np.mean(frames[i], axis=2)
                frame2 = np.mean(frames[i+1], axis=2)
                ssim_value, _ = ssim(frame1, frame2, full=True, data_range=1.0)
                result["diffs"].append(1 - ssim_value)
                
        return result
        
    except Exception as e:
        print(f"处理视频 {video_path} 时出错: {e}")
        return {"frame_paths": [], "diffs": []}

def process_videos_to_dataset(
    input_dir: str,
    output_dir: str,
    num_frames: int = 4
) -> Dataset:
    """
    处理所有视频并创建HuggingFace数据集
    
    参数:
        input_dir (str): 输入视频目录
        output_dir (str): 输出帧目录
        num_frames (int): 每个视频提取的帧数
        
    返回:
        Dataset: 包含所有视频帧的HuggingFace数据集
    """
    # 收集所有视频数据
    all_data = {
        "frame_0": [],
        "frame_1": [],
        "frame_2": [],
        "frame_3": [],
        "diff_0_1": [],
        "diff_1_2": [],
        "diff_2_3": []
    }
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.mp4'):
                video_path = os.path.join(root, file)
                print(f"正在处理: {file}")
                
                # 提取帧
                frame_data = extract_video_frames(video_path, output_dir, num_frames)
                
                if len(frame_data["frame_paths"]) == num_frames:
                    # 添加各帧路径
                    for i in range(num_frames):
                        all_data[f"frame_{i}"].append(frame_data["frame_paths"][i])
                    
                    # 添加帧间差异
                    for i in range(num_frames-1):
                        all_data[f"diff_{i}_{i+1}"].append(frame_data["diffs"][i])
    
    # 创建数据集
    dataset = Dataset.from_dict(all_data)
    
    # 将帧路径转换为PIL图像
    def load_images(example):
        for i in range(num_frames):
            example[f"frame_{i}"] = Image.open(example[f"frame_{i}"])
        return example
    
    dataset = dataset.map(load_images)
    
    # 转换为HuggingFace图像格式
    for i in range(num_frames):
        dataset = dataset.cast_column(f"frame_{i}", HfImage())
    
    return dataset

if __name__ == "__main__":
    # 配置参数
    input_dir = "Yi_Chen_Dancing_Animation_Videos"
    output_dir = "extracted_frames"
    num_frames = 4  # 每个视频提取4帧
    
    # 处理视频并创建数据集
    dataset = process_videos_to_dataset(input_dir, output_dir, num_frames)
    
    # 保存数据集
    save_path = "yi_chen_dancing_dataset"
    dataset.save_to_disk(save_path)
    
    # 上传到HuggingFace Hub
    dataset.push_to_hub("svjack/Yi_Chen_Dancing_Animation_Frames_4")
    
    print(f"数据集已创建并保存到: {save_path}")
```

```python
#### pip install moviepy==1.0.3 datasets

import os
import uuid
from datasets import load_dataset
from gradio_client import Client, handle_file
from shutil import copy2
from tqdm import tqdm
from PIL import Image
import tempfile
from moviepy.editor import concatenate_videoclips, VideoFileClip

# Load dataset
ds = load_dataset("svjack/Yi_Chen_Dancing_Animation_Frames_4")
dataset = ds['train']  # Assuming we want the train split

# Create output directory
output_dir = "Yi_Chen_Dancing_Animation_FramePack_Frames_4_Videos"
os.makedirs(output_dir, exist_ok=True)

# Initialize Gradio client
client = Client("http://localhost:7860/")

dataset_rev = [item for item in dataset]
dataset_rev = dataset_rev[::-1]

# Process each item
for item in tqdm(dataset, desc="Processing videos"):
    # Get all frames from the item
    frames = [
        item['frame_0'],
        item['frame_1'],
        item['frame_2'],
        item['frame_3']
    ]
    
    video_clips = []
    
    # Process each consecutive frame pair
    for i in range(len(frames) - 1):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as first_frame_file, \
             tempfile.NamedTemporaryFile(suffix='.png', delete=False) as last_frame_file:
            
            # Save PIL images to temp files
            frames[i].save(first_frame_file.name)
            frames[i+1].save(last_frame_file.name)
            
            # Process with Gradio client
            result = client.predict(
                input_image=handle_file(first_frame_file.name),
                end_frame=handle_file(last_frame_file.name),
                prompt="The girl dances gracefully, with clear movements, full of charm.",
                n_prompt="",
                seed=2119383695,
                total_second_length=3,
                latent_window_size=9,
                steps=25,
                cfg=1,
                gs=10,
                rs=0,
                gpu_memory_preservation=15,
                use_teacache=True,
                use_random_seed=True,
                save_section_frames=True,
                api_name="/process"
            )
            
            # Clean up temp files
            os.unlink(first_frame_file.name)
            os.unlink(last_frame_file.name)
        
        # Save intermediate video
        video_path = result[0]["video"]
        video_clips.append(VideoFileClip(video_path))
    
    # Generate unique filename using UUID
    video_filename = f"video_{uuid.uuid4()}.mp4"
    output_video_path = os.path.join(output_dir, video_filename)
    
    # Concatenate all video clips
    final_clip = concatenate_videoclips(video_clips)
    final_clip.write_videofile(output_video_path)
    
    # Close all clips to free resources
    for clip in video_clips:
        clip.close()
    
    tqdm.write(f"Processed: {video_filename}")

print("Processing complete!")

```

The software supports PyTorch attention, xformers, flash-attn, sage-attention. By default, it will just use PyTorch attention. You can install those attention kernels if you know how. 

For example, to install sage-attention (linux):

    pip install sageattention==1.0.6

However, you are highly recommended to first try without sage-attention since it will influence results, though the influence is minimal.

# GUI

![ui](https://github.com/user-attachments/assets/8c5cdbb1-b80c-4b7e-ac27-83834ac24cc4)

On the left you upload an image and write a prompt.

On the right are the generated videos and latent previews.

Because this is a next-frame-section prediction model, videos will be generated longer and longer.

You will see the progress bar for each section and the latent preview for the next section.

Note that the initial progress may be slower than later diffusion as the device may need some warmup.

# Sanity Check

Before trying your own inputs, we highly recommend going through the sanity check to find out if any hardware or software went wrong. 

Next-frame-section prediction models are very sensitive to subtle differences in noise and hardware. Usually, people will get slightly different results on different devices, but the results should look overall similar. In some cases, if possible, you'll get exactly the same results.

## Image-to-5-seconds

Download this image:

<img src="https://github.com/user-attachments/assets/f3bc35cf-656a-4c9c-a83a-bbab24858b09" width="150">

Copy this prompt:

`The man dances energetically, leaping mid-air with fluid arm swings and quick footwork.`

Set like this:

(all default parameters, with teacache turned off)
![image](https://github.com/user-attachments/assets/0071fbb6-600c-4e0f-adc9-31980d540e9d)

The result will be:

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/bc74f039-2b14-4260-a30b-ceacf611a185" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

**Important Note:**

Again, this is a next-frame-section prediction model. This means you will generate videos frame-by-frame or section-by-section.

**If you get a much shorter video in the UI, like a video with only 1 second, then it is totally expected.** You just need to wait. More sections will be generated to complete the video.

## Know the influence of TeaCache and Quantization

Download this image:

<img src="https://github.com/user-attachments/assets/42293e30-bdd4-456d-895c-8fedff71be04" width="150">

Copy this prompt:

`The girl dances gracefully, with clear movements, full of charm.`

Set like this:

![image](https://github.com/user-attachments/assets/4274207d-5180-4824-a552-d0d801933435)

Turn off teacache:

![image](https://github.com/user-attachments/assets/53b309fb-667b-4aa8-96a1-f129c7a09ca6)

You will get this:

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/04ab527b-6da1-4726-9210-a8853dda5577" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

Now turn on teacache:

![image](https://github.com/user-attachments/assets/16ad047b-fbcc-4091-83dc-d46bea40708c)

About 30% users will get this (the other 70% will get other random results depending on their hardware):

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/149fb486-9ccc-4a48-b1f0-326253051e9b" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>A typical worse result.</em>
    </td>
  </tr>
</table>

So you can see that teacache is not really lossless and sometimes can influence the result a lot.

We recommend using teacache to try ideas and then using the full diffusion process to get high-quality results.

This recommendation also applies to sage-attention, bnb quant, gguf, etc., etc.

## Image-to-1-minute

<img src="https://github.com/user-attachments/assets/820af6ca-3c2e-4bbc-afe8-9a9be1994ff5" width="150">

`The girl dances gracefully, with clear movements, full of charm.`

![image](https://github.com/user-attachments/assets/8c34fcb2-288a-44b3-a33d-9d2324e30cbd)

Set video length to 60 seconds:

![image](https://github.com/user-attachments/assets/5595a7ea-f74e-445e-ad5f-3fb5b4b21bee)

If everything is in order you will get some result like this eventually.

60s version:

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/c3be4bde-2e33-4fd4-b76d-289a036d3a47" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

6s version:

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/37fe2c33-cb03-41e8-acca-920ab3e34861" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

# More Examples

Many more examples are in [**Project Page**](https://lllyasviel.github.io/frame_pack_gitpage/).

Below are some more examples that you may be interested in reproducing.

---

<img src="https://github.com/user-attachments/assets/99f4d281-28ad-44f5-8700-aa7a4e5638fa" width="150">

`The girl dances gracefully, with clear movements, full of charm.`

![image](https://github.com/user-attachments/assets/0e98bfca-1d91-4b1d-b30f-4236b517c35e)

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/cebe178a-09ce-4b7a-8f3c-060332f4dab1" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

<img src="https://github.com/user-attachments/assets/853f4f40-2956-472f-aa7a-fa50da03ed92" width="150">

`The girl suddenly took out a sign that said “cute” using right hand`

![image](https://github.com/user-attachments/assets/d51180e4-5537-4e25-a6c6-faecae28648a)

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/116069d2-7499-4f38-ada7-8f85517d1fbb" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

<img src="https://github.com/user-attachments/assets/6d87c53f-81b2-4108-a704-697164ae2e81" width="150">

`The girl skateboarding, repeating the endless spinning and dancing and jumping on a skateboard, with clear movements, full of charm.`

![image](https://github.com/user-attachments/assets/c2cfa835-b8e6-4c28-97f8-88f42da1ffdf)

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/d9e3534a-eb17-4af2-a8ed-8e692e9993d2" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

<img src="https://github.com/user-attachments/assets/6e95d1a5-9674-4c9a-97a9-ddf704159b79" width="150">

`The girl dances gracefully, with clear movements, full of charm.`

![image](https://github.com/user-attachments/assets/7412802a-ce44-4188-b1a4-cfe19f9c9118)

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/e1b3279e-e30d-4d32-b55f-2fb1d37c81d2" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

<img src="https://github.com/user-attachments/assets/90fc6d7e-8f6b-4f8c-a5df-ee5b1c8b63c9" width="150">

`The man dances flamboyantly, swinging his hips and striking bold poses with dramatic flair.`

![image](https://github.com/user-attachments/assets/1dcf10a3-9747-4e77-a269-03a9379dd9af)

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/aaa4481b-7bf8-4c64-bc32-909659767115" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

<img src="https://github.com/user-attachments/assets/62ecf987-ec0c-401d-b3c9-be9ffe84ee5b" width="150">

`The woman dances elegantly among the blossoms, spinning slowly with flowing sleeves and graceful hand movements.`

![image](https://github.com/user-attachments/assets/396f06bc-e399-4ac3-9766-8a42d4f8d383)


<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/f23f2f37-c9b8-45d5-a1be-7c87bd4b41cf" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

<img src="https://github.com/user-attachments/assets/4f740c1a-2d2f-40a6-9613-d6fe64c428aa" width="150">

`The young man writes intensely, flipping papers and adjusting his glasses with swift, focused movements.`

![image](https://github.com/user-attachments/assets/c4513c4b-997a-429b-b092-bb275a37b719)

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/62e9910e-aea6-4b2b-9333-2e727bccfc64" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

# Prompting Guideline

Many people would ask how to write better prompts. 

Below is a ChatGPT template that I personally often use to get prompts:

    You are an assistant that writes short, motion-focused prompts for animating images.

    When the user sends an image, respond with a single, concise prompt describing visual motion (such as human activity, moving objects, or camera movements). Focus only on how the scene could come alive and become dynamic using brief phrases.

    Larger and more dynamic motions (like dancing, jumping, running, etc.) are preferred over smaller or more subtle ones (like standing still, sitting, etc.).

    Describe subject, then motion, then other things. For example: "The girl dances gracefully, with clear movements, full of charm."

    If there is something that can dance (like a man, girl, robot, etc.), then prefer to describe it as dancing.

    Stay in a loop: one image in, one motion prompt out. Do not explain, ask questions, or generate multiple options.

You paste the instruct to ChatGPT and then feed it an image to get prompt like this:

![image](https://github.com/user-attachments/assets/586c53b9-0b8c-4c94-b1d3-d7e7c1a705c3)

*The man dances powerfully, striking sharp poses and gliding smoothly across the reflective floor.*

Usually this will give you a prompt that works well. 

You can also write prompts yourself. Concise prompts are usually preferred, for example:

*The girl dances gracefully, with clear movements, full of charm.*

*The man dances powerfully, with clear movements, full of energy.*

and so on.

# Cite

    @article{zhang2025framepack,
        title={Packing Input Frame Contexts in Next-Frame Prediction Models for Video Generation},
        author={Lvmin Zhang and Maneesh Agrawala},
        journal={Arxiv},
        year={2025}
    }
