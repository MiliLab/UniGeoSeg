import sys
sys.path.append('')

import os
import glob
import random
import re
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import bisect
import torch
import numpy as np
import transformers

from unigeoseg.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX , REFER_TOKEN_INDEX, ANSWER_TOKEN_INDEX
from torch.utils.data import Dataset
from unigeoseg import conversation as conversation_lib
from unigeoseg.model import *
from unigeoseg.mm_utils import tokenizer_image_token

from PIL import Image
import cv2

from unigeoseg.mask_config.config import Config
from fvcore.common.config import CfgNode

from unigeoseg.eval_and_test.refer import REFER

from transformers import AutoTokenizer

import re
import torch

# 针对用户坐标格式的正则表达式（和之前一致，但在数据集端使用）
BOX_PATTERN = re.compile(r'x0,y0=\[([\d.]+),([\d.]+)\],\s*x1,y1=\[([\d.]+),([\d.]+)\]')
POINT_PATTERN = re.compile(r'\[([\d.]+),([\d.]+)\]')

def extract_intera_coords(raw_text):
    """
    从原始文本中提取intera任务的坐标（框/点）
    返回：coord_tensor（框：[1,4]；点：[n,2]，n=1-3；无坐标：None）
    """
    # 优先匹配框坐标
    box_match = BOX_PATTERN.search(raw_text)
    if box_match:
        x0, y0, x1, y1 = [float(box_match.group(i)) for i in range(1, 5)]
        return torch.tensor([[x0, y0, x1, y1]])  # [1,4]
    
    # 再匹配点坐标（1-3个）
    point_matches = POINT_PATTERN.findall(raw_text)
    if point_matches and 1 <= len(point_matches) <= 3:
        points = [[float(x), float(y)] for x, y in point_matches]
        return torch.tensor(points)  # [n,2]（n=1-3）
    
    # 无有效坐标
    return None

def preprocess_mask(mask, image_size):
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=0)
    bs, h, w = mask.shape
    processed_masks = []
    
    for i in range(bs):
        single_mask = mask[i]
        hh, ww = single_mask.shape[:2]
        if ww > hh:
            new_w = image_size
            new_h = int(hh * (image_size / ww))
        else:
            new_h = image_size
            new_w = int(ww * (image_size / hh))
        resized_mask = cv2.resize(single_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        pad_h = image_size - new_h
        pad_w = image_size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        padded_mask = cv2.copyMakeBorder(resized_mask, top, bottom, left, right, 
                                         cv2.BORDER_CONSTANT, value=0)
        processed_masks.append(padded_mask)
        
    processed_masks = np.stack(processed_masks, axis=0) 
    processed_masks = torch.from_numpy(processed_masks).to(torch.uint8)
    return processed_masks

def preprocess_image(image, image_size, pad_value=0):
    h, w = image.shape[:2]
    if w > h:
        new_w = image_size
        new_h = int(h * (image_size / w))
    else:
        new_h = image_size
        new_w = int(w * (image_size / h))
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = image_size - new_h
    pad_w = image_size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, 
                                      cv2.BORDER_CONSTANT, value=pad_value)
    padded_image = padded_image.transpose(2,0,1)
    return padded_image

def tokenizer_special_tokens(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                seg_token_index=SEG_TOKEN_INDEX, refer_token_index=REFER_TOKEN_INDEX, answer_token_index = ANSWER_TOKEN_INDEX, return_tensors=None):
        input_ids = []
        special_token_map = {'<image>': image_token_index, '<seg>': seg_token_index, '<refer>':refer_token_index, '<answer>':answer_token_index}
        prompt_chunks = re.split('(<image>|<seg>|<refer>|<answer>)', prompt)

        for chunk in prompt_chunks:
            if chunk in special_token_map:
                input_ids.append(special_token_map[chunk])
            else:
                input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        else:
            return input_ids

def preprocess_llama2(sources, tokenizer):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    input_ids = torch.stack(
        [tokenizer_special_tokens(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer_special_tokens(rou, tokenizer))
            instruction_len = len(tokenizer_special_tokens(parts[0], tokenizer)) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_referring_instruction(instruction, tokenizer, REFER_token='[SEG]'):
    tokenized = tokenizer.encode(instruction, add_special_tokens=False)
    tokenized = tokenized + [tokenizer.encode(REFER_token, add_special_tokens=False)[0]]

    token_refer_id = torch.tensor(tokenized)

    return token_refer_id

class RefSegRSDataset(Dataset):
    def __init__(self, base_data_path, tokenizer, split, image_size = 1024):
        super(RefSegRSDataset, self).__init__()
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.base_data_path = base_data_path
        self.tokenizer = tokenizer
        self.image_size = image_size
        split = split.lower()
        if split not in ['val', 'test']:
            print(f"[Warning] Unknown split '{split}'. Expected one of ['val', 'test']. Defaulting to 'val'.")
            split = 'val'

        self.RefSegRS_images_root = os.path.join(base_data_path, "rs_ref_seg/RefSegRS/images")
        self.RefSegRS_labels_root = os.path.join(base_data_path, "rs_ref_seg/RefSegRS/masks")
        self.RefSegRS_txt = os.path.join(base_data_path, f"rs_ref_seg/RefSegRS/output_phrase_{split}.txt")
        with open(self.RefSegRS_txt, 'r') as file:
            phases = file.readlines()
        images = []
        masks = []
        refs = []
        for phase in phases:
            match = re.match(r'(\d+)\s+(.*)', phase.strip())
            if match:
                image_path = os.path.join(self.RefSegRS_images_root, match.group(1)+'.tif')
                label_path = os.path.join(self.RefSegRS_labels_root, match.group(1)+'.tif')
                ref = match.group(2)
                images.append(image_path)
                masks.append(label_path)
                refs.append(ref)
        self.images = images
        self.masks = masks
        self.refs = refs
    def __len__(self):
        return len(self.images) 
    
    def __getitem__(self, idx):
        data_dict = {}
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = preprocess_image(image, self.image_size)
        processed_image = (torch.tensor(processed_image) - self.pixel_mean) / self.pixel_std
        data_dict['image'] = processed_image
        data_dict['image_name'] = os.path.basename(image_path).split('.')[0]
        
        ref = self.refs[idx]
        label_path = self.masks[idx]
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 1
        processed_mask = preprocess_mask(mask, self.image_size)
        data_dict['mask'] = processed_mask
        
        prefix_inst = 'This is an image <image>, Please doing Referring Segmentation according to the following instruction:'
        instruction = ' {}'.format(ref)
        sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                    {'from': 'gpt', 'value': '\nSure. It is <seg>. '}]]
        
        text_dict = preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels
        data_dict['dataset_type'] = 'refer_seg'
        
        token_refer_id = preprocess_referring_instruction(instruction, self.tokenizer)
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1
        data_dict['token_refer_id'] = token_refer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        
        return data_dict

class RRSISDDataset(Dataset):
    def __init__(self, base_data_path, tokenizer, split, image_size = 1024):
        super(RRSISDDataset).__init__()
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.base_data_path = base_data_path
        self.tokenizer = tokenizer
        self.image_size = image_size
        split = split.lower()
        if split not in ['val', 'test']:
            print(f"[Warning] Unknown split '{split}'. Expected one of ['val', 'test']. Defaulting to 'val'.")
            split = 'val'
        self.RRSISD_data_root = os.path.join(base_data_path, "rs_ref_seg/RRSIS-D/")
        refer = REFER(self.RRSISD_data_root,'rrsisd','unc')
        ref_ids = refer.getRefIds(split = split)
        all_imgs = refer.Imgs
        imgs = list(all_imgs[i] for i in ref_ids)
        images =[]
        for img in imgs:
            image_path = os.path.join(self.RRSISD_data_root, 'images/rrsisd/JPEGImages', img['file_name'])
            images.append(image_path)
        refs = []
        for i, img_id in enumerate(ref_ids):
            ref = refer.Refs[img_id]
            refs.append(ref['sentences'][0]['raw'])
        masks = []
        for i in ref_ids:
            mask = refer.getMask(i)
            masks.append(mask)
        self.images = images
        self.refs = refs
        self.masks = masks
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        data_dict = {}
        image_path = self.images[idx]
        # print(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = preprocess_image(image, self.image_size)
        # print(processed_image.shape)
        processed_image = (torch.tensor(processed_image) - self.pixel_mean) / self.pixel_std
        data_dict['image'] = processed_image
        
        data_dict['image_name'] = os.path.basename(image_path).split('.')[0]
        ref = self.refs[idx]
        mask = self.masks[idx]
        # print(mask.shape)
        processed_mask = preprocess_mask(mask, self.image_size)
        # mask = self.data_args.mask_processor(mask)
        data_dict['mask'] = processed_mask
        
        prefix_inst = 'This is an image <image>, Please doing Referring Segmentation according to the following instruction:'
        instruction = ' {}'.format(ref)
        sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                    {'from': 'gpt', 'value': '\nSure.'}]]
        
        text_dict = preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels
        data_dict['dataset_type'] = 'refer_seg'
        
        token_refer_id = preprocess_referring_instruction(instruction, self.tokenizer)
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1
        data_dict['token_refer_id'] = token_refer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        
        return data_dict
    
class ReasonSegDataset(Dataset):
    def __init__(self, base_data_path, tokenizer, split, image_size = 512):
        super(ReasonSegDataset, self).__init__()
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.base_data_path = base_data_path
        self.tokenizer = tokenizer
        self.image_size = image_size
        split = split.lower()
        if split not in ['val', 'test']:
            print(f"[Warning] Unknown split '{split}'. Expected one of ['val', 'test']. Defaulting to 'val'.")
            split = 'val'
        self.ReasonSeg_images_root = os.path.join(base_data_path, f"rs_reason_seg/RSReasonSeg/{split}/images")
        self.ReasonSeg_labels_root = os.path.join(base_data_path, f"rs_reason_seg/RSReasonSeg/{split}/labels")
        self.ReasonSeg_QAs_root = os.path.join(base_data_path, f"rs_reason_seg/RSReasonSeg/{split}/QAs")
        
        self.images = self.load_file_paths(self.ReasonSeg_images_root, valid_extensions=('.jpg', '.jpeg', '.png'))
        self.labels = self.load_file_paths(self.ReasonSeg_labels_root, valid_extensions=('.png',))
        self.QAs_paths = self.load_file_paths(self.ReasonSeg_QAs_root, valid_extensions=('.json', '.txt'))
        self.QAs = []
        for QAs_path in self.QAs_paths:
            with open(QAs_path, "r") as file:
                QA = json.load(file)
            self.QAs.append(QA)
    
    def __len__(self):
        return len(self.images)
    
    def load_file_paths(self, directory, valid_extensions=None):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist.")
        
        file_paths = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                if valid_extensions is None or filename.lower().endswith(valid_extensions):
                    file_paths.append(file_path)
        
        file_paths.sort()
        print(f"Found {len(file_paths)} files in {directory}")
        return file_paths
    
    def __getitem__(self, idx):
        data_dict = {}
        image_path = self.images[idx]
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = preprocess_image(image, self.image_size)
        
        processed_image = (torch.tensor(processed_image) - self.pixel_mean) / self.pixel_std
        data_dict['image'] = processed_image
        data_dict['image_name'] = os.path.basename(image_path).split('.')[0]
        
        label_path = self.labels[idx]
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        mask[mask != 0] = 1
        processed_mask = preprocess_mask(mask, self.image_size)
        data_dict['mask'] = processed_mask
        
        QAs = self.QAs[idx]
        
        question_num = len(QAs["questions"])
        question_idx = random.randint(0, question_num - 1)
        question = QAs["questions"][0]
        
        answer_num = len(QAs["answer"])
        if answer_num == 0:
            answer = "There is no target object in the image."
        else:
            answer_idx = random.randint(0, answer_num - 1)
            answer = QAs["answer"][0]
        
        prefix_inst = 'This is an image <image>, Please doing Reasoning Segmentation according to the following instruction:'
        instruction = ' {}'.format(question)
        sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                    {'from': 'gpt', 'value': 'Sure, It is <seg>. \n<answer>.'}]]
        
        text_dict = preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels
        data_dict['dataset_type'] = 'reason_seg'
        
        token_refer_id = preprocess_referring_instruction(instruction, self.tokenizer)
        token_answer_id = preprocess_referring_instruction(answer, self.tokenizer)
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1
        answer_embedding_indices = torch.zeros_like(input_ids)
        answer_embedding_indices[input_ids == ANSWER_TOKEN_INDEX] = 1
        
        data_dict['token_refer_id'] = token_refer_id
        data_dict['token_answer_id'] = token_answer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        data_dict['answer_embedding_indices'] = answer_embedding_indices

        return data_dict   

class RSRSDataset(Dataset):
    def __init__(self, base_data_path, tokenizer, image_size=512, split="train"):  # 新增split参数
        super(RSRSDataset, self).__init__()
        # 图像归一化参数（与现有保持一致）
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.base_data_path = base_data_path
        self.tokenizer = tokenizer
        self.image_size = image_size

        # 加载三元组索引文件（根据split参数选择train.json或test.json）
        self.index_json = os.path.join(
            self.base_data_path, 
            f"{split}.json"  # 这里改为动态变量
        )
        with open(self.index_json, 'r') as f:
            self.samples = json.load(f)  # 解析三元组列表

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_dict = {}
        sample = self.samples[idx]

        # 1. 加载并预处理图像
        img_path = os.path.join(self.base_data_path, sample["img"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB
        processed_image = preprocess_image(image, self.image_size)  # 尺寸调整+padding
        processed_image = (torch.tensor(processed_image) - self.pixel_mean) / self.pixel_std  # 归一化
        data_dict["image"] = processed_image

        # 2. 加载并预处理掩码（0/255二值掩码→0/1）
        mask_path = os.path.join(self.base_data_path, sample["mask"])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度图
        mask[mask == 255] = 1  # 二值掩码转换（255→1，0保持不变）
        processed_mask = preprocess_mask(mask, self.image_size)  # 尺寸调整+padding
        data_dict["mask"] = processed_mask

        # 3. 加载并处理文本（question和reason）
        json_path = os.path.join(self.base_data_path, sample["json"])
        with open(json_path, 'r') as f:
            qa_data = json.load(f)
        question_num = len(qa_data["question"])
        question = qa_data["question"][0]  # 取第一个问题

        answer_num = len(qa_data["reason"])
        if answer_num == 0:
            answer = ""
        else:
            # answer_idx = random.randint(0, answer_num - 1)
            answer = qa_data["reason"][0]
        
        prefix_inst = 'This is an image <image>, Please doing Reasoning Segmentation according to the following instruction:'
        instruction = ' {}'.format(question)
        sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                    {'from': 'gpt', 'value': 'Sure, It is <seg>. \n<answer>.'}]]
        
        text_dict = preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels
        data_dict['dataset_type'] = 'reason_seg'
        if '/ref/' in img_path:
            data_dict['prompt_type'] = 'ref'
            data_dict['intera_coords'] = None
        elif '/intera/' in img_path:
            data_dict['prompt_type'] = 'intera'
            # -------------------------- 新增：提取intera坐标 --------------------------
            # 从原始文本（question）中提取坐标（未分词，格式完整）
            raw_instruction = qa_data["question"][0]  # intera的坐标在question中
            intera_coords = extract_intera_coords(raw_instruction)
            data_dict['intera_coords'] = intera_coords  # 存入数据字典
        else:
            data_dict['prompt_type'] = 'reason'
            data_dict['intera_coords'] = None  # 其他任务设为None
        
        token_refer_id = preprocess_referring_instruction(instruction, self.tokenizer)
        token_answer_id = preprocess_referring_instruction(answer, self.tokenizer)
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1
        answer_embedding_indices = torch.zeros_like(input_ids)
        answer_embedding_indices[input_ids == ANSWER_TOKEN_INDEX] = 1
        
        data_dict['token_refer_id'] = token_refer_id
        data_dict['token_answer_id'] = token_answer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        data_dict['answer_embedding_indices'] = answer_embedding_indices
        
        return data_dict

class DataCollector(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data_dicts):
        input_ids = [data_dict['input_ids'] for data_dict in data_dicts]
        labels = [data_dict['labels'] for data_dict in data_dicts]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=IGNORE_INDEX
        )
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),    
        )
        
        if 'image' in data_dicts[0]:
            images = [data_dict['image'] for data_dict in data_dicts]
            batch['images'] = torch.stack(images)
        if 'mask' in data_dicts[0]:
            masks = [data_dict['mask'] for data_dict in data_dicts]
            batch['masks'] = torch.stack(masks)
        # 在处理dataset_type的代码块附近添加
        if 'prompt_type' in data_dicts[0]:
            batch['prompt_type'] = [data_dict['prompt_type'] for data_dict in data_dicts]
        if 'intera_coords' in data_dicts[0]:
            # 用列表收集（不同样本可能是框[1,4]或点[n,2]，不做padding）
            batch['intera_coords'] = [data_dict['intera_coords'] for data_dict in data_dicts]
        
        for data_dict in data_dicts:
            for key in ['input_ids', 'labels', 'image', 'prompt_type']:
                del data_dict[key]

        if 'dataset_type' in data_dicts[0]:
            batch['dataset_type'] = [data_dict['dataset_type'] for data_dict in data_dicts]
            
        if 'token_refer_id' in data_dicts[0]:
            token_refer_id = [data_dict['token_refer_id'] for data_dict in data_dicts]
            batch['token_refer_id'] = token_refer_id
        
        if 'token_answer_id' in data_dicts[0]:
            token_answer_id = [data_dict['token_answer_id'] for data_dict in data_dicts]
            batch['token_answer_id'] = token_answer_id
            
        
        if 'refer_embedding_indices' in data_dicts[0]:
            refer_embedding_indices = [data_dict['refer_embedding_indices'] for data_dict in data_dicts]
            refer_embedding_indices = torch.nn.utils.rnn.pad_sequence(
                refer_embedding_indices,
                batch_first=True,
                padding_value=0)
            batch['refer_embedding_indices'] = refer_embedding_indices
            
        if 'answer_embedding_indices' in data_dicts[0]:
            answer_embedding_indices = [data_dict['answer_embedding_indices'] for data_dict in data_dicts]
            answer_embedding_indices = torch.nn.utils.rnn.pad_sequence(
                answer_embedding_indices,
                batch_first=True,
                padding_value=0)
            batch['answer_embedding_indices'] = answer_embedding_indices
            
        if 'image_name' in data_dicts[0]:
            image_names = [data_dict['image_name'] for data_dict in data_dicts]
            batch['image_name'] = image_names
            
        return batch
if __name__ == "__main__":
    
    conversation_lib.default_conversation = conversation_lib.conv_templates['llava_phi']
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    dataset = RRSISDDataset(
    base_data_path = "/data1/lky/data/data", 
    tokenizer = tokenizer,
    image_size =1024,
    )
    
    data = dataset[0]
    print(data['image'].shape)
