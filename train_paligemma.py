import json
import os
import torch
import random
import wandb
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig
from accelerate import PartialState

# python -m torch.distributed.launch --nproc_per_node=2 train_paligemma.py
device_string = PartialState().process_index
print("device_string: ", device_string)

# 'best_model' 폴더에 저장된 adapter config와 weights를 불러옵니다.
adapter_path = "weight/checkpoint-200" # "./backup/decoder"  # adapter config와 model 파일이 위치한 폴더 경로
model_id = "google/paligemma-3b-ft-scicap-224"  # base 모델 ID

# qlora로 학습 진행
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, quantization_config=bnb_config, device_map={'':device_string})

for param in model.parameters():
    if param.dtype in [torch.float32, torch.float64, torch.float16]:
        param.requires_grad = True

# lora로 모델 로드
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 로컬 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, json_path, processor):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        image_path = (example["image_path"])
        question = example['input_query']
        answer = example['caption']
        image = Image.open(image_path).convert("RGB")

        return {
            'image': image,
            'question': question,
            'answer': answer
        }

# collate_fn 수정
def collate_fn(examples):
    texts = [example['question'] for example in examples]
    labels = [example['answer'] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]

    # processor를 사용하여 텍스트와 이미지를 토큰화
    tokens = processor(
        text=texts,
        images=images,
        suffix=labels,
        return_tensors="pt",
        padding="longest",
        tokenize_newline_separately=False,
        max_length=512,
        truncation=True
    )

    tokens = tokens.to(torch.bfloat16).to(f"cuda:{device_string}")
    return tokens

# 모델과 프로세서 설정
processor = PaliGemmaProcessor.from_pretrained(model_id)

# 데이터 경로 설정
train_path = "dataset/chart2text_statista/chart2text_statista_train.json"
valid_path = "dataset/chart2text_statista/chart2text_statista_val.json"

# 데이터셋 로드
train_dataset = CustomDataset(train_path, processor)

wandb.init(project="Chart-to-Text", name="PaliGemma_Training")

# TrainingArguments 설정
args = TrainingArguments(
    num_train_epochs=50,
    remove_unused_columns=False,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=32,
    warmup_steps=950,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=10,
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=100,
    push_to_hub=False,
    save_total_limit=5,
    output_dir="weight",
    bf16=True,
    report_to=["wandb"],
    dataloader_pin_memory=False,
    ddp_find_unused_parameters=False,
    gradient_checkpointing_kwargs={'use_reentrant':False}
)

# Trainer 설정
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    args=args
)

# 학습 시작
trainer.train()

# 모델 저장
trainer.save_model("weight")  # 모델 저장