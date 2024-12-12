import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from accelerate import PartialState
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig
from tqdm import tqdm

# 모델 설정
device_string = PartialState().process_index
print("device_string: ", device_string)

# 경로 설정
adapter_path = "weight/checkpoint-5400"  
model_id = "google/paligemma-3b-pt-224" 

# Quantization 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Base 모델 로드
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={'': device_string}
)

# LoRA adapter 로드
# model = PeftModel.from_pretrained(model, adapter_path)
processor = PaliGemmaProcessor.from_pretrained(model_id)

# 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, json_path, processor):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        image_path = example["image_path"]
        question = example["input_query"]
        answer = example["caption"]
        image = Image.open(image_path).convert("RGB")

        return {
            'image': image,
            'question': question,
            'answer': answer
        }

# Collate function
def collate_fn(examples):
    questions = [example['question'] for example in examples]
    images = [example['image'] for example in examples]

    # Processor를 사용하여 텍스트와 이미지를 토큰화
    tokens = processor(
        text=questions,
        images=images,
        return_tensors="pt",
        padding="longest",
        max_length=512,
        truncation=True
    )
    return tokens, questions, [example['answer'] for example in examples]

# 데이터 로드
valid_path = "dataset/chart2text_statista/chart2text_statista_test.json"
valid_dataset = CustomDataset(valid_path, processor)
valid_loader = DataLoader(valid_dataset, batch_size=40, collate_fn=collate_fn)

# 추론
model.eval()
predicted_answers = []
true_answers = []

with torch.no_grad():
    # tqdm을 사용하여 진행 상황 표시
    for batch, questions, answers in tqdm(valid_loader, desc="Validation Progress"):
        batch = {k: v.to(f"cuda:{device_string}") for k, v in batch.items()}
        outputs = model.generate(**batch, max_length=512)
        decoded_answers = processor.batch_decode(outputs, skip_special_tokens=True)

        # 질문을 제거하고 최종 출력만 추가
        final_outputs = [output.replace(question, "").strip() for output, question in zip(decoded_answers, questions)]
        
        predicted_answers.extend(final_outputs)
        true_answers.extend(answers)

# 결과 저장
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

output_path = "baseline_pred_answers.txt"

with open(os.path.join(output_dir, output_path), "w") as pred_file, \
     open(os.path.join(output_dir, "true_answers.txt"), "w") as true_file:
    for pred, true in zip(predicted_answers, true_answers):
        pred_file.write(pred + "\n")
        true_file.write(true + "\n")

print(f"Predicted answers saved to {os.path.join(output_dir, output_path)}")
print(f"True answers saved to {os.path.join(output_dir, 'true_answers.txt')}")