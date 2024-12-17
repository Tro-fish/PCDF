# 파일에서 'caption en' 제거
def remove_caption_en(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # 'caption en'이 포함된 라인은 무시
            if line.strip().lower() != "caption en":
                outfile.write(line)

# 입력 및 출력 파일 경로 설정
input_file = "finetuned_pred_answers.txt"  # 원본 파일 경로
output_file = "finetuned_answers_filtered.txt"  # 'caption en' 제거 후 저장할 파일 경로

# 함수 실행
remove_caption_en(input_file, output_file)

print(f"Processed file saved to: {output_file}")