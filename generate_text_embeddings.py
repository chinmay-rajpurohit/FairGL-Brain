import csv
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs("processed", exist_ok=True)

PROMPT_FILE = "processed/subject_prompts.txt"
OUTPUT_FILE = "processed/text_embeddings.pt"
ROI_PRIOR_FILE = "data/aal_roi_priors.csv"
ROI_OUTPUT_FILE = "processed/roi_prior_embeddings.pt"

model_name = "emilyalsentzer/Bio_ClinicalBERT"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name,
    low_cpu_mem_usage=False,
    use_safetensors=True,
    torch_dtype="auto"
).to(device)
model.eval()

batch_size = 16


def encode_texts(texts, label):
    embeddings = []

    print(f"Total {label}:", len(texts))

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            hidden_states = outputs.last_hidden_state

            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            summed = torch.sum(hidden_states * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            batch_embeddings = summed / counts

            embeddings.append(batch_embeddings.cpu())

    return torch.cat(embeddings, dim=0)


with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f.readlines() if line.strip()]

text_embeddings = encode_texts(prompts, "subject prompts")

print("Text embedding shape:", text_embeddings.shape)

torch.save(text_embeddings, OUTPUT_FILE)

print("Saved text embeddings to:", OUTPUT_FILE)

with open(ROI_PRIOR_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    roi_rows = sorted(reader, key=lambda row: int(row["roi_index"]))

roi_prior_texts = [
    f"{row['roi_name']}: {row['prior_text']}"
    for row in roi_rows
]

roi_embeddings = encode_texts(roi_prior_texts, "ROI priors")

print("ROI prior embedding shape:", roi_embeddings.shape)

torch.save(roi_embeddings, ROI_OUTPUT_FILE)

print("Saved ROI prior embeddings to:", ROI_OUTPUT_FILE)
