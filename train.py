import os
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import (
    Compose, Normalize, RandomResizedCrop, ToTensor, ColorJitter
)
from PIL import Image

DATA_DIR = "./data" 
MODEL_CHECKPOINT = "google/vit-base-patch16-224"
OUTPUT_DIR = "./vit_finetuned"
NUM_EPOCHS = 3 

print("Carga de datasets")
try:
    dataset = load_dataset("imagefolder", data_dir=DATA_DIR)
except Exception as e:
    print(f"Error al cargar el dataset. Error: {e}")
    exit()

labels = dataset['train'].features['label'].names
num_labels = len(labels)
id2label = {i: label for i, label in enumerate(labels)}
print(f"Clases detectadas: {labels}")

print("Carga del procesador de imágenes y del modelo ViT")
processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=num_labels,
    id2label=id2label,
    ignore_mismatched_sizes=True 
)

normalize = Normalize(mean=processor.image_mean, std=processor.image_std)

# Capas de convolución
train_transforms = Compose([
    RandomResizedCrop(processor.size["height"]),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
    ToTensor(),
    normalize,
])

val_transforms = Compose([
    processor.to_pil_module.Resize(processor.size["height"]),
    processor.to_pil_module.CenterCrop(processor.size["height"]),
    ToTensor(),
    normalize,
])

def apply_train_transforms(examples):
    examples['pixel_values'] = [train_transforms(img.convert("RGB")) for img in examples['image']]
    return examples

def apply_val_transforms(examples):
    examples['pixel_values'] = [val_transforms(img.convert("RGB")) for img in examples['image']]
    return examples

dataset["train"] = dataset["train"].with_transform(apply_train_transforms)
dataset["test"] = dataset["test"].with_transform(apply_val_transforms)


metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5, 
    per_device_train_batch_size=16,
    num_train_epochs=NUM_EPOCHS, 
    fp16=torch.cuda.is_available(), 
    logging_steps=10,
    load_best_model_at_end=True,
    report_to="none"
)

print("\nIniciando entrenamiento...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=processor,
    compute_metrics=compute_metrics,
)

trainer.train()

final_save_path = os.path.join(OUTPUT_DIR, "final_model")
trainer.save_model(final_save_path)
print(f"\n Entrenamiento completado. Modelo guardado en: {final_save_path}")