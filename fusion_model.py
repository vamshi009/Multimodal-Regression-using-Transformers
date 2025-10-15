import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import pandas as pd
from transformers import BertTokenizer, ViTImageProcessor,  BertModel, ViTModel
import numpy as np

class MultimodalDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load and process image
        image = Image.open(row["image_path"]).convert("RGB")
        image_tensor = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        # Tokenize text
        text_inputs = self.tokenizer(row["catalog_content"], return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0)

        # Target value
        target = torch.tensor(row["price"], dtype=torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_tensor": image_tensor,
            "target": target
        }




class MultimodalFusionRegressor(nn.Module):
    def __init__(self,
                 text_model_name="bert-base-uncased",
                 vision_model_name="google/vit-base-patch16-224",
                 hidden_dim=512):
        super(MultimodalFusionRegressor, self).__init__()

        # Load pretrained models
        self.text_model = BertModel.from_pretrained(text_model_name)
        self.vision_model = ViTModel.from_pretrained(vision_model_name)

        # Freeze encoders if desired (optional)
        # for param in self.text_model.parameters():
        #     param.requires_grad = False
        # for param in self.vision_model.parameters():
        #     param.requires_grad = False

        # Dimensions of CLS tokens
        text_dim = self.text_model.config.hidden_size   # e.g. 768
        image_dim = self.vision_model.config.hidden_size  # e.g. 768

        # Fusion + Regression head
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)  # Output: single continuous value
        )

    def forward(self, input_ids, attention_mask, image_tensor):
        # Text forward
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = text_outputs.last_hidden_state[:, 0, :]  # CLS token

        # Image forward
        image_outputs = self.vision_model(pixel_values=image_tensor)
        image_cls = image_outputs.last_hidden_state[:, 0, :]  # CLS token

        # Concatenate
        fused = torch.cat((text_cls, image_cls), dim=1)

        # Regression output
        output = self.fusion(fused)
        return output.squeeze(1)  # Shape: (batch,)


# Load your dataset
dataset = MultimodalDataset("train_with_image_paths.csv")  # replace with your CSV
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize model
model = MultimodalFusionRegressor()
model = model.cuda() if torch.cuda.is_available() else model

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image_tensor = batch["image_tensor"].to(device)
        targets = batch["target"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, image_tensor=image_tensor)

        loss = criterion(outputs, targets)
        total_loss += loss.item()

        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")




dataset = MultimodalDataset("holdout_with_image_paths.csv")  # replace with your CSV
test_loader = DataLoader(dataset, batch_size=8, shuffle=True)

model.eval()
total_preds = []
gt = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image_tensor = batch["image_tensor"].to(device)
        targets = batch["target"].to(device)

        preds = model(input_ids=input_ids, attention_mask=attention_mask, image_tensor=image_tensor)
        print("Preds:", preds)
        print("Targets:", targets)
        total_preds = total_preds + list(preds)
        gt = gt + list(targets)

print("len of preds", len(total_preds))
print("len of targets ", len(gt))


def compute_smape(true_vals, pred_vals):
    t = np.asarray(true_vals, dtype=float)
    p = np.asarray(pred_vals, dtype=float)
    denom = (np.abs(t) + np.abs(p)) / 2.0
    denom = np.where(denom == 0, 1e-8, denom)
    return np.mean(np.abs(p - t) / denom) * 100.0

print(compute_smape(total_preds, gt))

       
