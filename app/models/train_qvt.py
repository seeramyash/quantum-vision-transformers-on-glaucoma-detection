import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from glaucoma_dataset import GlaucomaDataset
from qvt_model import QVTModel

# Load and split data
csv_file = '../../resize/cleaned_patient_data_od_with_images.csv'  # adjust as needed
df = pd.read_csv(csv_file)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Label'])

train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)

train_dataset = GlaucomaDataset('train.csv')
val_dataset = GlaucomaDataset('val.csv')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = QVTModel().to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(5):  # Increase epochs for better results
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save the model
torch.save(model.state_dict(), 'qvt_trained.pth')
print("Training complete and model saved.")