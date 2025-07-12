import torch
from tqdm import tqdm
from .dataloader import train_loader, val_loader
from .model import Model, device, criterion, optimizer
from .callbacks import Callbacks



# Training loop with tqdm progress bar showing accuracy and loss per batch, and epoch metrics in tqdm bar, using Callbacks
def train(model, criterion, optimizer, train_loader, val_loader, epochs=10, callbacks=None):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, ncols=120, colour='blue')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

            # Calculate training accuracy
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long().squeeze(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            # Show batch loss and accuracy on tqdm bar
            batch_loss = train_loss / train_total
            batch_acc = train_correct / train_total
            pbar.set_postfix({'batch_loss': f'{batch_loss:.4f}', 'batch_acc': f'{batch_acc:.4f}'})

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                val_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long().squeeze(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

        # Show epoch metrics in tqdm bar
        tqdm.write(
            f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.4f}'
        })

        # Callbacks
        if callbacks is not None:
            callbacks.model_checkpoint.step(model, val_acc)
            if callbacks.early_stopping.step(val_loss, val_acc):
                print("Early stopping triggered.")
                break

# Instantiate callbacks
# Use the existing callbacks object, or create if not already present
callbacks = Callbacks(patience=15, min_delta=0.0001, best_model_path='best_model.pth')

# Example usage:
train(Model, criterion, optimizer, train_loader, val_loader, epochs=100, callbacks=callbacks)
