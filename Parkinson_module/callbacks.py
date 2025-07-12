import os
import torch


class Callbacks:
    class EarlyStoppingByBoth:
        def __init__(self, patience=15, min_delta=0.0001):
            self.patience = patience
            self.min_delta = min_delta
            self.wait = 0
            self.best_loss = float('inf')
            self.best_acc = 0
            self.stopped = False

        def step(self, val_loss, val_acc):
            improved = False
            if (self.best_loss - val_loss) > self.min_delta:
                self.best_loss = val_loss
                self.wait = 0
                improved = True
            elif (val_acc - self.best_acc) > self.min_delta:
                self.best_acc = val_acc
                self.wait = 0
                improved = True
            else:
                self.wait += 1

            if self.wait >= self.patience:
                self.stopped = True
                print("Early stopping by Both metrics reached")
            return self.stopped

    # class ModelCheckpoint:
    #     def __init__(self, best_model_path='best_model.pth'):
    #         self.best_val_acc = 0.0
    #         self.best_model_path = best_model_path

    #     def step(self, model, val_acc):
    #         if val_acc > self.best_val_acc:
    #             self.best_val_acc = val_acc
    #             torch.save(model.state_dict(), self.best_model_path)
    #             print(f"Best model saved with val_acc: {val_acc:.4f}")


    class ModelCheckpoint:
        def __init__(self, best_model_path='best_model.pth'):
            self.best_val_acc = 0.0
            self.best_model_path = best_model_path
            self.last_path = None  # Track the last saved model path

        def step(self, model, val_acc):
            if val_acc > self.best_val_acc:
                # Only create new path when we have a better accuracy
                new_path = f"PD_torch_logs/best_model{val_acc:.4f}.pth"
                
                # Remove the previous file if it exists
                if self.last_path and os.path.exists(self.last_path):
                    try:
                        os.remove(self.last_path)
                    except Exception as e:
                        print(f"Warning: Could not remove old model file {self.last_path}: {e}")
                
                # Save the new best model
                try:
                    torch.save(model.state_dict(), new_path)
                    self.best_val_acc = val_acc
                    self.last_path = new_path
                    print(f"New best model saved with val_acc: {val_acc:.4f} at {new_path}")
                except Exception as e:
                    print(f"Error saving model: {e}")
                    
    def __init__(self, patience=15, min_delta=0.0001, best_model_path='best_model.pth'):
        self.early_stopping = self.EarlyStoppingByBoth(patience=patience, min_delta=min_delta)
        self.model_checkpoint = self.ModelCheckpoint(best_model_path=best_model_path)