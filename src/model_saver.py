import os
import torch
import csv 

class ModelSaver:
    def __init__(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.best_val = -1
        self.best_model_path = os.path.join(save_dir, 'best_model.pth')
        self.final_model_path = os.path.join(save_dir, 'final_model.pth')
        self.log_path = os.path.join(save_dir, 'log.txt')
        self.round_path = os.path.join(save_dir, 'last_round.txt')

    def save_log(self, round_num, val_acc):
        with open(self.log_path, "a") as f:
            f.write(f"Round {round_num} | Val Acc: {val_acc:.4f}\n")

    def save_best(self, model, val_acc):
        if val_acc > self.best_val:
            self.best_val = val_acc
            torch.save(model.state_dict(), self.best_model_path)

    def save_final(self, model):
        torch.save(model.state_dict(), self.final_model_path)

    def save_round(self, round_num):
        with open(self.round_path, "w") as f:
            f.write(str(round_num))

    def export_log(self, output_csv_name="final_test_results.csv"):
        """
        Reads the existing log.txt file and exports it to a CSV file.
        Assumes log format is 'key,value' per line (e.g., model_name_final_test_accuracy,73.2700)
        """
        csv_path = os.path.join(self.save_dir, output_csv_name)
        log_entries = []

        # Read log.txt
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(',')
                        if len(parts) == 2:
                            key, value = parts[0].strip(), parts[1].strip()
                            log_entries.append((key, value))

        # Write to CSV
        if log_entries:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['model_name', 'final_test_accuracy'])
                for key, value in log_entries:
                    writer.writerow([key, value])
            print(f"✅ Final test results exported to: {csv_path}")
        else:
            print("⚠️ No log entries found. CSV not created.")