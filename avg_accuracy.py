
import csv, matplotlib.pyplot as plt
import os, sys

currentPath = os.path.dirname(os.path.abspath(__file__))
def read_csv():
    epochs, accuracy0_, accuracy5_, accuracy10_ = [], [], [], []
    # Open train loss csv file and read data, store avg of folds per epoch
    with open(os.path.join(currentPath, "./train_epochs100_augment5_lr0.002_202605281819/val_recall_anomaly.csv"), 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            epochs.append(int(row[0]))
            accuracy0 = list(map(float, row[1:]))
            accuracy0_.append(sum(accuracy0) / len(accuracy0))  # Average of folds
    
    # Open train loss csv file and read data, store avg of folds per epoch
    with open(os.path.join(currentPath, "./train_epochs100_augment5_lr0.002_do0.05_202605291406/val_recall_anomaly.csv"), 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            accuracy5 = list(map(float, row[1:]))
            accuracy5_.append(sum(accuracy5) / len(accuracy5))  # Average of folds
    
    # Open train loss csv file and read data, store avg of folds per epoch
    with open(os.path.join(currentPath, "./train_epochs100_augment5_lr0.002_do0.1_202605291311/val_recall_anomaly.csv"), 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            accuracy10 = list(map(float, row[1:]))
            accuracy10_.append(sum(accuracy10) / len(accuracy10))  # Average of folds
    return epochs, accuracy0_, accuracy5_, accuracy10_

def plot_chart(epochs, accuracy0, accuracy5, accuracy10):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracy0, label='Dropout 0.00')
    plt.plot(epochs, accuracy5, label='Dropout 0.05')
    plt.plot(epochs, accuracy10, label='Dropout 0.10')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(currentPath, "train_val_accuracy_dropouts_chart.svg"), format='svg')
    plt.show()


if __name__ == "__main__":
    epochs, accuracy0, accuracy5, accuracy10 = read_csv()
    plot_chart(epochs, accuracy0, accuracy5, accuracy10)