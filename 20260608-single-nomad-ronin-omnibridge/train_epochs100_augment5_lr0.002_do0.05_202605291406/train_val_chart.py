import csv, matplotlib.pyplot as plt
import os, sys

currentPath = os.path.dirname(os.path.abspath(__file__))
def read_csv():
    epochs, train_loss_, val_loss_ = [], [], []
    # Open train loss csv file and read data, store avg of folds per epoch
    with open(os.path.join(currentPath, "./val_train_loss.csv"), 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            epochs.append(int(row[0]))
            train_loss = list(map(float, row[1:]))
            train_loss_.append(sum(train_loss) / len(train_loss))  # Average of folds
    
    with open(os.path.join(currentPath, "./val_val_loss.csv"), 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            val_loss = list(map(float, row[1:]))
            val_loss_.append(sum(val_loss) / len(val_loss))  # Average of folds
    
    return epochs, train_loss_, val_loss_

def plot_chart(epochs, train_acc, val_acc):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, label='Train Loss')
    plt.plot(epochs, val_acc, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(currentPath, "train_val_loss_chart.svg"), format='svg')
    plt.show()

if __name__ == "__main__":
    epochs, train_loss_, val_loss_ = read_csv()
    plot_chart(epochs, train_loss_, val_loss_)