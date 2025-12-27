import numpy as np
import matplotlib.pyplot as plt

OUTPUT_PATH = 'Model_Accuracy.png'

def generate_training_accuracy_plot(output_path):
    epochs = np.arange(0, 10)
    train_accuracy = np.array([0.78, 0.98, 0.985, 0.995, 0.99, 0.993, 0.995, 0.998, 0.997, 0.999])
    val_accuracy = np.array([0.96, 0.965, 0.98, 0.975, 0.973, 0.975, 0.99, 0.985, 0.983, 0.985])
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accuracy, 'b-', linewidth=2, label='Train')
    plt.plot(epochs, val_accuracy, color='#ff7f0e', linewidth=2, label='Validation')
    
    plt.title('Model Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0.75, 1.01)
    plt.xlim(-0.2, 9.2)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    plt.close()


def main():
    print("Generating Model Accuracy Plot...")
    generate_training_accuracy_plot(OUTPUT_PATH)
    print("Done!")


if __name__ == "__main__":
    main()
