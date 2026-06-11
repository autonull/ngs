import os
import torch
import json
import matplotlib.pyplot as plt
from lean_ngs.dataset import get_mnist_dataloaders, get_split_mnist_dataloaders
from lean_ngs.model import LeanNGS
from lean_ngs.baselines import StandardMLP, FixedLeanNGS
from lean_ngs.train import train_regular_mnist, train_split_mnist

def plot_metrics(metrics_dict, title, filename):
    plt.figure(figsize=(12, 4))
    
    # Plot active units if available
    plt.subplot(1, 2, 1)
    for model_name, metrics in metrics_dict.items():
        if 'active_units' in metrics and metrics['active_units']:
            steps, units = zip(*metrics['active_units'])
            # Only plot if units are non-zero (i.e. not an MLP)
            if any(u > 0 for u in units):
                plt.plot(steps, units, label=model_name)
    plt.title('Active G-Units Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Number of Units')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    for model_name, metrics in metrics_dict.items():
        if 'test_acc' in metrics:
            epochs, accs = zip(*metrics['test_acc'])
            plt.plot(epochs, accs, label=model_name)
        elif 'task_accs' in metrics:
            # for split mnist, plot average accuracy over tasks seen so far
            avg_accs = []
            num_tasks = len(metrics['task_accs'][0]) # how many eval points
            for eval_idx in range(num_tasks):
                # calculate avg acc over tasks 0 to eval_idx
                acc_sum = 0
                for t in range(eval_idx + 1):
                    acc_sum += metrics['task_accs'][t][eval_idx][1]
                avg_accs.append(acc_sum / (eval_idx + 1))
            plt.plot(range(num_tasks), avg_accs, label=model_name)
            
    plt.title('Evaluation Accuracy')
    plt.xlabel('Epochs / Tasks Seen')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_catastrophic_forgetting(metrics_dict, filename):
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'orange', 'green']
    
    for idx, (model_name, metrics) in enumerate(metrics_dict.items()):
        if 'task_accs' not in metrics:
            continue
            
        task_0_accs = [acc for _, acc in metrics['task_accs'][0]]
        plt.plot(range(len(task_0_accs)), task_0_accs, marker='o', color=colors[idx % len(colors)], label=f"{model_name} (Task 0 Acc)")
        
    plt.title('Catastrophic Forgetting: Accuracy on Task 0 Over Time')
    plt.xlabel('Tasks Trained On')
    plt.ylabel('Accuracy on Task 0')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def run_experiments():
    os.makedirs('results', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # --- Experiment 1: Regular MNIST (Capacity Scaling) ---
    print("\n=== Experiment 1: Regular MNIST ===")
    train_loader, test_loader = get_mnist_dataloaders(batch_size=128)
    
    models = {
        'MLP_Baseline': StandardMLP(784, 10),
        'Fixed_LeanNGS': FixedLeanNGS(784, 10, k=32),
        'Adaptive_LeanNGS': LeanNGS(784, 10, k_init=32, max_k=512, adc_mode='pre_alloc')
    }
    
    metrics_exp1 = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        use_adc = 'Adaptive' in name
        metrics = train_regular_mnist(model, train_loader, test_loader, epochs=3, device=device, use_adc=use_adc)
        metrics_exp1[name] = metrics
        
    plot_metrics(metrics_exp1, 'Regular MNIST: Adaptive Growth vs Baselines', 'results/exp1_mnist.png')
    
    with open('results/exp1_metrics.json', 'w') as f:
        json.dump(metrics_exp1, f)
        

    # --- Experiment 2: Split-MNIST (Continual Learning) ---
    print("\n=== Experiment 2: Split-MNIST ===")
    tasks_train, tasks_test = get_split_mnist_dataloaders(batch_size=128)
    
    models_cl = {
        'MLP_Baseline': StandardMLP(784, 10),
        'Adaptive_LeanNGS': LeanNGS(784, 10, k_init=32, max_k=512, adc_mode='pre_alloc')
    }
    
    metrics_exp2 = {}
    for name, model in models_cl.items():
        print(f"\nTraining {name} on Split-MNIST...")
        use_adc = 'Adaptive' in name
        metrics = train_split_mnist(model, tasks_train, tasks_test, epochs_per_task=2, device=device, use_adc=use_adc)
        metrics_exp2[name] = metrics
        
    plot_metrics(metrics_exp2, 'Split-MNIST: Average Accuracy Over Seen Tasks', 'results/exp2_split_mnist.png')
    plot_catastrophic_forgetting(metrics_exp2, 'results/exp2_forgetting.png')
    
    with open('results/exp2_metrics.json', 'w') as f:
        json.dump(metrics_exp2, f)
        
    print("\nExperiments finished! Visualizations saved in 'results/' directory.")

if __name__ == '__main__':
    run_experiments()
