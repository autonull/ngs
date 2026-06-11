import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import os

def train_regular_mnist(model, train_loader, test_loader, epochs=5, device='cpu', 
                        adc_freq=500, lambda_ent=0.01, use_adc=True, 
                        split_thresh=0.05, prune_thresh=0.01):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    metrics = {
        'train_loss': [],
        'test_acc': [],
        'active_units': []
    }
    
    global_step = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            x, y = x.view(x.size(0), -1).to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            
            loss = criterion(out, y)
            
            if hasattr(model, 'compute_entropy_loss') or (hasattr(model, 'model') and hasattr(model.model, 'compute_entropy_loss')):
                if hasattr(model, 'compute_entropy_loss'):
                    ent_loss = model.compute_entropy_loss()
                else:
                    ent_loss = model.model.compute_entropy_loss()
                loss += lambda_ent * ent_loss
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # ADC step
            if use_adc and global_step % adc_freq == 0 and global_step > 0:
                if hasattr(model, 'adapt_density'):
                    pruned, split = model.adapt_density(split_thresh=split_thresh, prune_thresh=prune_thresh, optimizer=optimizer)
                    pbar.set_postfix({'loss': loss.item(), 'pruned': pruned, 'split': split})
            
            # Track active units
            if global_step % 100 == 0:
                metrics['train_loss'].append(loss.item())
                if hasattr(model, 'get_num_active'):
                    metrics['active_units'].append((global_step, model.get_num_active()))
                elif hasattr(model, 'model') and hasattr(model.model, 'get_num_active'):
                    metrics['active_units'].append((global_step, model.model.get_num_active()))
                else:
                    metrics['active_units'].append((global_step, 0))
                    
            global_step += 1
            
        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.view(x.size(0), -1).to(device), y.to(device)
                out = model(x)
                _, pred = torch.max(out, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
                
        acc = correct / total
        metrics['test_acc'].append((epoch, acc))
        print(f"Epoch {epoch+1} Test Acc: {acc:.4f}")
        
    return metrics

def train_split_mnist(model, tasks_train, tasks_test, epochs_per_task=2, device='cpu',
                      adc_freq=500, lambda_ent=0.01, use_adc=True,
                      split_thresh=0.05, prune_thresh=0.01):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    metrics = {
        'task_accs': {i: [] for i in range(len(tasks_train))}, # accuracy on each task over time
        'active_units': []
    }
    
    global_step = 0
    
    for task_idx, train_loader in enumerate(tasks_train):
        print(f"--- Training on Task {task_idx} ---")
        
        for epoch in range(epochs_per_task):
            model.train()
            
            pbar = tqdm(train_loader, desc=f"T{task_idx} E{epoch+1}/{epochs_per_task}")
            for x, y in pbar:
                x, y = x.view(x.size(0), -1).to(device), y.to(device)
                
                optimizer.zero_grad()
                out = model(x)
                
                loss = criterion(out, y)
                
                if hasattr(model, 'compute_entropy_loss') or (hasattr(model, 'model') and hasattr(model.model, 'compute_entropy_loss')):
                    if hasattr(model, 'compute_entropy_loss'):
                        ent_loss = model.compute_entropy_loss()
                    else:
                        ent_loss = model.model.compute_entropy_loss()
                    loss += lambda_ent * ent_loss
                    
                loss.backward()
                optimizer.step()
                
                # ADC step
                if use_adc and global_step % adc_freq == 0 and global_step > 0:
                    if hasattr(model, 'adapt_density'):
                        pruned, split = model.adapt_density(split_thresh=split_thresh, prune_thresh=prune_thresh, optimizer=optimizer)
                        pbar.set_postfix({'loss': loss.item(), 'pruned': pruned, 'split': split})
                        
                # Track active units
                if global_step % 100 == 0:
                    if hasattr(model, 'get_num_active'):
                        metrics['active_units'].append((global_step, model.get_num_active()))
                    elif hasattr(model, 'model') and hasattr(model.model, 'get_num_active'):
                        metrics['active_units'].append((global_step, model.model.get_num_active()))
                    else:
                        metrics['active_units'].append((global_step, 0))
                        
                global_step += 1
                
        # Eval on ALL tasks seen so far (and future ones, why not)
        model.eval()
        for t_idx, test_loader in enumerate(tasks_test):
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.view(x.size(0), -1).to(device), y.to(device)
                    out = model(x)
                    _, pred = torch.max(out, 1)
                    total += y.size(0)
                    correct += (pred == y).sum().item()
            acc = correct / total
            metrics['task_accs'][t_idx].append((task_idx, acc)) # (after_task, accuracy)
            print(f"Test Acc on Task {t_idx}: {acc:.4f}")
            
    return metrics
