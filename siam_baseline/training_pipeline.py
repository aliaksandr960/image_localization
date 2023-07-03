import csv
import torch
import torch.nn.functional as F
from tqdm import tqdm


EPOCH_KEY = 'EPOCH'
BATCH_KEY = 'BATCH'
LOSS_KEY = 'LOSS'
ACC_KEY = 'ACC'
VAL_LOSS_KEY = 'VAL_LOSS'
VAL_ACC_KEY = 'VAL_ACC'


def run_trining(net, criterion, optimizer,
                epoch_count, epoch_max_batch,
                train_loader, valid_loader,
                train_history_path, valid_history_path, log_path,
                device):
    
    # Evaluate batch
    def shared_step(batch, train=True):
        # get the inputs; data is a list of [inputs, labels]
        batch = [i.to(device).permute(0, 3, 1, 2).float() for i in batch]
        anchor, positive, negative = batch

        # zero the parameter gradients
        if train: 
            optimizer.zero_grad()

        # forward + backward + optimize
        anchor_outputs = net(anchor)
        positive_outputs = net(positive)
        negative_outputs = net(negative)
        
        anchor_to_positive = F.pairwise_distance(anchor_outputs, positive_outputs)
        anchor_to_negative = F.pairwise_distance(anchor_outputs, negative_outputs)
        
        correct_items = torch.sum(anchor_to_positive < anchor_to_negative)
        accuracy = correct_items / anchor_to_positive.shape[0]
        
        loss = criterion(anchor_outputs, positive_outputs, negative_outputs)
        if train:
            loss.backward()
            optimizer.step()
        return loss, accuracy


    train_history = []
    valid_history = []
    net.train()
    for epoch in range(1, epoch_count+1):
        # Training
        total_train_loss = 0
        with tqdm(train_loader, position=0) as train_data_iterator:
            for batch_n, data in enumerate(train_data_iterator, 1):
                if epoch_max_batch < batch_n:
                    break
                
                loss, acc = shared_step(data)
                total_train_loss += float(loss)

                # print statistics
                history_item = {EPOCH_KEY: epoch,
                                BATCH_KEY: batch_n,
                                LOSS_KEY: float(loss),
                                ACC_KEY: float(acc)}
                train_history.append(history_item)
                train_data_iterator.set_postfix(history_item)
                
        # Validation
        total_valid_loss = 0
        total_valid_acc = 0
        net.eval()
        with tqdm(valid_loader, position=0) as valid_data_iterator:
            for batch_n, data in enumerate(valid_data_iterator, 1):
                with torch.no_grad():
                    loss, acc = shared_step(data, train=False)
                total_valid_loss += float(loss)
                total_valid_acc += float(acc)

        avg_train_loss = total_train_loss / len(train_loader)
        avg_valid_loss = total_valid_loss / len(valid_loader)
        avg_valid_acc = total_valid_acc / len(valid_loader)

        # print statistics
        history_item = {EPOCH_KEY: epoch,
                        BATCH_KEY: batch_n,
                        LOSS_KEY: avg_train_loss,
                        VAL_LOSS_KEY: avg_valid_loss,
                        VAL_ACC_KEY: avg_valid_acc}
        valid_history.append(history_item)
        valid_data_iterator.set_postfix(history_item)

        print(f'Epoch {epoch} finished, avg training loss {avg_train_loss}, avg valid loss {avg_valid_loss}, avg valid acc {avg_valid_acc}')
        torch.save(net.state_dict(), f'{log_path}/{str(epoch)}.pth')

        fields = [i for i in train_history[0].keys()]
        with open(train_history_path, 'w', newline='') as f: 
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader() 
            writer.writerows(train_history)
            
        fields = [i for i in valid_history[0].keys()]
        with open(valid_history_path, 'w', newline='') as f: 
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader() 
            writer.writerows(valid_history)
    
    return train_history, valid_history
