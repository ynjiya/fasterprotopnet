# no synchronize & time.perf_counter()
# profiler scheduler

import time
import torch

from helpers import list_of_distances, make_one_hot


from settings_custom import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run

model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'

from torch.profiler import profile, record_function, ProfilerActivity, schedule
from torch.autograd.profiler_util import EventList
import csv
import os

prof_schedule = schedule(
    skip_first=10,  # Skip the first 5 iterations
    wait=1,        # Wait for 1 iteration after skipping
    warmup=5,      # Warmup for 5 iteration
    active=50,     # Actively profile for 10 iterations
    repeat=3       # Repeat the cycle 2 times
)
torch.backends.cudnn.benchmark = True  # Speeds up convolutions


def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=False,
                   coefs=None, log=print, epoch=-1):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start_og = time.perf_counter()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    total_cluster_time = 0
    total_separation_cost = 0
    total_avg_separation_cost = 0
    
    print(f'Epoch {epoch}')
    with profile(
        activities=[
            ProfilerActivity.CUDA,
            ProfilerActivity.CPU
        ],
        schedule=prof_schedule
    ) as prof:
        with record_function("Step_total_per_train_or_test"):
            for i, (image, label) in enumerate(dataloader):
                with record_function("Step_total_per_image"):

                    start_per_image = time.perf_counter()
                    input = image.cuda(non_blocking=True)
                    target = label.cuda(non_blocking=True)
                    print(f'---------\nimage {i}')

                    # torch.enable_grad() has no effect outside of no_grad()
                    grad_req = torch.enable_grad() if is_train else torch.no_grad()
                    with grad_req:
                        # Forward pass
                        start_time = time.perf_counter()
                        with record_function("Step_model_forward"):
                            output, min_distances = model(input)
                        torch.cuda.synchronize()
                        print(f'whole model forward pass: {time.perf_counter() - start_time}')

                        # Loss computation
                        start_time = time.perf_counter()
                        with record_function("Step_cross_entropy"):
                            cross_entropy = torch.nn.functional.cross_entropy(output, target)
                        torch.cuda.synchronize()
                        print(f'cross_entropy: {time.perf_counter() - start_time}')

                        if class_specific:
                            # Cluster cost (Simplified)
                            print('class_specific')
                            start_time = time.perf_counter()
                            with record_function("Step_cluster_cost"):
                                cluster_cost = torch.mean(min_distances)
                            cluster_cost_elapsed_time = time.perf_counter() - start_time
                            torch.cuda.synchronize()
                            print(f'cluster_cost: {cluster_cost_elapsed_time}')
                            total_cluster_time += cluster_cost_elapsed_time


                            start_time = time.perf_counter()
                            with record_function("Step_l1"):
                                if use_l1_mask:
                                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                                else:
                                    l1 = model.module.last_layer.weight.norm(p=1)
                            torch.cuda.synchronize()
                            print(f'l1: {time.perf_counter() - start_time}')
                        else:
                            min_distance, _ = torch.min(min_distances, dim=1)
                            cluster_cost = torch.mean(min_distance)
                            l1 = model.module.last_layer.weight.norm(p=1)

                        # Training step
                        if is_train:
                            start_time = time.perf_counter()
                            with record_function("Step_backward_pass"):
                                if class_specific:
                                    if coefs is not None:
                                        loss = (coefs['crs_ent'] * cross_entropy
                                            + coefs['clst'] * cluster_cost
                                            # + coefs['sep'] * separation_cost
                                            + coefs['l1'] * l1)
                                    else:
                                        loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
                                else:
                                    if coefs is not None:
                                        loss = (coefs['crs_ent'] * cross_entropy
                                            + coefs['clst'] * cluster_cost
                                            + coefs['l1'] * l1)
                                    else:
                                        loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                            torch.cuda.synchronize()
                            print(f'backward_pass: {time.perf_counter() - start_time}')

                    # Statistics
                    start_time = time.perf_counter()
                    with record_function("Step_statistics"):
                        # _, predicted = torch.max(output.data, 1)
                        _, predicted = torch.max(output.detach(), 1)
                        n_examples += target.size(0)
                        # n_correct += (predicted == target).sum().item()
                        n_correct += (predicted == target).sum()  # Keep as tensor (avoid .item() per batch)
                        # n_batches += 1
                        # total_cross_entropy += cross_entropy.item()
                        # total_cluster_cost += cluster_cost.item()
                        # total_separation_cost += separation_cost.item()
                        # total_avg_separation_cost += avg_separation_cost.item()
                    torch.cuda.synchronize()
                    print(f'statistics: {time.perf_counter() - start_time}')
                    

                    # Cleanup
                    start_time = time.perf_counter()
                    del input, target, output, predicted, min_distances
                    print(f'- cleanup: {time.perf_counter() - start_time}')
                torch.cuda.synchronize()
                print(f'~~~ total per image: {time.perf_counter() - start_per_image}')
                prof.step()  # Notify the profiler at the end of each iteration
        end = time.perf_counter()

    # Final profiling results
    print("\nFinal Profiling Results:")
    print('len(prof.key_averages())', len(prof.key_averages()))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    # Export profiler data to a file (optional)
    # prof.export_chrome_trace("trace.json")
    print('Done printing Final Profiling Results')
    
    # Extract events that start with 'Step_'
    custom_events = EventList([evt for evt in prof.key_averages() if evt.key.startswith('Step_')])
    
    # Clear profiler data to free memory
    del prof
    # torch.cuda.empty_cache()  # Clear GPU memory
    
    # Select a device (if you have more than one GPU, specify the index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check memory usage before emptying the cache
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    print(f"Before empty_cache(): Allocated = {allocated_before/1024**2:.2f} MB, Reserved = {reserved_before/1024**2:.2f} MB")

    # Clear the cache
    torch.cuda.empty_cache()  # Clear GPU memory

    # Check memory usage after emptying the cache
    allocated_after = torch.cuda.memory_allocated(device)
    reserved_after = torch.cuda.memory_reserved(device)
    print(f"After empty_cache(): Allocated = {allocated_after/1024**2:.2f} MB, Reserved = {reserved_after/1024**2:.2f} MB")

    print('len custom_events', len(custom_events))
    # Process each custom event
    for evt in custom_events:
        csv_file = os.path.join(model_dir, f"{evt.key}.csv")
        print(csv_file)
        
        # Create the CSV file if it doesn't exist
        if not os.path.exists(csv_file):
            with open(csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "epoch",
                    "is_train",
                    "cpu_time_str",
                    "cpu_time_total_str",
                    "device_time_str",
                    "device_time_total_str",
                    "self_cpu_time_total_str",
                    "self_device_time_total_str"
                ])
        
        # Append the event data to the CSV file
        row = [
            str(epoch),
            str(is_train),
            evt.cpu_time_str,
            evt.cpu_time_total_str,
            evt.device_time_str,
            evt.device_time_total_str,
            evt.self_cpu_time_total_str,
            evt.self_device_time_total_str
        ]
        print(evt.key, row)
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)
            
    del custom_events       
    torch.cuda.empty_cache()  # Clear GPU memory
    n_correct = n_correct.cpu().item()

    
    log('\ttime: \t{0}'.format(end -  start_og))
    log('\ttotal_cluster_time: \t{0}'.format(total_cluster_time))
    # log('\tavg_cluster_time: \t{0}'.format(total_cluster_time / n_batches))
    # log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    # log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    # if class_specific:
        # log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        # log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    return n_correct / n_examples


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print, epoch=-1):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log, epoch=epoch)


def test(model, dataloader, class_specific=False, log=print, epoch=-1):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log, epoch=epoch)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')
