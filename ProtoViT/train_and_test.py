# no synchronize & time.perf_counter()
# profiler scheduler

import time
import torch
from tqdm import tqdm
from helpers import list_of_distances, make_one_hot
import torch.nn.functional as F


from settings import base_architecture, img_size, prototype_shape, num_classes, \
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
    repeat=2       # Repeat the cycle 2 times
)

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print, ema = None, clst_k = 1,sum_cls = True, epoch=-1):
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
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    total_orth_loss = 0 
    total_comp_loss = 0 
    total_loss = 0 
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
                    input = image.cuda()
                    target = label.cuda()
                    print(f'---------\nimage {i}')

                    # torch.enable_grad() has no effect outside of no_grad()
                    grad_req = torch.enable_grad() if is_train else torch.no_grad()
                    with grad_req:
                        # nn.Module has implemented __call__() function
                        # so no need to call .forward
                        start_time = time.perf_counter()
                        with record_function("Step_model_forward"):
                            output, min_distances, values = model(input)
                        print(f'whole model forward pass: {time.perf_counter() - start_time}')
                        del input

                        # compute loss
                        start_time = time.perf_counter()
                        with record_function("Step_cross_entropy"):
                            cross_entropy = torch.nn.functional.cross_entropy(output, target)
                        torch.cuda.synchronize()
                        print(f'cross_entropy: {time.perf_counter() - start_time}')

                        if class_specific:    
                            print('class_specific')
                            start_time = time.perf_counter()
                            with record_function("Step_cluster_cost"):
                                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                                # calculate cluster cost
                                # to encourage the prototype to focus on foreground, we calcualted a weighted cluster loss
                                # max_dist = (model.prototype_shape[1]
                                #             * model.prototype_shape[2])
                                #             #* model.prototype_shape[3]) # dim*1*1 
                                prototypes_of_correct_class = torch.t(model.prototype_class_identity[:,label]).cuda()
                                prototypes_of_correct_class = prototypes_of_correct_class.unsqueeze(-1)
                                max_activations = -min_distances

                                ### retrieve slots 
                                # a soft approximation of 1, 0 s 
                                slots = torch.sigmoid(model.patch_select*model.temp)# 2000, 1, 4
                                #factor = ((slots.sum(-1))**0.5).unsqueeze(-1) # 2000, 1, 1
                                if clst_k == 1:
                                    if not sum_cls: 
                                        correct_class_prototype_activations =  values * prototypes_of_correct_class # bsz, 2000, 4
                                        correct_class_proto_act_max_sub_patch, _ = torch.max(correct_class_prototype_activations, dim = 2) # bsz, 2000
                                        correct_class_prototype_activations, _ = torch.max(correct_class_proto_act_max_sub_patch, dim=1) # bsz 
                                    else:
                                        correct_class_prototype_activations = (values.sum(-1)) * prototypes_of_correct_class.squeeze(-1) # bsz, 2000, 1
                                        correct_class_prototype_activations, _ = torch.max(correct_class_prototype_activations, dim=1) 
                                        
                                    cluster_cost = torch.mean(correct_class_prototype_activations)
                                else:
                                    # clst_k is a hyperparameter that lets the cluster cost apply in a "top-k" fashion:
                                    #the original cluster cost is equivalent to the k = 1 case
                                    correct_class_prototype_activations =  values * prototypes_of_correct_class # bsz, 2000, 4
                                    correct_class_proto_act_max_sub_patch, _ = torch.max(correct_class_prototype_activations, dim = 2) # bsz, 2000
                                    top_k_correct_class_prototype_activations, _ = torch.topk(correct_class_proto_act_max_sub_patch,
                                                                                            k = clst_k, dim=1)
                                    cluster_cost = torch.mean(top_k_correct_class_prototype_activations)
                            cluster_cost_elapsed_time = time.perf_counter() - start_time
                            torch.cuda.synchronize()
                            print(f'cluster_cost: {cluster_cost_elapsed_time}')
                            total_cluster_time += cluster_cost_elapsed_time

                            # calculate separation cost
                            start_time = time.perf_counter()
                            with record_function("Step_separation_cost"):
                                prototypes_of_wrong_class = (1 - prototypes_of_correct_class.squeeze(-1)).unsqueeze(-1)
                                # inverted_distances_to_nontarget_prototypes, _ = \
                                #     torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                                # separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)
                                
                                if not sum_cls:
                                    incorrect_class_prototype_activations_sub, _ = torch.max(values * prototypes_of_wrong_class, dim=2)# bsz, 2000
                                    incorrect_class_prototype_activations, _ = torch.max(incorrect_class_prototype_activations_sub, dim=1) # bsz
                                else:
                                    #values_slot = (values.clone())*slots
                                    incorrect_class_prototype_activations = (values.sum(-1)) * prototypes_of_wrong_class.squeeze(-1)
                                    incorrect_class_prototype_activations, _ = torch.max(incorrect_class_prototype_activations, dim=1) 
                                separation_cost = torch.mean(incorrect_class_prototype_activations)
                            torch.cuda.synchronize()
                            print(f'separation_cost: {time.perf_counter() - start_time}')

                            # calculate avg cluster cost
                            # start_time = time.perf_counter()
                            # with record_function("Step_avg_separation_cost"):
                            start_time = time.perf_counter()
                            avg_separation_cost = \
                                torch.sum(values * prototypes_of_wrong_class, dim=1) / (values.shape[-1]*torch.sum(prototypes_of_wrong_class, dim=1))
                            avg_separation_cost = torch.mean(avg_separation_cost)
                            torch.cuda.synchronize()
                            print(f'avg_separation_cost: {time.perf_counter() - start_time}')

                            start_time = time.perf_counter()
                            with record_function("Step_orth_cost"):
                                #optimize orthogonality of prototype_vector, borrowed from tesnet 
                                # ortho loss version 1 
                                #factor =  (model.prototype_shape[-1])**0.5
                                prototype_normalized = F.normalize(model.prototype_vectors,p=2,dim=1)#/factor
                                cur_basis_matrix = torch.squeeze(prototype_normalized)#*slots #[2000,dim, 4]
                                #cur_basis_matrix = cur_basis_matrix.mean(-1).mean(-1) # [2000, dim]
                                subspace_basis_matrix = cur_basis_matrix.reshape(model.num_classes,model.num_prototypes_per_class,-1)#[200,10,dim*4]
                                subspace_basis_matrix_T = torch.transpose(subspace_basis_matrix,1,2) #[200,10,dim*4]->[200,4*dim,10]
                                orth_operator = torch.matmul(subspace_basis_matrix,subspace_basis_matrix_T)  # [200,10,dim] [200,dim,10] -> [200,10,10]
                                I_operator = torch.eye(subspace_basis_matrix.size(1),subspace_basis_matrix.size(1)).cuda() #[10,10]
                                difference_value = orth_operator - I_operator #[200,10,10]-[10,10]->[200,10,10]
                                orth_cost = torch.sum(torch.relu(torch.norm(difference_value,p=1,dim=[1,2]) - 0)) #[200]->[1]
                            torch.cuda.synchronize()
                            print(f'orth_cost: {time.perf_counter() - start_time}')
                            
                            ### component distance loss 
                            """
                            Penalize the max cosine distance between the 1, 2, 3, 4 ... patches to the each the other patch
                            """
                            # start_time = time.perf_counter()
                            # with record_function("Step_component_dist_loss"):
                            # proto_norm_k = F.normalize(model.prototype_vectors,p=2, dim=1)# each of the prototype patch has norm 1
                            # #dist_all = torch.zeros((1000)).cuda()
                            # dist_init = 1- F.cosine_similarity(proto_norm_k[:, :,0], proto_norm_k[:, :, 0])
                            # for j in range((model.prototype_shape[-1])):
                            #     for k in range((model.prototype_shape[-1])):
                            #         dist_init += 1 - F.cosine_similarity(proto_norm_k[:, :, k], proto_norm_k[:, :, j])
                            # avg_diff = dist_init.sum()

                            # proto_norm_k = F.normalize(model.prototype_vectors,p=2, dim=1)# each of the prototype patch has norm 1
                            # dist_init = 1- F.cosine_similarity(proto_norm_k[:, :,0], proto_norm_k[:, :, 0])
                            # dist_jk = torch.tensor([]).cuda()#torch.empty((proto_norm_k.shape[0], proto_norm_k.shape[-1]))
                            # for j in range((model.prototype_shape[-1])):
                            #     dist_jk = torch.tensor([]).cuda()
                            #     for k in range((model.prototype_shape[-1])):
                            #         cos_jk = 1 - F.cosine_similarity(proto_norm_k[:, :, k], proto_norm_k[:, :, j])
                            #         dist_jk = torch.concat((dist_jk, cos_jk.unsqueeze(-1)), dim = -1)
                            #     #dist_jk_slots = dist_jk*slots
                            #     dist_jk_max, _ = dist_jk.max(dim=-1)
                            #     dist_init += dist_jk_max
                            # avg_diff = dist_init.sum()

                            proto_norm_k = F.normalize(model.prototype_vectors,p=2, dim=1)# each of the prototype patch has norm 1
                            dist_jk = 1- F.cosine_similarity(proto_norm_k[:, :,0], proto_norm_k[:, :, 0])
                            dist_init = torch.tensor([]).cuda()#torch.empty((proto_norm_k.shape[0], proto_norm_k.shape[-1]))
                            for j in range((model.prototype_shape[-1])):
                                dist_jk = 1- F.cosine_similarity(proto_norm_k[:, :,0], proto_norm_k[:, :, 0])
                                for k in range((model.prototype_shape[-1])):
                                    cos_jk = 1 - F.cosine_similarity(proto_norm_k[:, :, k], proto_norm_k[:, :, j])
                                    dist_jk += cos_jk#torch.concat((dist_jk, cos_jk.unsqueeze(-1)), dim = -1)
                                #dist_jk_max, _ = dist_jk.max(dim=-1)
                                dist_init = torch.concat((dist_init, cos_jk.unsqueeze(-1)), dim = -1)
                            # find the prototype patch that is most disimilar to the others 
                            dist_init_slots = dist_init*slots
                            most_disimilar, _ = dist_init_slots.max(-1)
                            avg_diff = most_disimilar.sum()
                            # print(f'component distance loss avg_diff: {time.perf_counter() - start_time}')

                            # l2 norm of slots to encourage sparsity 
                            # start_time = time.perf_counter()
                            # with record_function("Step_l1"):
                            if use_l1_mask:
                                l1_mask = 1 - torch.t(model.prototype_class_identity).cuda()
                                l1 = (model.last_layer.weight * l1_mask).norm(p=1)
                            else:
                                l1 = model.last_layer.weight.norm(p=1) 
                            # print(f'l1: {time.perf_counter() - start_time}')


                        else:
                            min_distance, _ = torch.min(min_distances, dim=1)
                            cluster_cost = torch.mean(min_distance)
                            l1 = model.last_layer.weight.norm(p=1)
                        del min_distances
                        
                        # evaluation statistics
                        # with record_function("Step_statistics"):
                        _, predicted = torch.max(output.data, 1)
                        n_examples += target.size(0)
                        n_correct += (predicted == target).sum().item()
                        n_batches += 1
                        total_cross_entropy += cross_entropy.item()
                        total_cluster_cost += cluster_cost.item()
                        total_separation_cost += separation_cost.item()
                        total_avg_separation_cost += avg_separation_cost.item()
                        total_orth_loss += orth_cost.item()
                        total_comp_loss += avg_diff.item()
                        avg_number_patch = (slots >= 0.5).sum()/slots.shape[1]
                        avg_slots = slots.squeeze(0).sum(1)/slots.shape[1]
                        # print(f'statistics: {time.perf_counter() - start_time}')
                        torch.cuda.synchronize()
                        del target
                        del output
                        del predicted

                    # compute gradient and do SGD step
                    if is_train:
                        start_time = time.perf_counter()
                        with record_function("Step_backward_pass"):
                            if class_specific:
                                if coefs is not None:
                                    loss = (coefs['crs_ent'] * cross_entropy
                                        + coefs['clst'] * cluster_cost
                                        + coefs['sep'] * separation_cost
                                        + coefs['l1'] * l1
                                        +coefs['orth']*orth_cost
                                        +coefs['coh']*avg_diff
                                        )
                                        #+coefs['slots']*slots_loss)
                                    total_loss += loss.item()
                                else:
                                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
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
                            if ema is not None:
                                ema.update(model)
                        torch.cuda.synchronize()
                        print(f'backward_pass: {time.perf_counter() - start_time}')

                    #del weighted_min_distance
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
    
    
    log('\ttime: \t{0}'.format(end -  start_og))
    #log('\tlearning rate info: \t{0}'.format(optimizer))
    log('\ttotal loss: \t{0}'.format(total_loss / n_batches))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\torthogonal loss\t{0}'.format(total_orth_loss/n_batches))
    log('\ttotal_cluster_time: \t{0}'.format(total_cluster_time))
    log('\tavg_cluster_time: \t{0}'.format(total_cluster_time / n_batches))

    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    log('\tslot of prototype 0: \t{0}'.format(slots.squeeze()[0]))
    log('\tEstimated avg number of subpatches: \t{0}'.format(avg_number_patch))
    log('\tEstimated avg slots logit: \t{0}'.format(avg_slots))
    
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\tcoherence loss: \t\t{0}%'.format(total_comp_loss / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(model.last_layer.weight.norm(p=1).item()))
    p = model.prototype_vectors.view(model.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))
    loss_values = {
    "cross entropy Loss": total_cross_entropy / n_batches,
    "clst loss":  total_cluster_cost / n_batches,
    'sep loss': total_separation_cost / n_batches,
    'avg separation_cost':total_avg_separation_cost / n_batches,
    'l1 loss': model.last_layer.weight.norm(p=1).item(),
    'orth loss':total_orth_loss/n_batches,
    'acc':n_correct / n_examples * 100}
    return (n_correct / n_examples), loss_values


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print, ema = None, clst_k = 1,sum_cls = True, epoch=-1):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log, ema = ema, clst_k =clst_k,sum_cls = sum_cls, epoch=epoch)


def test(model, dataloader, class_specific=False, log=print, ema = None, clst_k = 1, sum_cls = True, epoch=-1):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log, ema = ema, clst_k = clst_k,sum_cls = sum_cls, epoch=epoch)


def last_only(model, log=print):
    for p in model.features.parameters():
        p.requires_grad = False
    model.prototype_vectors.requires_grad = False
    for p in model.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')

# def slots_only(model, log=print):
#     for p in model.features.parameters():
#         p.requires_grad = False


def warm_only(model, log=print):
    for p in model.features.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')


def joint(model, log=print):
    for p in model.features.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True
    log('\tjoint')
