base_architecture = 'resnet34'
img_size = 224
prototype_shape = (2000, 64, 1, 1)
num_classes = 200
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '120_150_110-custom-step-2'

# data_path =  ""
data_path = '../BetterProtoPNet/datasets/cub200_cropped/'
train_dir = data_path + 'train_cropped_augmented/'
test_dir = data_path + 'test_cropped/'
train_push_dir = data_path + 'train_cropped/'

train_batch_size =120
test_batch_size = 150
train_push_batch_size = 110

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
    'orth': 1e-4,
    'sub_sep': -1e-7,
}

num_train_epochs = 15
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]

#print(push_epochs)
