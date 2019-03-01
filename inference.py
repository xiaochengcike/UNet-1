import time

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from Network import Unet
from dataop import MySet, create_list
from utils import make_one_hot, dice_coeff


if __name__ == '__main__':
    train_path = '../data-blood/Train/'
    check_point_path = 'Checkpoint/u-net.pth.gz'
    compute_times = 50

    torch.cuda.set_device(1)

    _, infer_list = create_list(train_path, ratio=0.9)
    infer_set = MySet(infer_list)
    infer_loader = DataLoader(infer_set, batch_size=1, shuffle=True)

    net = Unet.UNet().cuda()
    net.eval()
    net.load_state_dict(torch.load(check_point_path))

    for data, label in infer_loader:
        data = Variable(data.cuda())
        output = net(data)
        print(data.size())

        output_arr = output.squeeze().data.cpu().numpy()
        label_arr = label.squeeze().cpu().numpy()

        label_arr = make_one_hot(label_arr, num_class=3)
        test_f1_background = dice_coeff(output_arr[0, :, :, :], label_arr[0, :, :, :])
        test_f1_a = dice_coeff(output_arr[1, :, :, :], label_arr[1, :, :, :])
        test_f1_b = dice_coeff(output_arr[2, :, :, :], label_arr[2, :, :, :])

        start_time = time.time()
        for _ in range(compute_times):
            _ = net(data)
        iteration_time = (time.time() - start_time)/compute_times
        info_line = "F1 of background : {} F1 of class a: {} F1 of class B: {} with computation time {}".format(
            test_f1_background, test_f1_a, test_f1_b, iteration_time
        )

        print(info_line)





