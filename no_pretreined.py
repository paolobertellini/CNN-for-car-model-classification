
from torch.utils.data import DataLoader

import plots
from dataset import CarDataset
from model import Small
from testing import test
from training import train

def no_pretreined(args, device):

    # -- CONV NET -- #
    net = CNN().to(device)

    # statistics
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    true_list = []
    predict_list = []

    for epoch in range(args.epochs):

        # taining
        train_loss, train_acc = train(trainloader, net, args.batch_size, args.learning_rate, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        #test
        test_loss, test_acc, true, predict = test(args.batch_size, testloader, net, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        true_list += true
        predict_list += predict

        print(f"EPOCH {epoch}: [TRAINING loss: {train_loss:.5f} acc: {train_acc:.2f}%]",
                             f"[TESTING loss: {test_loss:.5f} acc: {test_acc:.2f} %]")


    print(f"TRAINING AND TESTING FINISHED")
    print(f"TRAIN LOSS HISTORY: {train_losses}")
    print(f"TRAIN ACCURACY HISTORY: {train_accs}")
    print(f"TEST LOSS HISTORY: {test_losses}")
    print(f"TEST ACCURACY HISTORY: {test_accs}")
    plots.printPlots(classes, args.dataset_dir, args.epochs, train_losses, train_accs, test_losses, test_accs, predict_list, true_list)