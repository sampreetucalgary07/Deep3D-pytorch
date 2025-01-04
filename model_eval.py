from preprocess import checknumpyImage
from evaluation import randomLRImage
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from Deep3D_model import Net


################ 4. Model Training ###############


def showRandomTest(
    testL, testR, testR1=None, model=None, random_index=None, half=False
):

    if random_index == 0:
        raise (ValueError("Random index should be greater than 0"))

    if model is None:
        raise (ValueError("Model not found"))

    if random_index is None:
        random_index = np.random.randint(2, len(testL))

    l_test, r_test, r1_test, pred_test, _ = randomLRImage(
        half, testL, testR, testR1, model, random_index
    )

    print("Random index: ", random_index)
    print("L0 shape: ", l_test.shape)

    left_image = checknumpyImage(l_test[0], False)
    right_image = checknumpyImage(r_test[0], False)
    pred_image = checknumpyImage(pred_test[0], True)
    prev_right_image = checknumpyImage(r1_test[0], False)

    print("Left Image shape: ", left_image.shape)
    plt.figure(figsize=(24, 12))
    plt.subplot(1, 4, 1)
    plt.title("Left Image (test) : " + str(_))
    plt.imshow(left_image)
    plt.subplot(1, 4, 2)
    plt.title("Right Image (test)")
    plt.imshow(right_image)
    plt.subplot(1, 4, 3)
    plt.title("Previous Right Image (test)")
    plt.imshow(prev_right_image)
    plt.subplot(1, 4, 4)
    plt.title("Predicted Right Image")
    plt.imshow(pred_image)
    plt.show()


def train_one_epoch(loss_fn, optimizer, model, train_L0, train_R0, train_R1=None):
    loss_train = 0
    model = model.cuda()
    for l, r in zip(tqdm(train_L0, total=len(train_L0)), train_R0):
        r = r.cuda()
        optimizer.zero_grad()
        pred_right = model(l.cuda())
        loss = loss_fn(pred_right, r)
        loss_train += loss.item()
        loss.backward()
        optimizer.step()
        del loss
        del pred_right

    return loss_train
