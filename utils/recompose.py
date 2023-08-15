import numpy as np
from PIL import Image

# split the full_imgs to pacthes
def extract_ordered(full_imgs, full_gt, patch_h, patch_w, channel):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == channel)
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    N_patches_h = int(img_h / patch_h)  # round to lowest int

    if img_h % patch_h != 0:
        print("warning: {} patches in height, with about {} pixels left over"
              .format(N_patches_h, img_h % patch_h))

    N_patches_w = int(img_w / patch_w)  # round to lowest int

    if img_h % patch_h != 0:
        print("warning: {} patches in width, with about {} pixels left over"
              .format(N_patches_w, img_w % patch_w))
    print("number of patches per image: ", N_patches_h * N_patches_w)
    N_patches_tot = (N_patches_h * N_patches_w) * full_imgs.shape[0]
    patches = np.empty((N_patches_tot, full_imgs.shape[1], patch_h, patch_w), dtype=np.float16)
    patches_gt = np.empty((N_patches_tot, full_gt.shape[1], patch_h, patch_w), dtype=np.uint8)
    iter_tot = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        print(i)
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                patch = full_imgs[i, :, h * patch_h:(h * patch_h) + patch_h, w * patch_w:(w * patch_w) + patch_w]
                patch_gt = full_gt[i, :, h * patch_h:(h * patch_h) + patch_h, w * patch_w:(w * patch_w) + patch_w]
                patches[iter_tot] = patch
                patches_gt[iter_tot] = patch_gt
                iter_tot += 1  # total
    assert (iter_tot == N_patches_tot)
    return patches, patches_gt  # array with all the full_imgs divided in patches
def extract_ordered_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w, channel):
    img_h = full_imgs.shape[1]  # height of the full image
    img_w = full_imgs.shape[2]  # width of the full image
    assert ((img_h - patch_h) % stride_h == 0 and (img_w - patch_w) % stride_w == 0)

    num_patches_one = ((img_h - patch_h) // stride_h + 1) \
                      * ((img_w - patch_w) // stride_w + 1)

    num_patches_total = num_patches_one * full_imgs.shape[0]
    print("Number of patches on h : ", (img_h - patch_h) // stride_h + 1)
    print("Number of patches on w : ", (img_w - patch_w) // stride_w + 1)
    print("number of patches per image: {}, totally for this testing dataset: {}"
          .format(num_patches_one, num_patches_total))
    patches = np.empty((num_patches_total, full_imgs.shape[3], patch_h, patch_w), dtype=np.float16)
    iter_total = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                patch = full_imgs[i, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w,:]
                patch = np.transpose(patch,(2,0,1))
                patches[iter_total] = patch
                iter_total += 1  # total
    assert (iter_total == num_patches_total)
    return patches  # array with all the full_imgs divided in patches
def get_data_testing_overlap(X_test,n_test_images, patch_height, patch_width,
                             stride_height, stride_width, channel):
    """
    :param test_images_file: the filename of hdf5 test_images_file
    :param test_gt_file: the filename of hdf5 test_gt_file
    :param n_test_images: the num of test image
    :param patch_height: the height of each patch
    :param patch_width: the width of each width
    :param stride_height: the stride of height
    :param stride_width: the stride of width
    :return:
    """

    test_images = X_test
    # preproceing the test images
    # extend both images and masks so they can be divided exactly by the patches dimensions
    test_images = test_images[0:n_test_images, :, :, :]
    test_images = paint_border_overlap(test_images, patch_height, patch_width,
                                       stride_height, stride_width, channel)

    print("extended test images shape:", test_images.shape)
    print("print sample data:", test_images[0, 0, 0:100, 0:100])

    print("test images range (min-max): {}-{} ".format(np.min(test_images), np.max(test_images)))

    # extract the TEST patches from the full images
    patches_imgs_test = extract_ordered_overlap(test_images, patch_height, patch_width,
                                                stride_height, stride_width, channel)

    print("test PATCHES images shape:", patches_imgs_test.shape)
    print("test PATCHES images range (min-max):{} - {}"
          .format(np.min(patches_imgs_test), np.max(patches_imgs_test)))

    return patches_imgs_test, test_images.shape[1], test_images.shape[2]
def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w, channel):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[3] == channel)  # check the channel is 1 or 3
    img_h = full_imgs.shape[1]  # height of the full image
    img_w = full_imgs.shape[2]  # width of the full image
    leftover_h = (img_h - patch_h) % stride_h  # leftover on the h dim
    leftover_w = (img_w - patch_w) % stride_w  # leftover on the w dim

    # extend dimension of img h by adding zeros
    if leftover_h != 0:
        print("the side H is not compatible with the selected stride of {}".format(stride_h))
        print("img_h: {}, patch_h: {}, stride_h: {}".format(img_h, patch_h, stride_h))
        print("(img_h - patch_h) MOD stride_h: ", leftover_h)
        print("So the H dim will be padded with additional {} pixels ".format(stride_h - leftover_h))
        tmp_full_imgs = np.zeros((full_imgs.shape[0], full_imgs.shape[1], img_h + (stride_h - leftover_h), img_w))
        tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:img_h, 0:img_w] = full_imgs
        full_imgs = tmp_full_imgs

    # extend dimension of img w by adding zeros
    if leftover_w != 0:  # change dimension of img_w
        print("the side W is not compatible with the selected stride of {}".format(stride_w))
        print("img_w: {}, patch_w: {}, stride_w: {}".format(img_w, patch_w, stride_w))
        print("(img_w - patch_w) MOD stride_w: ", leftover_w)
        print("So the W dim will be padded with additional {} pixels ".format(stride_w - leftover_w))
        tmp_full_imgs = np.zeros(
            (full_imgs.shape[0], full_imgs.shape[1], full_imgs.shape[2], img_w + (stride_w - leftover_w)))
        tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:full_imgs.shape[2], 0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    print("new full images shape:", full_imgs.shape)
    return full_imgs


def get_loss_weight(patch_height, patch_width, mode, stride_height=8,
                    stride_width=8, batch_size=4, border = 126):
    loss_weight = np.zeros((patch_height, patch_width))
    center_x = patch_height /2 - 1
    center_y = patch_width / 2 - 1
    if mode == 0:
        return None

    for k in range(patch_height//2):
        for i in range(k, patch_width - k):
            loss_weight[k, i] = k
            loss_weight[i, k] = k
            loss_weight[patch_height - k - 1, i] = k
            loss_weight[i, patch_width - k - 1] = k
    max_value = np.max(loss_weight)
    max_value = float(max_value)
    if mode == 4:
        # in this mode, loss weight outside is 0, inner is 1
        loss_weight[np.where(loss_weight < border)] = 0
        loss_weight[np.where(loss_weight >= border)] = 1
        loss_weight = np.reshape(loss_weight, (patch_width * patch_height))
    else:
        if mode == 1:
            loss_weight = loss_weight/max_value * loss_weight/max_value
        elif mode == 2:
            loss_weight = loss_weight/max_value
        elif mode == 3:
            loss_weight = np.sqrt(loss_weight/max_value)

        loss_weight = np.reshape(loss_weight, (patch_width * patch_height))
        weight_sum = patch_height * patch_width
        cur_sum = np.sum(loss_weight)
        loss_weight *= weight_sum/cur_sum
    #     loss_weight = np.reshape(loss_weight[:,:,0], (patch_width * patch_height))
    #     loss_weight += 0.01
    #     weight_sum = patch_height * patch_width
    #     cur_sum = np.sum(loss_weight)
    #     loss_weight *= weight_sum/cur_sum

        #loss_weight = np.reshape(loss_weight[:,0], (patch_width*patch_height,1))
    result = loss_weight
    print("shape of loss_weight:", result.shape)
    return result

def recompose_overlap(preds, img_h, img_w, stride_h, stride_w, channel, loss_weight=None):
    assert (len(preds.shape) == 4)  # 4D arrays
    #assert (preds.shape[1] == channel)
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]

    N_patches_h = (img_h - patch_h) // stride_h + 1
    N_patches_w = (img_w - patch_w) // stride_w + 1
    N_patches_img = N_patches_h * N_patches_w

    print("N_patches_h: ", N_patches_h)
    print("N_patches_w: ", N_patches_w)
    print("N_patches_img: ", N_patches_img)
    assert (preds.shape[0] % N_patches_img == 0)
    N_full_imgs = preds.shape[0] // N_patches_img

    print("According to the dimension inserted, there are {} full images (of {} x {} each)"
          .format(N_full_imgs, img_h, img_w))

    full_prob = np.zeros(
        (N_full_imgs, preds.shape[1], img_h, img_w))  # initialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))

    k = 0  # iterator over all the patches

    # extract each patch
    center = [patch_h // 2, patch_w // 2]
    expand = patch_h // 2
    left = center[1] - expand
    right = center[1] + expand
    top = center[0] - expand
    bottom = center[0] + expand

    if loss_weight is not None:
        weight = np.reshape(loss_weight, (patch_h, patch_w))
        weight +=0.000000001
    else:
        weight = 1

    for i in range(N_full_imgs):
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                full_prob[i, :, h * stride_h + top:(h * stride_h) + bottom,
                          w * stride_w + left:(w * stride_w) + right] += \
                    preds[k, :, top:bottom, left:right]*weight
                full_sum[i, :, h * stride_h + top:(h * stride_h) + bottom,
                         w * stride_w + left:(w * stride_w) + right] += weight
                k += 1
    #assert (k == preds.shape[0])
    #assert (np.min(full_sum) >= 0.0)  # must larger than 0
    #print(np.min(full_sum))

    final_avg = full_prob / (full_sum + 0.0000000001)
    #print("the shape of prediction result", final_avg.shape)
    #print("max value of prediction result", np.max(final_avg))
    #assert (np.max(final_avg) <= 1.01)  # max value for a pixel is 1.0
    #assert (np.min(final_avg) >= 0.0)  # min value for a pixel is 0.0
    return final_avg

# visualize image (as PIL image, NOT as matplotlib!)
def visualize(data, filename):
    assert (len(data.shape) == 3)  # height*width*channels
    # print data
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))  # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename)
    return img
