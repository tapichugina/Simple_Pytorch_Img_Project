import numpy as np
import matplotlib.pyplot as plt



def diagonal_line(img,row0,col0,img_size):
    coord_col=np.arange(0,img_size)


    if orientation=="right":
        coord_row=coord_col+row0-col0
    else:
        coord_row=-coord_col+row0+col0

    coords = np.column_stack((coord_row,coord_col))
    coords = coords[(coords >= 0).all(axis=1)]
    coords = coords[(coords < img_size).all(axis=1)]
    img[coords[:,0],coords[:,1]]=1

    return img
        


def gen_img(img_size,direction):
    img=np.zeros((img_size,img_size))
    start_row=np.random.randint(img_size)
    start_col=np.random.randint(img_size)


    if direction == 0:
        #"vertical"
        img[:, start_col] = 1

    elif direction == 1:
        #"horizontal"
        img[start_row, :] = 1

    elif direction == 2 :
        #"diagonal_right"
        img = diagonal_line(img, start_row, start_col, img_size, "right")

    elif direction == 3 :
        #"diagonal_left"
        img = diagonal_line(img, start_row, start_col, img_size, "left")
        

    return img.reshape(1,img_size,img_size)

def plot_images(images, targets, n_plot=30):
    n_rows = n_plot // 10 + ((n_plot % 10) > 0)
    fig, axes = plt.subplots(n_rows, 10, figsize=(15, 1.5 * n_rows))
    axes = np.atleast_2d(axes)

    for i, (image, target) in enumerate(zip(images[:n_plot], targets[:n_plot])):
        row, col = i // 10, i % 10    
        ax = axes[row, col]
        ax.set_title('#{} - Label:{}'.format(i, target), {'size': 12})
        # plot filter channel in grayscale
        ax.imshow(image.squeeze(), cmap='gray', vmin=0, vmax=1)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.label_outer()

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    n_images=1000
    img_size=10
    SEED = 42
    np.random.seed(SEED)
    label=np.random.randint(low=0, high=4,size=n_images)
    images = np.array([gen_img(img_size,target) for target in label],dtype=np.uint8)

    plot_images(images,label, n_plot=100);
