from skimage import io
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
from filter import *
from segment_graph import *

def segment(in_image, sigma, k, min_size):
    start_time = time.time()
    height, width, band = in_image.shape
    print(f"Height: {height}, Width: {width}")

    def to_tensor(band):
        return torch.from_numpy(band).cuda().float()
    
    smooth_red = torch.tensor(smooth(in_image[:, :, 0], sigma)).cuda()
    smooth_green = torch.tensor(smooth(in_image[:, :, 1], sigma)).cuda()
    smooth_blue = torch.tensor(smooth(in_image[:, :, 2], sigma)).cuda()

  
    image_tensor = torch.stack([smooth_red, smooth_green, smooth_blue], dim=2)

 
    graph_start_time = time.time()
    
   
    device = 'cuda'
    

    y_right, x_right = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width-1, device=device),
        indexing='ij'
    )
    starts_right = (y_right * width + x_right).long()
    ends_right = starts_right + 1

    y_down, x_down = torch.meshgrid(
        torch.arange(height-1, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    starts_down = (y_down * width + x_down).long()
    ends_down = ((y_down + 1) * width + x_down).long()


    y_dr, x_dr = torch.meshgrid(
        torch.arange(height-1, device=device),
        torch.arange(width-1, device=device),
        indexing='ij'
    )
    starts_dr = (y_dr * width + x_dr).long()
    ends_dr = ((y_dr + 1) * width + (x_dr + 1)).long()


    y_ur, x_ur = torch.meshgrid(
        torch.arange(1, height, device=device),
        torch.arange(width-1, device=device),
        indexing='ij'
    )
    starts_ur = (y_ur * width + x_ur).long()
    ends_ur = ((y_ur - 1) * width + (x_ur + 1)).long()


    starts = torch.cat([
        starts_right.reshape(-1),
        starts_down.reshape(-1),
        starts_dr.reshape(-1),
        starts_ur.reshape(-1)
    ])
    
    ends = torch.cat([
        ends_right.reshape(-1),
        ends_down.reshape(-1),
        ends_dr.reshape(-1),
        ends_ur.reshape(-1)
    ])


    flat_image = image_tensor.view(-1, 3)
    diffs = torch.norm(flat_image[starts] - flat_image[ends], dim=1)

    starts_np = starts.cpu().numpy().astype(np.int64)
    ends_np = ends.cpu().numpy().astype(np.int64)
    diffs_np = diffs.cpu().numpy().astype(np.float64)


    edges_np = np.empty((len(starts_np), 3), dtype=object)
    edges_np[:, 0] = starts_np
    edges_np[:, 1] = ends_np
    edges_np[:, 2] = diffs_np

    graph_time = time.time() - graph_start_time
    print(f"Graph building time (CUDA): {graph_time:.2f} seconds")

    num_edges = edges_np.shape[0]


    u = segment_graph(width * height, num_edges, edges_np, k)


    for i in range(num_edges):
        a = u.find(edges_np[i, 0])
        b = u.find(edges_np[i, 1])
        if a != b and (u.size(a) < min_size or u.size(b) < min_size):
            u.join(a, b)


    num_cc = u.num_sets()
    output = np.zeros((height, width, 3), dtype=np.uint8)
    

    colors = np.random.randint(0, 255, (height * width, 3), dtype=np.uint8)
    
    # Map components to colors
    for y in range(height):
        for x in range(width):
            comp = u.find(y * width + x)
            output[y, x] = colors[comp]


    elapsed = time.time() - start_time
    print(f"Execution time: {int(elapsed // 60)}m {int(elapsed % 60)}s")

    output_path = "segmented_result.jpg"
    io.imsave(output_path, output)
    print(f"Segmented image saved to {output_path}")

if __name__ == "__main__":
    sigma = 0.5
    k = 500
    min_size = 100
    input_image = io.imread("data/bridge.jpg")
    
    print("Image loaded. Processing...")
    segment(input_image, sigma, k, min_size)
