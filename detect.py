import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import sys
sys.path.append("TensorRT-YOLO")

from python.infer import TRTYOLO, ImageBatcher
from python.utils import visualize_detections, generate_labels_with_colors


def parse_opt() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='YOLO Series Inference Script.')
    parser.add_argument('-e', '--engine', required=True, type=str, help='The serialized TensorRT engine.')
    parser.add_argument('-i', '--input', required=True, type=str, help="Path to the image or directory to process.")
    parser.add_argument('-o', '--output', type=str, default=None, help='Directory where to save the visualization results.')
    parser.add_argument("-l", "--labels", default="./labels.txt", help="File to use for reading the class labels from, default: ./labels.txt")

    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()

    labels = generate_labels_with_colors(opt.labels)
    model = TRTYOLO(opt.engine)
    model.warmup()

    total_time = 0.0
    total_infers = 0
    total_images = 0
    print(f"Infering data in {opt.input}")
    batcher = ImageBatcher(
        input_path=opt.input, 
        batch_size=model.batch_size, 
        imgsz=model.imgsz, 
        dtype=model.dtype, 
        dynamic=model.dynamic
    )
    for batch, images, batch_shape in tqdm(batcher, total=len(batcher.batches), desc="Processing batches", unit="batch"):
        start_time_ns = time.perf_counter_ns()
        detections = model.infer(batch, batch_shape)
        end_time_ns = time.perf_counter_ns()
        elapsed_time_ms = (end_time_ns - start_time_ns) / 1e6
        total_time += elapsed_time_ms
        total_images += len(images)
        total_infers += 1
        if opt.output:
            output_dir = Path(opt.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor() as executor:
                args_list = [(str(image), str(output_dir / image.name), detections[i], labels) for i, image in enumerate(images)]
                executor.map(visualize_detections, *zip(*args_list))

    print("Benchmark results include time for H2D and D2H memory copies")
    average_latency = total_time / total_infers
    average_throughput = total_images / (total_time / 1000)
    print(f"Average Latency: {average_latency:.3f} ms")
    print(f"Average Throughput: {average_throughput:.1f} ips")
    print("Finished Processing.")