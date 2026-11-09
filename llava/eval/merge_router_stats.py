"""
Merge router weight statistics from multiple evaluation chunks.

Supports separate pre-router and per-LLM-layer router weights.
"""
import argparse
import json
import numpy as np
import os


def merge_router_stats(stats_dir, num_chunks, output_file):
    """Merge per-chunk router statistics JSON files into one."""
    all_per_sample_weights = []
    total_samples = 0
    first_data = None
    has_pre_router = False

    for idx in range(num_chunks):
        chunk_file = os.path.join(stats_dir, f'chunk_{idx}_stats.json')
        if os.path.exists(chunk_file):
            with open(chunk_file, 'r') as f:
                data = json.load(f)
            
            if first_data is None:
                first_data = data
                has_pre_router = data['metadata'].get('has_pre_router', False)
            
            if 'per_sample_weights' in data:
                all_per_sample_weights.extend(data['per_sample_weights'])
            
            n = data['metadata']['num_samples']
            total_samples += n
            print(f'Loaded chunk_{idx}_stats.json ({n} samples)')
        else:
            print(f'Warning: chunk_{idx}_stats.json not found')

    if not all_per_sample_weights:
        print('Error: no valid statistics files')
        return

    all_llm_weights_matrix = []
    all_pre_router_weights = []
    
    for sample in all_per_sample_weights:
        if 'llm_router_weights' in sample:
            sample_matrix = []
            for layer_idx in sorted(sample['llm_router_weights'].keys(), key=int):
                sample_matrix.append(sample['llm_router_weights'][layer_idx])
            if sample_matrix:
                all_llm_weights_matrix.append(sample_matrix)
            
            if sample.get('pre_router_weights') is not None:
                all_pre_router_weights.append(sample['pre_router_weights'])
        elif 'weights' in sample:
            sample_matrix = []
            for layer_idx in sorted(sample['weights'].keys(), key=int):
                sample_matrix.append(sample['weights'][layer_idx])
            if sample_matrix:
                all_llm_weights_matrix.append(sample_matrix)

    all_llm_weights_array = np.array(all_llm_weights_matrix)  # [num_samples, num_llm_layers, num_vit_layers]
    llm_global_mean = np.mean(all_llm_weights_array, axis=0)
    llm_global_std = np.std(all_llm_weights_array, axis=0)

    output_data = {
        'metadata': {
            'num_samples': total_samples,
            'num_llm_layers': int(first_data['metadata']['num_llm_layers']),
            'num_vit_layers': int(first_data['metadata']['num_vit_layers']),
            'has_pre_router': has_pre_router,
            'model_path': first_data['metadata']['model_path'],
            'dataset': first_data['metadata']['dataset'],
            'num_chunks_merged': num_chunks
        },
        'llm_router': {
            'mean_weights': llm_global_mean.tolist(),
            'std_weights': llm_global_std.tolist(),
        },
        'per_sample_weights': all_per_sample_weights
    }

    if all_pre_router_weights:
        pre_router_array = np.array(all_pre_router_weights)  # [num_samples, num_vit_layers]
        pre_router_mean = np.mean(pre_router_array, axis=0)
        pre_router_std = np.std(pre_router_array, axis=0)
        
        output_data['pre_router'] = {
            'mean_weights': pre_router_mean.tolist(),
            'std_weights': pre_router_std.tolist(),
        }
        has_pre_router = True

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f'\nMerged router statistics written to: {output_file}')
    print(f'Total samples: {total_samples}')
    print(f'pre-router: {"present" if has_pre_router else "absent"}')

    if has_pre_router and 'pre_router' in output_data:
        print('\n' + '=' * 70)
        print('Pre-router mean weights (before LLM layer 0)')
        print('=' * 70)
        pre_mean = output_data['pre_router']['mean_weights']
        top_3_vit = np.argsort(pre_mean)[-3:][::-1]
        print(f'Pre-router: top-3 ViT layer indices = {list(top_3_vit)}, weights = {[f"{pre_mean[i]:.4f}" for i in top_3_vit]}')
        print(f'All ViT layer weights: {[f"{w:.4f}" for w in pre_mean]}')

    print('\n' + '=' * 70)
    print('Per-LLM-layer router mean weights (layer-wise router)')
    print('=' * 70)
    for llm_layer in [0, 8, 16, 24, 31]:
        if llm_layer < len(llm_global_mean):
            weights = llm_global_mean[llm_layer]
            top_3_vit = np.argsort(weights)[-3:][::-1]
            print(f'LLM layer {llm_layer:2d}: top-3 ViT layers = {list(top_3_vit)}, weights = {[f"{weights[i]:.4f}" for i in top_3_vit]}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-dir", type=str, required=True, help="Directory containing chunk_*_stats.json files")
    parser.add_argument("--num-chunks", type=int, required=True, help="Number of chunks to merge")
    parser.add_argument("--output-file", type=str, required=True, help="Output path for merged JSON")
    args = parser.parse_args()

    merge_router_stats(args.stats_dir, args.num_chunks, args.output_file)
