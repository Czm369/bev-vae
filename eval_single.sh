#!/bin/bash                                                                                                                                            
                                                                                                                                        
# export TORCH_CUDA_ARCH_LIST="8.0+PTX"    
export HYDRA_FULL_ERROR=1                                                                                                                              
export CUDA_VISIBLE_DEVICES=0        
export WAYVE_DATA_DIR="path_to_wayve_data"
export NUSCENES_DATA_DIR="path_to_nuscenes_data"                                                                                                                   
export ARGOVERSE_DATA_DIR="path_to_argoverse_data"    
export NUPLAN_DATA_DIR="path_to_nuplan_data"
export NUIMAGES_DATA_DIR="path_to_nuimages_data"
export CKPT_DIR="path_to_checkpoint_dir"                                                                                                                                                                            
export OUTPUT_DIR="path_to_output_dir"                           

python bev_vae/eval.py experiment=$1 devices=1 batch_size=1