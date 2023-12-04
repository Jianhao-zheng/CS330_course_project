#!/bin/bash
PYTHON=/home/george/.conda/envs/MetaWorld/bin/python

$PYTHON -m data_collection.gcrl --goal_sample_size 50 --save_path data/gcrl_50_task_50_goal_sz.npz
$PYTHON -m data_collection.gcrl --goal_sample_size 500 --save_path data/gcrl_50_task_500_goal_sz.npz
$PYTHON -m data_collection.gcrl --goal_sample_size 5000 --save_path data/gcrl_50_task_5000_goal_sz.npz
