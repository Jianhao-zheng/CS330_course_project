# CS330_course_project

## Getting started

* We use `Meta-World` as the simulator. The official one is under development. In case official version changes and potential custom modification, install it using our forked repo:

  ```bash
  conda create -n MetaWorld python=3.8
  conda activate MetaWorld
  cd 'where you want to put the Meta-World'
  git clone git@github.com:Jianhao-zheng/Metaworld.git
  cd Metaworld
  pip install -e .
  ```

To run the basic script `random_sample.py`, you need to install opencv lib:

  ```bash
  pip install opencv-python
  ```

* Code structure

  ```bash
  ├── data_collection                  # Code for unsurpervised data collection
  │   │── random_sample.py             # Collect data randomly
  │   │── vanilla_rnd.py               # Collect data by vanilla_rnd
  │   │── gcrl_rnd.py              
  │   │── data_collection_aps.py             
  │   └── data_collection_cic.py             
  │   └── ...                          # Utility modules
  ├── train                          
  │   └── offline_train_IQL.py         # Implicit Q-learning
  │── eval         
  │   └── eval.py                      # Evaluate single-task model
  └── └── eval_multi_task_model.py     # Evaluate multi-task model    
  ```

### Run data collector

* See the argument of `random_sample.py`, `vanilla_rnd.py`, `gcrl_rnd.py`, `data_collection_aps.py`, `data_collection_cic.py` for how to use the five unsupervised data collection method.
* You need to parse the argument corresponding to the name of the task and the path to save the data
* We share the data we collected by the 5 methods on the 10 tasks from MT10 in [this link](https://drive.google.com/drive/folders/1-mggsQw4oI4W2BRMs2P7W0CbSyWEtzxo)

### Run offline IQL

* To train a single-task IQL:
  ```bash
  python train/offline_train_IQL.py --data_path <path_to_the_training_data> --log <path_to_save_the_ckpt>
  ```

* To train s multi-task IQL, first put the data into the correct path, see details in `train/offline_train_IQL_multi_task.py`. Then run the following:
  ```bash
  python train/offline_train_IQL_multi_task.py --alg <choose_from_random/rnd/gcrl/aps/cic>
  ```

### Evaluate the trained model

* To evaluate a single-task model, you can use `eval/eval.py`. For example, if you want to evaluate a model trained by cic data on task of drawer open, download the model from [this link](https://drive.google.com/drive/folders/1pvwnuM6FVMFZtajo48gw_nSsDhCTsb1E), and run the following:
  ```bash
  python eval/eval.py --ckpt ./model/single-task/drawer_open/unsupervised_only/cic.d3 --task_name drawer-open-v2
  ```
* You can further watch how the robot arm behaves by parsing `--visual` and print the state value by parsing `--verbose`:
  ```bash
  python eval/eval.py --ckpt ./model/single-task/drawer_open/unsupervised_only/cic.d3 --task_name drawer-open-v2 --visual --verbose
  ```
* To evaluate multi-task model, see the example here:
  ```bash
  python eval/eval_multi_task_model.py --ckpt ./model/multi-task/cic.d3 --task_name drawer-open-v2
  ```