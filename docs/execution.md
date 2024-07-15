## Execution

### Download datasets

- [ECD dataset](https://rpg.ifi.uzh.ch/davis_data.html): It contains four hand-held rotational motion sequences, which is recorded with a DAVIS240C (240 x 180 px).
  - [shapes_rotation.bag](https://rpg.ifi.uzh.ch/datasets/davis/shapes_rotation.bag)
  - [poster_rotation.bag](https://rpg.ifi.uzh.ch/datasets/davis/poster_rotation.bag)
  - [boxes_rotation.bag](https://rpg.ifi.uzh.ch/datasets/davis/boxes_rotation.bag)
  - [dynamic_rotation.bag](https://rpg.ifi.uzh.ch/datasets/davis/dynamic_rotation.bag)

- [ECRot dataset](https://github.com/tub-rip/ECRot): It contains six synthetic sequences (using a DAVIS240C model) and ten real-world sequences, recorded with a DVXplorer (640 x 480 px resolution).

After downloading, we recommend to organize your dataset directory as follows:

```shell
│── dataset_root_dir/
│   ├── dataset/               # e.g., ECRot_dataset
│       ├── sequence/          # e.g., bicycle
│              └── events.bag  # The rosbag that contains events (and camera_info)
│                  ...
```

Finally, remember to change `dataset_root_dir` in the launch files.

### Download initial trajectories and maps

In addition to events, EMBA takes as input initial camera trajectories and gradient maps (if available) for refinement. We provide the front-end trajectories and the corresponding maps (download [here](https://drive.google.com/file/d/1MgoxOirA2jMDV4kc6S8FHodIqI0PEl-r/view?usp=sharing)), so that users can conveniently download and test.

Check and make sure that the directory of the input data looks like the following (do not change):

```shell
│── input_data_dir/
│   ├── dataset/                           # e.g., ECRot_dataset
│       ├── sequence/                      # e.g., bicycle
│           ├── map/  
│               └── frontend/              # Initial maps generated from the below initial  trajectories
│                   ├── filename_traj/     # Initial maps generated from the below initial trajectories
│                       ├── bin/           # initial maps stored in .bin format (used in EMBA)
│                           ├── Gx.bin
│                           └── Gy.bin
│                       ├── jpg/           # Visualization of the initial maps
│                       └── txt/           # Just in case if you cannot load binary files
│                       ...
│           └── traj
│               ├── frontend/              # Raw front-end trajectories (not used in EMBA)
│               ├── groundtruth/           # GT trajectories (not used in EMBA)
│               └── interpolation/         # Interpolated front-end trajectories (used in EMBA)
│                   ├── filename_raw_traj.txt
│                   └── ...
```

Note that every trajectory in `traj/interpolation/` has a corresponding map in `map/frontend/`.
By default, the resolution of initial map is 1024 x 512, except some trajectory filenames end with 2048, which corresponds to the map size of 2048 x 1024.

Finally, remember to change the `input_data_dir` and `filename_raw_traj` in the launch files.

### Run EMBA

Change the `output_data_dir` in the launch file, for saving output data.

Run the launch file that corresponds to the data sequence you want to test on, e.g.:

    roslaunch emba bicycle.launch

Note that to test on `playroom`, whose rosbag does not have `camera_info`, you need to copy the [calibration file](https://github.com/Shuang1997/emba/blob/release/calib/DVS-playroom.yaml) to `/home/username/.ros/camera_info` before you run `roslaunch emba playroom.launch`.
