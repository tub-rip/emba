## Parameter Guide

Here we introduce the main parameters of EMBA, which are specified (and briefly explained) in launch files; see an [example launch file](../launch/bicycle.launch).

### Topics

- `events_topic`: The topic name for the event data. Note that the message type should be `dvs_msgs/EventArray`.
- `camera_info_topic`: The topic name for the camera calibration. If it is not included in your rosbag, you can create a [yaml file](https://github.com/Shuang1997/emba/blob/release/calib/DVS-playroom.yaml) with the calibration information and copy it to `/home/username/.ros/camera_info`. If you do so, you also need to replace the parameter name `camera_info_topic` with `camera_name` and pass its value in the launch file (like [playroom.launch](https://github.com/Shuang1997/emba/blob/release/launch/playroom.launch)).

### EMBA Parameters

**General settings**

- `filename_raw_traj`: Raw trajectory (and the corresponding map) to refine.
- `init_map_available`: Whether the initial gradient map is available. If not, the gradient map would be initialized with random noise (i.e., recovering the map from scratch).
- `start_time` and `stop_time`: Time interval of BA.
- `C_th`: Contrast threshold of the event camera.
- `thres_valid_pixel`: Threshold of valid pixel selection.
- `alpha`: Weight of the L2 regularizer.
- `damping_factor`: Damping factor to slow down the map updating (not recommended, just set to 1).
- `event_sampling_rate`: Rate of systematic event sampling. We do not recommended to sample events, which would affect the map quality, unless your memory runs out.
- `dt_knots`: Time interval between the consecutive knots/control poses of the linear spline trajectory.

**Sliding-window settings**

EMBA is optimization-based, and for the sake of versatility we provide a sliding-window implementation.
In the experiments, we set the size of the time window size to the entire bundle adjustment (BA) observation window to refine the whole trajectory and map.

- `time_window_size`: Size of the sliding time window [s].
- `sliding_window_stride`: Stride of the sliding time window [s].

**Levenberg-Marquardt solver settings**

- `max_num_iter`: Maximal iteration times.
- `tol_fun`: Function tolerance for detecting convergence.
- `num_times_tol_fun_sat = 2`: If the function tolerance is achieved for consecutive two times, the solver convergences.
- `use_CG`: Whether to use the conjugate gradient (CG) solver? Otherwise, the Schur complement is used to solve the normal equation.
- `use_IRLS`: Whether to use robust cost function (IRLS).
- `cost_type`: If `use_IRLS = True`, pick a cost type between `huber` and `cauchy`. Otherwise, the original quadratic cost is used.
- `a`: Coefficient $\eta$ for the `huber` and `cauchy` loss functions. See the supplementary material of our paper for the full formula.
