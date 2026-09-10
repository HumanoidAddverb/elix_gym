from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class ElixisRevoluteStandCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.948]
        default_joint_angles = {
            'l_hip_yaw': 0.,
            'l_hip_roll': 0.,
            'l_hip_pitch': -0.26,
            'l_knee': 0.50,
            'l_feet_pitch': -0.24,
            'l_feet_roll': 0.0,

            'r_hip_yaw': 0.,
            'r_hip_roll': 0.,
            'r_hip_pitch': -0.26,
            'r_knee': 0.50,
            'r_feet_pitch': -0.24,
            'r_feet_roll': 0.0,

            'torso': 0.0,
            'LJ1': 0.0,
            'LJ2': 0.0,
            'LJ3': 0.0,
            'LJ4': 0.0,
            'LJ5': 0.0,
            'LJ6': 0.0,
            'LJ7': 0.0,

            'RJ1': 0.0,
            'RJ2': 0.0,
            'RJ3': 0.0,
            'RJ4': 0.0,
            'RJ5': 0.0,
            'RJ6': 0.0,
            'RJ7': 0.0,
        }

    class commands(LeggedRobotCfg.commands):
        heading_command = False
        curriculum = False

        walk_vel_threshold = 0.1
        rel_standing_vel = 0.3
        resampling_time = 8.0
        command_decay_tau = 0.39
        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-0.75, 0.75]
            heading = [-3.14, 3.14]

            max_vel_x = [-0.8, 0.8]
            max_vel_y = [-0.5, 0.5]
            max_vel_yaw = [-0.75, 0.75]

    class env(LeggedRobotCfg.env):
        num_obs = 77
        num_observations = 77
        num_privileged_obs = 109
        obs_history_len = 5
        num_actions = 12

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {
            'hip_yaw': 80.,
            'hip_roll': 80.,
            'hip_pitch': 80.,
            'knee': 140.,
            'feet_pitch': 40.,
            'feet_roll': 40.,
            'torso': 100,
            'LJ1': 96.,
            'LJ2': 96.,
            'LJ3': 96.,
            'LJ4': 96.,
            'LJ5': 96.,
            'LJ6': 96.,
            'LJ7': 96.,
            'RJ1': 96.,
            'RJ2': 96.,
            'RJ3': 96.,
            'RJ4': 96.,
            'RJ5': 96.,
            'RJ6': 96.,
            'RJ7': 96.,
        }
        damping = {
            'hip_yaw': 5.,
            'hip_roll': 5.,
            'hip_pitch': 5.,
            'knee': 5.,
            'feet_pitch': 4.,
            'feet_roll': 4.,
            'torso': 5.,
            'LJ1': 2.,
            'LJ2': 2.,
            'LJ3': 2.,
            'LJ4': 2.,
            'LJ5': 2.,
            'LJ6': 2.,
            'LJ7': 2.,
            'RJ1': 2.,
            'RJ2': 2.,
            'RJ3': 2.,
            'RJ4': 2.,
            'RJ5': 2.,
            'RJ6': 2.,
            'RJ7': 2.,
        }
        action_scale = 0.25
        decimation = 4
        arm_gravity_comp = True
        action_jitter = True
        jitter_max_ms = 40
        jitter_decay = 0.85

    class sim(LeggedRobotCfg.sim):
        dt = 0.005

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 0.85]
        randomize_base_mass = True
        added_mass_range = [0., 6.]
        push_robots = True
        push_interval_s = 5.0
        max_push_vel_xy = 0.85
        randomize_com = True
        randomize_kp_factor = False
        randomize_kd_factor = False
        kp_factor_range = [0.9, 1.0]
        kd_factor_range = [0.95, 1.0]
        randomize_dof_pos = True
        dof_pos_range = [0.95, 1.05]

    class upper_body_rand:
        randomize = True
        resample_interval_s = 3.0
        ema_alpha_range = [0.005, 0.05]
        joint_amplitude = 0.15
        elbow_amplitude = 0.6

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/elixis_7Jan/meshes/elixis_revolute_joints.urdf'
        name = "elixis"
        foot_name = "feet_roll"
        penalize_contacts_on = ["knee", "hip_pitch", "hip_roll"]
        terminate_after_contacts_on = ["torso", "hip"]
        self_collisions = 0
        flip_visual_attachments = False
        armature = 0.02
        add_link_mass = False
        add_mass_on = ["torso", "hip", "head_pitch", "LJ1", "LJ2", "LJ3", "RJ1", "RJ2", "RJ3","l_hip_pitch", "r_hip_pitch", "l_knee", "r_knee"]

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        static_friction = 1.0
        dynamic_friction = 0.7
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20
        num_cols = 20
        max_init_terrain_level = 5

        terrain_proportions = [0.5, 0.5, 0., 0., 0.0]
        restitution = 0.
        slope_treshold = 0.90

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.939
        gait_period = 0.9
        stand_phase = 0.45

        feet_min_dist = 0.2
        feet_max_dist = 0.5

        air_symmetry_sigma = 0.15
        only_positive_rewards = False

        foot_target_height = -0.012 + 0.03 + 0.0770 + 0.025

        class scales(LeggedRobotCfg.rewards.scales):
            alive = 1.0

            tracking_lin_vel = 1.
            tracking_ang_vel = 0.5

            dof_acc = -2.5e-7
            torques = -2e-5
            dof_vel = -5e-3
            dof_pos_limits = -1.

            hip_yaw = -1.0
            hip_roll = -0.4
            feet_roll = -0.1
            joint_pos = -0.1

            torso_sway_damping = -0.85
            lin_vel_z = -1.0

            stand_still = 0.6

            orientation = 1.
            base_height = -2.
            collision = -5.

            feet_air_time = 1.
            feet_clearance = 1.
            feet_force_support = 0.01
            feet_longitudinal_alignment = 0.0
            feet_swing_forces = -0.02
            feet_stance_width = 0.0

            action_rate = -0.2
            action_2 = -0.1
            feet_air_symmetry = 0.5
            force_symmetry = 0.5

            power2 = -1.0e-5
            power = -1.0e-4
            torso_roll = 0.5

        tracking_sigma = 0.2

    class noise:
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 0.25
            dof_pos = 1.
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 30.
        clip_actions = 4.

class ElixisRevoluteStandCfgPPO(LeggedRobotCfgPPO):
    seed = 4007

    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.009

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCritic"
        max_iterations = 20000
        run_name = ''
        experiment_name = 'elixis_revolute_stand'
