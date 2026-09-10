import numpy as np
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch
import torch

LEFT_SWING_TARGET = 0.4
RIGHT_SWING_TARGET_START = 0.5
RIGHT_SWING_TARGET_END = 0.9

class ElixisRevoluteStand(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.use_history = False
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        if not self.use_history:
            self.obs_history_len = 1
            self.cfg.env.obs_history_len = 1
            self.cfg.env.num_observations = self.num_obs
            self.num_observations = self.num_obs

        self.obs_buf = torch.zeros(size=(self.num_envs, self.cfg.env.num_obs * self.cfg.env.obs_history_len)).to(self.device)

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(size=(self.num_envs, self.cfg.env.num_obs)).to(self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[0:3] = noise_scales.ang_vel * noise_level
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0.
        noise_vec[45:47] = 0.
        noise_vec[47:62] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[62:77] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _init_gravity_comp(self):
        self._arm_gravity_comp = getattr(self.cfg.control, "arm_gravity_comp", True)
        if not self._arm_gravity_comp:
            return
        jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.cfg.asset.name)
        self.jacobian = gymtorch.wrap_tensor(jacobian)
        props = self.gym.get_actor_rigid_body_properties(self.envs[0], self.actor_handles[0])
        self.link_masses = torch.tensor([p.mass for p in props], device=self.device, dtype=torch.float)
        self.gravity_accel = torch.tensor([0.0, 0.0, self.cfg.sim.gravity[2]], device=self.device, dtype=torch.float)
        self._jac_dof_offset = self.jacobian.shape[-1] - self.num_dof

    def _compute_gravity_comp(self):
        self.gym.refresh_jacobian_tensors(self.sim)
        j_lin = self.jacobian[:, :, 0:3, self._jac_dof_offset:]
        return -torch.einsum('eiad,i,a->ed', j_lin, self.link_masses, self.gravity_accel)

    def _init_buffers(self):
        super()._init_buffers()
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        lower_keywords = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "feet_pitch", "feet_roll")
        lower_idx = []
        upper_idx = []
        for i, name in enumerate(self.dof_names):
            if any(k in name for k in lower_keywords):
                lower_idx.append(i)
            else:
                upper_idx.append(i)
        self.lower_dof_indices = torch.tensor(lower_idx, dtype=torch.long, device=self.device)
        self.upper_dof_indices = torch.tensor(upper_idx, dtype=torch.long, device=self.device)
        assert len(lower_idx) == 12 and len(upper_idx) == 15, (f"Expected 12 lower + 15 upper DOFs, got {len(lower_idx)} + {len(upper_idx)}")

        elbow_amp = self.cfg.upper_body_rand.elbow_amplitude
        other_amp = self.cfg.upper_body_rand.joint_amplitude
        is_elbow = [self.dof_names[i] in ("LJ4", "RJ4") for i in upper_idx]
        amps = [elbow_amp if e else other_amp for e in is_elbow]
        self.upper_joint_amplitude = torch.tensor(amps, dtype=torch.float, device=self.device)

        def _dof(name):
            return list(self.dof_names).index(name)
        self.hip_yaw_idx     = torch.tensor([_dof("l_hip_yaw"),  _dof("r_hip_yaw")],  dtype=torch.long, device=self.device)
        self.hip_roll_idx    = torch.tensor([_dof("l_hip_roll"), _dof("r_hip_roll")], dtype=torch.long, device=self.device)
        self.feet_roll_idx   = torch.tensor([_dof("l_feet_roll"),_dof("r_feet_roll")],dtype=torch.long, device=self.device)
        self.left_hipyr_idx  = torch.tensor([_dof("l_hip_yaw"),  _dof("l_hip_roll")], dtype=torch.long, device=self.device)
        self.right_hipyr_idx = torch.tensor([_dof("r_hip_yaw"),  _dof("r_hip_roll")], dtype=torch.long, device=self.device)
        print(f"[ElixisRevoluteStand] dof_names = {list(self.dof_names)}")
        print(f"[ElixisRevoluteStand] lower_dof_indices = {self.lower_dof_indices.tolist()}  hip_yaw = {self.hip_yaw_idx.tolist()}  hip_roll = {self.hip_roll_idx.tolist()}")

        self.walking = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.phase = torch.full((self.num_envs,), self.cfg.rewards.stand_phase, device=self.device, dtype=torch.float)

        self.phase_offset = torch.rand(self.num_envs, device=self.device)
        steps_per_cycle = self.cfg.rewards.gait_period / self.dt
        self.command_targets = self.commands.clone()
        self._init_foot()
        self._init_gravity_comp()
        self.last_air_time_left = torch.zeros(self.num_envs, device=self.device)
        self.last_air_time_right = torch.zeros(self.num_envs, device=self.device)
        self.prev_contact = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.bool)
        self.default_default_dof_pos = self.default_dof_pos.clone()
        self.dof_pos_lower = torch.tensor([-0.01, -0.01, -0.46, 0.30, -0.44, -0.01,   -0.01, -0.01, -0.46, 0.30, -0.44, -0.01], device=self.device)
        self.dof_pos_upper = torch.tensor([ 0.01,  0.01, -0.06, 0.70, -0.04,  0.01,    0.01,  0.01, -0.06, 0.70, -0.04,  0.01], device=self.device)
        self.kp_factors = torch.ones(self.num_envs, 12, device=self.device, dtype=torch.float, requires_grad=False)
        self.kd_factors = torch.ones(self.num_envs, 12, device=self.device, dtype=torch.float, requires_grad=False)

        u = self.upper_dof_indices
        default_upper = self.default_dof_pos[0, u]
        self.upper_target_pos = default_upper.unsqueeze(0).repeat(self.num_envs, 1).clone()
        self.upper_target_sampled = self.upper_target_pos.clone()
        self.ema_alpha = torch.zeros(self.num_envs, 1, device=self.device)
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        if self.cfg.upper_body_rand.randomize:
            self._resample_ema_alpha(all_env_ids)
            self._resample_upper_targets(all_env_ids)
        self._action_track_max = torch.zeros(self.num_envs, device=self.device)
        self._action_track_min = torch.zeros(self.num_envs, device=self.device)

        self._jitter_enabled = self.cfg.control.action_jitter
        if self._jitter_enabled:
            self._jitter_max_steps = int(self.cfg.control.jitter_max_ms / (self.cfg.sim.dt * 1000))
            steps = torch.arange(self._jitter_max_steps + 1, device=self.device, dtype=torch.float)
            weights = self.cfg.control.jitter_decay ** steps
            self._jitter_probs = (weights / weights.sum()).cpu().numpy()
            self._jitter_current_action = self.actions.clone()
            self._jitter_new_action = self.actions.clone()
            self._jitter_substep_idx = torch.zeros(1, dtype=torch.long, device=self.device)
            self._jitter_hold_remaining = torch.zeros(1, dtype=torch.long, device=self.device)

    def _resample_ema_alpha(self, env_ids):
        if len(env_ids) == 0:
            return
        a_min, a_max = self.cfg.upper_body_rand.ema_alpha_range
        self.ema_alpha[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * (a_max - a_min) + a_min

    def _resample_upper_targets(self, env_ids):
        if len(env_ids) == 0:
            return
        u = self.upper_dof_indices
        default_upper = self.default_dof_pos[0, u]
        rand = (torch.rand(len(env_ids), len(u), device=self.device) * 2.0 - 1.0) * self.upper_joint_amplitude
        target = default_upper.unsqueeze(0) + rand
        target = torch.clamp(target, self.dof_pos_limits[u, 0], self.dof_pos_limits[u, 1])
        self.upper_target_sampled[env_ids] = target

    def _update_upper_targets(self):
        if not self.cfg.upper_body_rand.randomize:
            return
        resample_steps = max(1, int(self.cfg.upper_body_rand.resample_interval_s / self.dt))
        resample_ids = (self.episode_length_buf % resample_steps == 0).nonzero(as_tuple=False).flatten()
        self._resample_upper_targets(resample_ids)
        self.upper_target_pos = self.ema_alpha * self.upper_target_sampled + (1.0 - self.ema_alpha) * self.upper_target_pos

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _update_gait_phase(self):
        self.walking = self.commands[:, 0].abs() > self.cfg.commands.walk_vel_threshold
        cycles = self.episode_length_buf * self.dt / self.cfg.rewards.gait_period + self.phase_offset
        phase = torch.remainder(cycles, 1.0)
        self.phase = torch.where(self.walking, phase, torch.full_like(phase, self.cfg.rewards.stand_phase))

    def _post_physics_step_callback(self):
        self.update_feet_state()

        decay = np.exp(-self.dt / self.cfg.commands.command_decay_tau)
        diff = (self.commands[:,:2] - self.command_targets[:,:2]) * decay
        diff[diff.abs() < 0.02] = 0.0
        self.commands[:,:2] = self.command_targets[:,:2] + diff

        self._update_gait_phase()
        self._update_upper_targets()

        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        touchdown_left  = contact[:, 0] & ~self.prev_contact[:, 0]
        touchdown_right = contact[:, 1] & ~self.prev_contact[:, 1]

        self.last_air_time_left  = torch.where(touchdown_left,  self.feet_air_time[:, 0], self.last_air_time_left)
        self.last_air_time_right = torch.where(touchdown_right, self.feet_air_time[:, 1], self.last_air_time_right)
        self.prev_contact = contact.clone()

        return super()._post_physics_step_callback()

    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            added_mass = np.random.uniform(rng[0], rng[1])
            props[1].mass += added_mass
            props[7].mass += added_mass
        return props

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def update_command_curriculum(self, env_ids):
        lin_ok = torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]
        ang_ok = torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_ang_vel"]

        step = 0.1
        if lin_ok:
            for axis, key in [("lin_vel_x", "max_vel_x"), ("lin_vel_y", "max_vel_y")]:
                cap = getattr(self.cfg.commands.ranges, key)
                self.command_ranges[axis][0] = max(cap[0], self.command_ranges[axis][0] - step)
                self.command_ranges[axis][1] = min(cap[1], self.command_ranges[axis][1] + step)
        if ang_ok:
            cap = self.cfg.commands.ranges.max_vel_yaw
            self.command_ranges["ang_vel_yaw"][0] = max(cap[0], self.command_ranges["ang_vel_yaw"][0] - step)
            self.command_ranges["ang_vel_yaw"][1] = min(cap[1], self.command_ranges["ang_vel_yaw"][1] + step)

    def _compute_torques(self, actions):
        if self._jitter_enabled:
            if self._jitter_substep_idx.item() == 0:
                delay = np.random.choice(self._jitter_max_steps + 1, p=self._jitter_probs)
                self._jitter_hold_remaining[0] = delay
                self._jitter_new_action[:] = actions
            if self._jitter_hold_remaining.item() > 0:
                self._jitter_hold_remaining[0] -= 1
                actions = self._jitter_current_action
            else:
                self._jitter_current_action[:] = self._jitter_new_action
            self._jitter_substep_idx[0] = (self._jitter_substep_idx[0] + 1) % self.cfg.control.decimation
            self.actions[:] = actions
        actions_scaled = actions * self.cfg.control.action_scale

        torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        u = self.upper_dof_indices
        upper_pos_err = self.upper_target_pos - self.dof_pos[:, u]
        torques[:, u] = self.p_gains[u] * upper_pos_err - self.d_gains[u] * self.dof_vel[:, u]
        if self._arm_gravity_comp:
            torques[:, u] += self._compute_gravity_comp()[:, u]
        l = self.lower_dof_indices
        lower_pos_err = actions_scaled + self.default_dof_pos[:, l] - self.dof_pos[:, l]
        torques[:, l] = self.p_gains[l] * lower_pos_err - self.d_gains[l] * self.dof_vel[:, l]

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        prev_lin = self.commands[env_ids, :2].clone()
        super()._resample_commands(env_ids)
        r = torch.rand(len(env_ids), device=self.device)
        stand = (r < self.cfg.commands.rel_standing_vel) | (self.commands[env_ids, 0].abs() <= self.cfg.commands.walk_vel_threshold)
        stand_ids = env_ids[stand]
        self.commands[stand_ids, :3] = 0.0

        forward = quat_apply(self.base_quat[stand_ids], self.forward_vec[stand_ids])
        self.commands[stand_ids, 3] = torch.atan2(forward[:, 1], forward[:, 0])
        self.command_targets[env_ids] = self.commands[env_ids]
        self.commands[env_ids,:2] = prev_lin

    def compute_observations(self):
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(-1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(-1)
        base_height = self.root_states[:, 2] - self._terrain_height_at(self.root_states[:, :2])
        foot_heights = self.feet_pos[:, :, 2] - self._terrain_height_at(self.feet_pos[:, :, :2])
        static_friction_coeffs = (self.friction_coeffs + self.cfg.terrain.static_friction) / 2.0
        dynamic_friction_coeffs = (self.friction_coeffs + self.cfg.terrain.dynamic_friction) / 2.0
        l = self.lower_dof_indices
        u = self.upper_dof_indices
        self.privileged_obs_buf = torch.cat((
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos[:, l] - self.default_dof_pos[:, l]) * self.obs_scales.dof_pos,
                self.dof_vel[:, l] * self.obs_scales.dof_vel,
                self.actions,
                self.last_actions,
                self.last_last_actions,
                sin_phase,
                cos_phase,
                static_friction_coeffs.squeeze(-1).to(self.device),
                dynamic_friction_coeffs.squeeze(-1).to(self.device),
                base_height.unsqueeze(-1),
                foot_heights,
                (self.dof_pos[:, u] - self.default_dof_pos[:, u]) * self.obs_scales.dof_pos,
                self.dof_vel[:, u] * self.obs_scales.dof_vel
            ),
            dim=-1,
        )
        self.local_obs_buf = torch.cat((
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos[:, l] - self.default_dof_pos[:, l]) * self.obs_scales.dof_pos,
                self.dof_vel[:, l] * self.obs_scales.dof_vel,
                self.actions,
                sin_phase,
                cos_phase,
                (self.dof_pos[:, u] - self.default_dof_pos[:, u]) * self.obs_scales.dof_pos,
                self.dof_vel[:, u] * self.obs_scales.dof_vel
            ),
            dim=-1,
        ).to(self.device)

        if self.add_noise:
            self.local_obs_buf += (2 * torch.rand_like(self.local_obs_buf) - 1) * self.noise_scale_vec

        if self.use_history:
            self.obs_buf = torch.concat((self.obs_buf[:, self.cfg.env.num_obs:], self.local_obs_buf), dim=-1)
        else:
            self.obs_buf = self.local_obs_buf

        self.last_last_actions[:] = self.last_actions[:]

    def reset_idx(self, env_ids):
        if self.cfg.domain_rand.randomize_dof_pos and self.default_dof_pos.shape[0] == 1:
            self.default_dof_pos = self.default_dof_pos.expand(self.num_envs, -1).clone()

        if self.cfg.commands.curriculum:
            self.update_command_curriculum(env_ids)

        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.phase_offset[env_ids] = torch.rand(len(env_ids), device=self.device)
        self.commands[env_ids,:2] = self.command_targets[env_ids,:2]
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            self.extras["episode"]["max_command_y"] = self.command_ranges["lin_vel_y"][1]
            self.extras["episode"]["max_command_yaw"] = self.command_ranges["ang_vel_yaw"][1]

        self.extras["episode"]["action_abs_max"] = self._action_track_max[env_ids].mean()
        self.extras["episode"]["action_min"] = self._action_track_min[env_ids].mean()
        self._action_track_max[env_ids] = 0
        self._action_track_min[env_ids] = 0
        l = self.lower_dof_indices
        if self.cfg.domain_rand.randomize_dof_pos:
            ep_len = self.episode_length_buf[env_ids]
            K = torch.zeros(len(env_ids), device=self.device)
            K[(ep_len >= 150) & (ep_len < 250)] = 0.0125
            K[(ep_len >= 250) & (ep_len < 450)] = 0.0250
            K[(ep_len >= 450) & (ep_len < 750)] = 0.0375
            K[ep_len >= 750] = 0.0625
            base = self.default_default_dof_pos.clone()
            nd = base.expand(len(env_ids), -1).clone()
            leg_noise = K.unsqueeze(-1) * (torch.rand(len(env_ids), 12, device=self.device) - 0.5)
            nd[:, l] = (nd[:, l] + leg_noise).clamp(self.dof_pos_lower, self.dof_pos_upper)
            self.default_dof_pos[env_ids] = nd

        if self.cfg.domain_rand.randomize_kp_factor:
            min_kp_factor, max_kp_factor = self.cfg.domain_rand.kp_factor_range
            self.kp_factors[env_ids, :] = torch.rand(len(env_ids), self.num_actions, device=self.device, dtype=torch.float, requires_grad=False) * (max_kp_factor - min_kp_factor) + min_kp_factor

        if self.cfg.domain_rand.randomize_kd_factor:
            min_kd_factor, max_kd_factor = self.cfg.domain_rand.kd_factor_range
            self.kd_factors[env_ids, :] = torch.rand(len(env_ids), self.num_actions, device=self.device, dtype=torch.float, requires_grad=False) * (max_kd_factor - min_kd_factor) + min_kd_factor

        if self.cfg.upper_body_rand.randomize:
            self._resample_ema_alpha(env_ids)
            self._resample_upper_targets(env_ids)
        else:
            self.upper_target_sampled[env_ids] = self.default_default_dof_pos[0, self.upper_dof_indices].unsqueeze(0)
        self.upper_target_pos[env_ids] = self.upper_target_sampled[env_ids].clone()

        u = self.upper_dof_indices
        self.dof_pos[env_ids[:, None], u] = self.upper_target_sampled[env_ids]
        self.dof_vel[env_ids[:, None], u] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reward_alive(self):
        return 1.0

    def _reward_hip_yaw(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_yaw_idx]), dim=1)

    def _reward_hip_roll(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_roll_idx]), dim=1)

    def _reward_feet_roll(self):
        return torch.sum(torch.square(self.dof_pos[:, self.feet_roll_idx]), dim=1)

    def _reward_stand_still(self):
        l = self.lower_dof_indices
        pose_err = torch.sum(torch.abs(self.dof_pos[:, l] - self.default_dof_pos[:, l]), dim=1)
        return torch.exp(-pose_err / 0.5) * (~self.walking).float()

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        self.feet_air_time += self.dt
        self.feet_air_time *= (~contact).float()

        left_swing_target = (self.phase < LEFT_SWING_TARGET)
        right_swing_target = ((self.phase >= RIGHT_SWING_TARGET_START) & (self.phase < RIGHT_SWING_TARGET_END))

        rew = torch.zeros(self.num_envs, device=self.device)
        rew += (~contact[:, 0] & left_swing_target).float()
        rew += (~contact[:, 1] & right_swing_target).float()

        return rew

    def _reward_action_2(self):
        return torch.sum(torch.square((self.actions - self.last_actions) - (self.last_actions - self.last_last_actions)), dim=1)

    def _reward_feet_clearance(self):
        target_height = self.cfg.rewards.foot_target_height
        foot_heights = self.feet_pos[:, :, 2] - self._terrain_height_at(self.feet_pos[:, :, :2])

        left_target_z = target_height * (self.phase < LEFT_SWING_TARGET).float()
        right_target_z = target_height * ((self.phase >= RIGHT_SWING_TARGET_START) & (self.phase < RIGHT_SWING_TARGET_END)).float()

        left_error = torch.square(foot_heights[:, 0] - left_target_z)
        right_error = torch.square(foot_heights[:, 1] - right_target_z)
        return torch.exp(-(left_error + right_error) / 0.01)

    def _reward_stand_still_stationary(self):
        diff = self.base_pos[:, :2] - self.env_origins[:, :2]
        distance_sq = torch.sum(torch.square(diff), dim=1)
        return torch.exp(-distance_sq / 0.02) * (~self.walking).float()

    def _reward_orientation(self):
        tilt_sq = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return torch.exp(-tilt_sq / 0.1)

    def _reward_feet_force_support(self):
        contact_forces = self.contact_forces[:, self.feet_indices, 2]
        left_stance = (self.phase >= LEFT_SWING_TARGET).float()
        right_stance = ((self.phase < RIGHT_SWING_TARGET_START) | (self.phase >= RIGHT_SWING_TARGET_END)).float()

        return left_stance * torch.tanh(contact_forces[:, 0] / 750.0) + right_stance * torch.tanh(contact_forces[:, 1] / 750.0)

    def _reward_feet_longitudinal_alignment(self):
        feet_pos_local_0 = quat_rotate_inverse(self.base_quat, self.feet_pos[:, 0, :] - self.base_pos)
        feet_pos_local_1 = quat_rotate_inverse(self.base_quat, self.feet_pos[:, 1, :] - self.base_pos)
        foot_x_diff = feet_pos_local_0[:, 0] - feet_pos_local_1[:, 0]
        return torch.exp(-torch.square(foot_x_diff) / 0.01)

    def _reward_torso_sway_damping(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_torso_roll(self):
        return torch.exp(-torch.square(self.rpy[:, 0]) / 0.002)

    def _reward_joint_pos(self):
        l = self.lower_dof_indices
        pos_square_error = torch.square(self.dof_pos[:, l] - self.default_dof_pos[:, l])
        weights = torch.tensor([1.5,1.5,1.0, 1.0,1.0,1.0,
                                1.5,1.5,1.0, 1.0,1.0,1.0
                                ],device=self.device)
        return torch.sum(pos_square_error*weights, dim=-1)

    def _reward_lin_vel_y_penalty(self):
        no_lateral_cmd = torch.abs(self.commands[:, 1]) < 0.05
        return torch.square(self.base_lin_vel[:, 1]) * ( no_lateral_cmd & self.walking).float()

    def _reward_feet_stance_width(self):
        feet_pos_local_0 = quat_rotate_inverse(self.base_quat, self.feet_pos[:, 0, :] - self.base_pos)
        feet_pos_local_1 = quat_rotate_inverse(self.base_quat, self.feet_pos[:, 1, :] - self.base_pos)
        foot_y_diff = torch.abs(feet_pos_local_0[:, 1] - feet_pos_local_1[:, 1])
        target_width = 0.18
        return torch.exp(-torch.square(foot_y_diff - target_width) / 0.01)

    def _reward_feet_swing_forces(self):
        contact_forces = self.contact_forces[:, self.feet_indices, 2]

        left_swing = (self.phase < LEFT_SWING_TARGET).float()
        right_swing = ((self.phase >= RIGHT_SWING_TARGET_START) & (self.phase < RIGHT_SWING_TARGET_END)).float()

        left_penalty = left_swing * torch.clamp_min(contact_forces[:, 0] - 10.0, 0.0)
        right_penalty = right_swing * torch.clamp_min(contact_forces[:, 1] - 10.0, 0.0)

        return left_penalty + right_penalty

    def _reward_hip_yaw_strict(self):
        left_swing  = (self.phase < LEFT_SWING_TARGET).float().unsqueeze(1)
        right_swing = ((self.phase >= RIGHT_SWING_TARGET_START) & (self.phase < RIGHT_SWING_TARGET_END)).float().unsqueeze(1)

        left_penalty  = torch.sum(torch.square(self.dof_pos[:, self.left_hipyr_idx]),  dim=1, keepdim=True)
        right_penalty = torch.sum(torch.square(self.dof_pos[:, self.right_hipyr_idx]), dim=1, keepdim=True)

        return (left_penalty * left_swing + right_penalty * right_swing).squeeze(1)

    def _reward_feet_distance(self):
        foot_dist = torch.norm(self.feet_pos[:, 0, :] - self.feet_pos[:, 1, :], dim=1)
        d_min = torch.clamp(foot_dist - self.cfg.rewards.feet_min_dist, -0.5, 0.)
        d_max = torch.clamp(foot_dist - self.cfg.rewards.feet_max_dist, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_base_height(self):
        base_height = self.root_states[:, 2] - self._terrain_height_at(self.root_states[:, :2])
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_feet_air_symmetry(self):
        air_time_diff = torch.abs(self.last_air_time_left - self.last_air_time_right)
        stride_symmetry = torch.exp(-air_time_diff / self.cfg.rewards.air_symmetry_sigma)

        return stride_symmetry

    def _reward_force_symmetry(self):
        contact_forces = self.contact_forces[:, self.feet_indices, 2]
        double_stance = (((self.phase >= LEFT_SWING_TARGET) & (self.phase < RIGHT_SWING_TARGET_START)) | ((self.phase >= RIGHT_SWING_TARGET_END) & (self.phase < 1.0))).float()

        force_left  = torch.tanh(contact_forces[:, 0] / 400.0)
        force_right = torch.tanh(contact_forces[:, 1] / 400.0)
        force_symmetry = double_stance * torch.exp(-torch.abs(force_left - force_right) / 0.2)
        return force_symmetry

    def _reward_power2(self):
        l = self.lower_dof_indices
        return torch.sum(torch.square(self.torques[:, l] * self.dof_vel[:, l]), dim=1)

    def _reward_power(self):
        l = self.lower_dof_indices
        return torch.sum(torch.abs(self.torques[:, l] * self.dof_vel[:, l]), dim=1)

    def _reward_torques(self):
        l = self.lower_dof_indices
        return torch.sum(torch.square(self.torques[:, l]), dim=1)

    def _reward_dof_vel(self):
        l = self.lower_dof_indices
        return torch.sum(torch.square(self.dof_vel[:, l]), dim=1)

    def _reward_dof_acc(self):
        l = self.lower_dof_indices
        return torch.sum(torch.square((self.last_dof_vel[:, l] - self.dof_vel[:, l]) / self.dt), dim=1)

    def _reward_dof_pos_limits(self):
        l = self.lower_dof_indices
        out_of_limits = -(self.dof_pos[:, l] - self.dof_pos_limits[l, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos[:, l] - self.dof_pos_limits[l, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        clipped = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self._action_track_max = torch.maximum(self._action_track_max, clipped.abs().max(dim=1).values)
        self._action_track_min = torch.minimum(self._action_track_min, clipped.min(dim=1).values)
        return super().step(actions)
