# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture the depth frames the policy actually receives, during a run.

The viewport shows the renderer's output, not the policy's input: by the time the image
reaches the network it has been clipped, scaled to [0, 1] and (from Level 3 on) had
noise added. This wrapper pulls the depth slice straight out of the observation vector,
so what gets written is exactly what the convolutional encoder sees.

The slice is located by asking the observation manager where the ``depth`` term sits in
the policy group, so it keeps working if terms are reordered or resized.
"""

from __future__ import annotations

import os

import gymnasium as gym
import torch


class DepthCaptureWrapper(gym.Wrapper):
    """Dump one environment's depth observations every ``interval_s`` for N episodes.

    Writes one PNG contact sheet per episode. Capture stops after ``num_episodes``; the
    wrapper then costs one tensor comparison per step.
    """

    def __init__(
        self,
        env: gym.Env,
        out_dir: str,
        num_episodes: int = 3,
        interval_s: float = 0.1,
        env_index: int = 0,
        group: str = "policy",
        term: str = "depth",
    ) -> None:
        super().__init__(env)
        self.out_dir = out_dir
        self.num_episodes = int(num_episodes)
        self.env_index = int(env_index)
        self.group = group
        self.term = term

        base = env.unwrapped
        self.height, self.width = self._resolve_image_shape(base)
        self.offset, self.length = self._resolve_slice(base)

        self.step_dt = base.step_dt
        self.interval_steps = max(1, round(interval_s / self.step_dt))
        self.sheet_columns = 10
        self.sheet_scale = 2

        self._step = 0
        self._episode = 0
        self._frames: list[torch.Tensor] = []
        self._done = False

        os.makedirs(out_dir, exist_ok=True)
        print(
            f"[INFO] Depth capture: env {self.env_index}, {self.num_episodes} episodes, "
            f"every {self.interval_steps} steps ({interval_s}s at dt={self.step_dt}), "
            f"slice [{self.offset}:{self.offset + self.length}] as {self.height}x{self.width} -> {out_dir}"
        )

    def _resolve_image_shape(self, base) -> tuple[int, int]:
        """Read the camera resolution off the sensor rather than hard-coding it."""
        for sensor in base.scene.sensors.values():
            cfg = getattr(sensor, "cfg", None)
            if cfg is not None and hasattr(cfg, "height") and hasattr(cfg, "width"):
                return int(cfg.height), int(cfg.width)
        raise RuntimeError("No camera sensor with height/width found in the scene.")

    def _resolve_slice(self, base) -> tuple[int, int]:
        """Offset and width of the depth term inside the concatenated policy group."""
        manager = base.observation_manager
        names = manager.active_terms[self.group]
        dims = manager.group_obs_term_dim[self.group]
        if self.term not in names:
            raise RuntimeError(f"Observation group '{self.group}' has no term '{self.term}' (has {names}).")
        index = names.index(self.term)
        offset = sum(int(torch.tensor(d).prod()) for d in dims[:index])
        return offset, int(torch.tensor(dims[index]).prod())

    def _record(self, obs) -> None:
        group_obs = obs[self.group] if isinstance(obs, dict) else obs
        frame = group_obs[self.env_index, self.offset : self.offset + self.length]
        self._frames.append(frame.detach().clone().reshape(self.height, self.width).cpu())

    def _flush(self) -> None:
        if not self._frames:
            return
        path = os.path.join(self.out_dir, f"episode_{self._episode:02d}.png")
        self._write_contact_sheet(self._frames, path)
        print(f"[INFO] Depth capture: wrote {len(self._frames)} frames -> {path}")
        self._frames = []

    def _write_contact_sheet(self, frames: list[torch.Tensor], path: str) -> None:
        """Tile the frames left-to-right, top-to-bottom, each labelled with its time.

        Isaac Lab's ``save_images_to_file`` lays out a bare grid whose column count
        depends on the frame count, which leaves the reading order ambiguous. Here the
        column count is fixed and every tile carries its index and elapsed time, so a
        particular moment can be pointed at.
        """
        from PIL import Image, ImageDraw

        scale = self.sheet_scale
        cols = min(self.sheet_columns, len(frames))
        rows = (len(frames) + cols - 1) // cols
        tile_w, tile_h = self.width * scale, self.height * scale
        pad, label_h = 2, 11

        cell_w, cell_h = tile_w + pad, tile_h + label_h + pad
        sheet = Image.new("L", (cols * cell_w + pad, rows * cell_h + pad), color=64)
        draw = ImageDraw.Draw(sheet)

        for i, frame in enumerate(frames):
            array = (frame.clamp(0.0, 1.0) * 255).to(torch.uint8).numpy()
            tile = Image.fromarray(array, mode="L").resize((tile_w, tile_h), Image.NEAREST)
            x = pad + (i % cols) * cell_w
            y = pad + (i // cols) * cell_h
            sheet.paste(tile, (x, y + label_h))
            seconds = i * self.interval_steps * self.step_dt
            draw.text((x + 1, y), f"{i:02d}  {seconds:.1f}s", fill=255)

        sheet.save(path)

    def step(self, action):
        obs, rew, terminated, truncated, extras = self.env.step(action)
        if self._done:
            return obs, rew, terminated, truncated, extras

        if self._step % self.interval_steps == 0:
            self._record(obs)
        self._step += 1

        episode_over = bool(terminated[self.env_index]) or bool(truncated[self.env_index])
        if episode_over:
            self._flush()
            self._episode += 1
            self._step = 0
            if self._episode >= self.num_episodes:
                self._done = True
                print(f"[INFO] Depth capture: finished {self.num_episodes} episodes.")

        return obs, rew, terminated, truncated, extras
