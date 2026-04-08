#!/usr/bin/env python3
"""
数据采集客户端 - 双格式版本（交互式）
仿照 integrated_data_collection.py 的数据处理方式和交互流程

功能：
1. 启动前重置RealSense相机
2. 用户确认后开始采集
3. 采集完成后询问是否保存
4. 支持多轮采集或退出

数据格式（双格式保存，后续处理时可灵活选择）：

=== 格式1: 任务空间（推荐用于模仿学习）===
- env_qpos_proprioception (13维): 末端位姿(7) + 手部关节(6)
  * 末端位姿: [x, y, z, qw, qx, qy, qz] (位置3 + 四元数4)
  * 手部关节: [finger1, finger2, finger3, finger4, finger5, finger6]
- action (13维): 相对末端位姿变化(7) + 相对手部关节变化(6)
  * 相对变化 = 当前帧 - 前一帧 (第一帧为全0)

=== 格式2: 关节空间（备用格式）===
- joint_space_qpos (12维): 机械臂关节(6) + 手部关节(6)
- arm_qpos (6维): 机械臂6个关节角度
- ee_pose_absolute (7维): 末端位姿绝对值

=== 其他数据 ===
- gello_agent_joint (6维): Gello主从臂关节角度
- umi_action (6维): DexUMI手套角度
- force (6维): Inspire Hand力觉数据
- color/depth/cloud: 相机数据（如果可用）
"""
import json
import socket
import time
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import h5py
import multiprocessing as mp
import cv2
from multi_realsense import MultiRealSense


def camera_process(camera_queue, response_queue):
    """Process to handle camera capture in isolation."""
    try:
        camera_context = MultiRealSense(use_front_cam=True, 
                                        use_right_cam=False,
                                        front_num_points=4096)
        print("[DEBUG] Camera process initialized successfully.")
        
        while True:
            # Wait for request from main process
            camera_queue.get()
            
            try:
                cam_dict = camera_context()
                response_queue.put(cam_dict)
            except Exception as e:
                response_queue.put(None)  # Signal failure
                print(f"[ERROR] Camera capture in process failed: {e}")
    except Exception as e:
        print(f"[ERROR] Camera initialization in process failed: {e}")
        response_queue.put(None)  # Signal failure to main process


@dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 5000
    out_dir: str = "/home/raids/Desktop/gello_software_v1/demos_collection/gello_umi_drill_test_20251122_integrated"
    hz: int = 25
    max_seconds: Optional[int] = None
    length: int = 700  # 单次采集的固定长度（帧数）
    verbose: bool = True


class TeleopCollector:
    def __init__(self, args: Args):
        self.args = args
        self.addr = (args.host, args.port)
        self.dt = 1.0 / args.hz
        self.save_dir = Path(args.out_dir).expanduser()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.args.verbose:
            print(f"[collector] save_dir: {self.save_dir}")
            print(f"[collector] will connect to {self.addr}")

        # 相机相关
        self.camera_ok = False
        self.camera_queue = None
        self.response_queue = None
        self.camera_process = None
        
        # 重置RealSense相机
        self._cleanup_realsense()
        
        # 尝试初始化 RealSense 多相机进程
        self._init_camera()
    
    def _cleanup_realsense(self):
        """清理RealSense相关进程"""
        print("\n[collector] Cleaning up RealSense processes...")
        os.system("pkill -9 -f 'realsense' 2>/dev/null")
        os.system("pkill -9 -f 'rs-' 2>/dev/null")
        os.system("pkill -9 -f 'multi_realsense' 2>/dev/null")
        
        # 尝试释放USB设备
        try:
            result = subprocess.run(
                "lsusb | grep -i 'Intel'",
                shell=True,
                capture_output=True,
                text=True
            )
            if result.stdout:
                print("[collector] Found Intel RealSense devices, attempting reset...")
                time.sleep(2)
        except:
            pass
        
        print("[collector] RealSense cleanup completed")
    
    def _init_camera(self):
        """初始化相机进程"""
        try:
            import pyrealsense2 as _rs  # noqa: F401
            ctx = _rs.context()  # type: ignore
            devices = ctx.query_devices()  # type: ignore
            if len(devices) > 0:
                print(f"[collector] Found {len(devices)} RealSense device(s)")
                self.camera_queue = mp.Queue()
                self.response_queue = mp.Queue()
                self.camera_process = mp.Process(target=camera_process, args=(self.camera_queue, self.response_queue))
                self.camera_process.start()
                time.sleep(3)  # 等待相机初始化
                
                # 健康检查
                print("[collector] Testing camera health...")
                try:
                    self.camera_queue.put("test", timeout=1)
                    result = self.response_queue.get(timeout=3)
                    if result is None:
                        print("[collector] Camera health check failed")
                        self.camera_ok = False
                        if self.camera_process is not None:
                            self.camera_process.terminate()
                            self.camera_process = None
                    else:
                        print("[collector] Camera initialized successfully")
                        self.camera_ok = True
                except Exception as e:
                    print(f"[collector] Camera health check failed: {e}")
                    self.camera_ok = False
                    if self.camera_process is not None:
                        self.camera_process.terminate()
                        self.camera_process = None
            else:
                print("[WARNING] No RealSense devices found. Skipping camera initialization.")
                self.camera_ok = False
        except Exception as e:
            self.camera_ok = False
            print(f"[collector] camera not available: {e}; continue without camera")

    def _connect(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if self.args.verbose:
            print(f"[collector] connecting to {self.addr} ...")
        sock.connect(self.addr)
        sock.settimeout(1.0)
        if self.args.verbose:
            print("[collector] connected")
        return sock

    def _send(self, sock, obj):
        data = (json.dumps(obj) + "\n").encode("utf-8")
        sock.sendall(data)

    def _recv(self, sock):
        buf = b""
        while True:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                if self.args.verbose:
                    print("[collector] recv timeout; retrying ...")
                return None
            if not chunk:
                raise RuntimeError("server disconnected")
            buf += chunk
            if b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    msg = json.loads(line.decode("utf-8"))
                    return msg
                except Exception as e:
                    if self.args.verbose:
                        print(f"[collector] json decode failed: {e}")
                    return None

    def run(self):
        """主运行循环 - 交互式多轮采集"""
        sock = self._connect()
        
        try:
            demo_count = 0
            
            print("\n" + "="*60)
            print("交互式数据采集系统就绪")
            print("="*60)
            
            while True:
                print(f"\n{'='*60}")
                print(f"准备采集第 {demo_count + 1} 条示教数据")
                print(f"{'='*60}")
                print("命令:")
                print("  [Enter] - 开始采集")
                print("  'q' - 退出程序")
                
                cmd = input("\n请输入命令: ").strip().lower()
                
                if cmd == 'q':
                    print("\n退出程序...")
                    break
                elif cmd == '':
                    # 开始采集
                    demo_count += 1
                    try:
                        result = self._collect_one_demo(sock, demo_count)
                        if result is not None:
                            print(f"\n✓ 成功保存第 {demo_count} 条数据: {result}")
                        else:
                            demo_count -= 1  # 未保存则不计数
                            print("\n数据未保存")
                    except KeyboardInterrupt:
                        print("\n\n[INFO] 采集被中断")
                        demo_count -= 1
                    except Exception as e:
                        print(f"\n[ERROR] 采集失败: {e}")
                        import traceback
                        traceback.print_exc()
                        demo_count -= 1
                else:
                    print("未知命令，请重新输入")
            
            print(f"\n总共采集了 {demo_count} 条示教数据")
            
        finally:
            sock.close()
            # 关闭相机进程
            try:
                if self.camera_ok and self.camera_process is not None:
                    self.camera_process.terminate()
                    self.camera_process.join(timeout=2.0)
            except Exception:
                pass
    
    def _collect_one_demo(self, sock, demo_number: int):
        """采集一条示教数据"""
        print(f"\n[collector] 开始采集 demo {demo_number}...")
        print(f"[collector] 长度: {self.args.length} 帧 @ {self.args.hz} Hz")
        
        # 重置数据缓存
        color_array = []
        depth_array = []
        cloud_array = []
        
        # === 格式1: 任务空间 (末端位姿 + 手部关节) ===
        env_qpos_array = []  # 末端位姿(7) + 手部关节(6) = 13维
        action_array = []    # 相对末端位姿变化(7) + 相对手部关节变化(6) = 13维
        
        # === 格式2: 关节空间 (机械臂关节 + 手部关节) ===
        arm_qpos_array = []  # 机械臂关节(6)
        ee_pose_array = []   # 末端位姿(7) - 绝对值
        
        # === 其他数据 ===
        gello_agent_array = []  # (6,)
        umi_action_array = []   # (6,)
        force_array = []        # (6,)
        
        # 用于计算相对动作
        prev_ee = None
        prev_handqpos = None
        
        # 采集循环
        for i in range(self.args.length):
                start = time.time()
                self._send(sock, {"type": "GET_STATE"})
                resp = self._recv(sock)
                if resp is None:
                    continue
                if not resp.get("ok", False):
                    if self.args.verbose:
                        print(f"[collector] server responded not ok: {resp}")
                    continue

                # 相机请求
                if self.camera_ok and self.camera_queue is not None and self.response_queue is not None:
                    try:
                        self.camera_queue.put("request")
                        cam_dict = self.response_queue.get(timeout=0.5)
                        if cam_dict is not None:
                            target_size = (224, 224)
                            color_resized = cv2.resize(cam_dict['color'], target_size, interpolation=cv2.INTER_LINEAR)
                            depth_resized = cv2.resize(cam_dict['depth'], target_size, interpolation=cv2.INTER_LINEAR)
                            color_array.append(color_resized)
                            depth_array.append(depth_resized)
                            cloud_array.append(cam_dict['point_cloud'])
                            if self.args.verbose and (i % 20 == 0):
                                print(f"[collector] step {i}: camera frame appended")
                        else:
                            self.camera_ok = False
                            if self.args.verbose:
                                print(f"[collector] step {i}: camera returned None; disabling camera")
                    except Exception as e:
                        self.camera_ok = False
                        if self.args.verbose:
                            print(f"[collector] step {i}: camera exception ({e}); disabling camera")

                # 解析服务器返回的数据
                armqpos = np.array(resp["armqpos6"], dtype=float)
                handqpos = np.array(resp["handqpos6"], dtype=float)
                ee = np.array(resp["armee7"], dtype=float)
                gelloaction = np.array(resp["gelloaction6"], dtype=float)
                umiaction = np.array(resp["umiaction6"], dtype=float)
                force = np.array(resp["force6"], dtype=float)

                # === 格式1: 任务空间 (末端位姿 + 手部关节) ===
                # 状态：末端位姿 (7维) + 手部关节 (6维) = 13维
                env_qpos = np.concatenate([ee, handqpos])
                env_qpos_array.append(env_qpos)
                
                # 动作：相对末端位姿 + 相对手部关节 = 13维
                if prev_ee is not None and prev_handqpos is not None:
                    rel_ee = ee - prev_ee
                    rel_hand = handqpos - prev_handqpos
                    action = np.concatenate([rel_ee, rel_hand])
                else:
                    action = np.zeros(13)  # 第一帧设为0
                action_array.append(action)
                
                # 更新prev
                prev_ee = ee.copy()
                prev_handqpos = handqpos.copy()
                
                # === 格式2: 关节空间 (机械臂关节 + 手部关节) ===
                arm_qpos_array.append(armqpos)  # 机械臂关节(6)
                ee_pose_array.append(ee)        # 末端位姿(7) - 绝对值
                
                # === 其他数据 ===
                gello_agent_array.append(gelloaction)
                umi_action_array.append(umiaction)
                force_array.append(force)
                
                if self.args.verbose and (i % 10 == 0):
                    print(f"[collector] step {i}/{self.args.length} | "
                          f"ee[0]={ee[0]:.3f} | hand[0]={handqpos[0]:.0f} | "
                          f"rel_ee[0]={action[0]:.3f}")

                elapsed = time.time() - start
                sleep = self.dt - elapsed
                if sleep > 0:
                    time.sleep(sleep)
        
        print(f"\n[collector] Demo {demo_number} 采集完成！")
        
        # 询问是否保存
        choice = input("\n保存这条数据吗? (y/n): ").strip().lower()
        if choice != 'y':
            print("[collector] 数据已丢弃")
            return None
        
        # 保存数据
        record_file = self._get_next_h5_path()
        if self.args.verbose:
            print(f"[collector] 正在保存数据到 {record_file}")
        
        self._save_h5(
            record_file,
            env_qpos_array,
            action_array,
            arm_qpos_array,
            ee_pose_array,
            gello_agent_array,
            umi_action_array,
            force_array,
            color_array,
            depth_array,
            cloud_array
        )
        
        return record_file
    
    def _get_next_h5_path(self):
        """获取下一个可用的HDF5文件名"""
        existing_files = [f for f in os.listdir(self.save_dir) if f.startswith("demo") and f.endswith(".h5")]
        max_num = 0
        for f in existing_files:
            m = re.match(r"demo(\d+)\.h5", f)
            if m:
                num = int(m.group(1))
                max_num = max(max_num, num)
        next_num = max_num + 1
        return self.save_dir / f"demo{next_num:02d}.h5"

    def _save_h5(self, filepath, env_qpos, action, arm_qpos, ee_pose, gello, umi, force, color, depth, cloud):
        """保存HDF5文件 - 同时保存两种数据格式"""
        discard_end = 1
        seq_length = len(action)
        
        # 格式1: 任务空间 (末端位姿 + 手部关节)
        env_qpos_np = np.array(env_qpos)[:seq_length]
        action_np = np.array(action)[:seq_length]
        
        # 格式2: 关节空间 (机械臂关节 + 手部关节)
        arm_qpos_np = np.array(arm_qpos)[:seq_length]
        ee_pose_np = np.array(ee_pose)[:seq_length]
        
        # 其他数据
        gello_np = np.array(gello)[:seq_length]
        umi_np = np.array(umi)[:seq_length]
        force_np = np.array(force)[:seq_length]
        
        with h5py.File(filepath, "w") as f:
            # === 格式1: 任务空间 (推荐用于模仿学习) ===
            f.create_dataset("env_qpos_proprioception", data=env_qpos_np[:-discard_end])
            f.create_dataset("action", data=action_np[:-discard_end])
            
            # === 格式2: 关节空间 (备用格式) ===
            f.create_dataset("arm_qpos", data=arm_qpos_np[:-discard_end])
            f.create_dataset("ee_pose_absolute", data=ee_pose_np[:-discard_end])
            
            # 组合格式2的完整状态 (机械臂关节 + 手部关节)
            hand_joints = env_qpos_np[:, 7:]  # 从env_qpos中提取手部关节
            joint_space_qpos = np.concatenate([arm_qpos_np, hand_joints], axis=1)
            f.create_dataset("joint_space_qpos", data=joint_space_qpos[:-discard_end])
            
            # === 其他传感器数据 ===
            f.create_dataset("gello_agent_joint", data=gello_np[:-discard_end])
            f.create_dataset("umi_action", data=umi_np[:-discard_end])
            f.create_dataset("force", data=force_np[:-discard_end])
            
            # === 相机数据 ===
            if len(color) > 0:
                color_np = np.array(color)[:seq_length]
                depth_np = np.array(depth)[:seq_length]
                cloud_np = np.array(cloud)[:seq_length]
                f.create_dataset("color", data=color_np[:-discard_end])
                f.create_dataset("depth", data=depth_np[:-discard_end])
                f.create_dataset("cloud", data=cloud_np[:-discard_end])
            
            # === 元数据 ===
            f.attrs['format_version'] = '2.0'
            f.attrs['description'] = 'Dual format: task-space and joint-space'
            f.attrs['task_space_state'] = 'env_qpos_proprioception (13D: ee_pose + hand_joints)'
            f.attrs['task_space_action'] = 'action (13D: rel_ee + rel_hand)'
            f.attrs['joint_space_state'] = 'joint_space_qpos (12D: arm_joints + hand_joints)'
            f.attrs['joint_space_ee'] = 'ee_pose_absolute (7D: absolute ee pose)'
        
        if self.args.verbose:
            print(f"[collector] Saved {seq_length - discard_end} frames")
            print(f"[collector] === 格式1: 任务空间 ===")
            print(f"[collector]   - env_qpos_proprioception: {env_qpos_np.shape} (ee_pose + hand_joints)")
            print(f"[collector]   - action: {action_np.shape} (rel_ee + rel_hand)")
            print(f"[collector] === 格式2: 关节空间 ===")
            print(f"[collector]   - joint_space_qpos: {joint_space_qpos.shape} (arm_joints + hand_joints)")
            print(f"[collector]   - arm_qpos: {arm_qpos_np.shape} (arm_joints)")
            print(f"[collector]   - ee_pose_absolute: {ee_pose_np.shape} (absolute ee pose)")
            if len(color) > 0:
                print(f"[collector] === 相机数据 ===")
                print(f"[collector]   - color: {np.array(color).shape}")
                print(f"[collector]   - depth: {np.array(depth).shape}")
                print(f"[collector]   - cloud: {np.array(cloud).shape}")


def main(args: Args):
    TeleopCollector(args).run()


if __name__ == "__main__":
    try:
        import tyro
        main(tyro.cli(Args))
    except Exception:
        main(Args())