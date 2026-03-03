import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time

# ====== 原有导入（保持不变）======
from lerobot.robots.lecarm import LecarmClient, LecarmClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.bi_so101_leader import BiSO101Leader, BiSO101LeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ====== 全局变量 ======
running = False
latest_action = {}          # 用于 GUI 显示的最新 action
print_log_enabled = True    # 控制是否打印日志到终端


# ====== 后台主循环（你的原逻辑 + 日志开关 + 数据共享）======
def run_teleop(use_dummy, fps, remote_ip):
    global running, latest_action, print_log_enabled

    if use_dummy:
        if print_log_enabled:
            print("🧪 USE_DUMMY mode enabled: robot will not connect, only print actions.")

    robot_config = LecarmClientConfig(remote_ip=remote_ip, id="my_lecarm")
    bi_cfg = BiSO101LeaderConfig(
        left_arm_port="/dev/lecarm_left",
        right_arm_port="/dev/lecarm_right",
        id="so101_leader_bi",
    )

    leader = BiSO101Leader(bi_cfg)
    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")
    keyboard = KeyboardTeleop(keyboard_config)
    robot = LecarmClient(robot_config)

    if not use_dummy:
        robot.connect()
    else:
        if print_log_enabled:
            print("🧪 robot.connect() skipped, only printing actions.")

    leader.connect()
    keyboard.connect()
    init_rerun(session_name="lecarm_teleop")

    if not robot.is_connected or not leader.is_connected or not keyboard.is_connected:
        if print_log_enabled:
            print("⚠️ 警告: 部分设备未连接,继续运行以debug！")

    while running:
        t0 = time.perf_counter()

        observation = robot.get_observation() if not use_dummy else {}
        arm_actions = leader.get_action()
        arm_actions = {f"arm_{k}": v for k, v in arm_actions.items()}
        keyboard_keys = keyboard.get_action()
        base_action = robot._from_keyboard_to_base_action(keyboard_keys)

        action = {**arm_actions, **base_action}
        log_rerun_data(observation, action)

        # 👇 关键：更新全局变量供 GUI 读取
        latest_action = action.copy()

        # 👇 控制是否打印日志
        if print_log_enabled:
            if use_dummy:
                print(f"[USE_DUMMY] action → {action}")
            else:
                print(f"Sent base_action → {base_action}")

        busy_wait(max(1.0 / fps - (time.perf_counter() - t0), 0.0))

    # 清理
    try:
        leader.disconnect()
        keyboard.disconnect()
        if not use_dummy:
            robot.disconnect()
        if print_log_enabled:
            print("⏹️ 设备已安全断开。")
    except Exception as e:
        if print_log_enabled:
            print(f"❌ 断开设备时出错: {e}")


# ====== GUI 应用 ======
class LeCarmGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LeCarm 遥操作控制面板")
        self.root.geometry("750x500")

        # 参数变量
        self.use_dummy_var = tk.BooleanVar(value=False)
        self.fps_var = tk.IntVar(value=30)
        self.ip_var = tk.StringVar(value="192.168.1.21")
        self.print_log_var = tk.BooleanVar(value=True)  # 新增：是否打印日志

        self.create_widgets()
        self.update_display()  # 启动自动刷新

    def create_widgets(self):
        # --- 参数设置区 ---
        param_frame = ttk.LabelFrame(self.root, text="运行参数", padding=10)
        param_frame.pack(fill="x", padx=10, pady=5)

        ttk.Checkbutton(param_frame, text="启用 DUMMY 模式", variable=self.use_dummy_var).pack(anchor="w")
        ttk.Checkbutton(param_frame, text="在终端打印日志消息", variable=self.print_log_var).pack(anchor="w", pady=(5, 0))

        ttk.Label(param_frame, text="FPS:").pack(anchor="w", pady=(5, 0))
        ttk.Spinbox(param_frame, from_=1, to=60, textvariable=self.fps_var, width=10).pack(anchor="w")

        ttk.Label(param_frame, text="Remote IP:").pack(anchor="w", pady=(5, 0))
        ttk.Entry(param_frame, textvariable=self.ip_var, width=20).pack(anchor="w")

        # --- 控制按钮 ---
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)
        self.start_btn = ttk.Button(btn_frame, text="▶ 启动", command=self.start_teleop)
        self.start_btn.pack(side="left", padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="⏹ 停止", command=self.stop_teleop, state="disabled")
        self.stop_btn.pack(side="left", padx=5)

        # --- 实时数据显示区 ---
        data_frame = ttk.LabelFrame(self.root, text="实时动作数据", padding=10)
        data_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # 左臂
        ttk.Label(data_frame, text="左臂 (°)").grid(row=0, column=0, sticky="w")
        self.left_labels = []
        left_names = ["肩_pan", "肩_lift", "肘_flex", "腕_flex", "腕_roll", "夹爪"]
        for i, name in enumerate(left_names):
            ttk.Label(data_frame, text=name).grid(row=i+1, column=0, sticky="w")
            lbl = ttk.Label(data_frame, text="0.0")
            lbl.grid(row=i+1, column=1, sticky="w", padx=(10, 20))
            self.left_labels.append(lbl)

        # 右臂
        ttk.Label(data_frame, text="右臂 (°)").grid(row=0, column=2, sticky="w")
        self.right_labels = []
        right_names = ["肩_pan", "肩_lift", "肘_flex", "腕_flex", "腕_roll", "夹爪"]
        for i, name in enumerate(right_names):
            ttk.Label(data_frame, text=name).grid(row=i+1, column=2, sticky="w")
            lbl = ttk.Label(data_frame, text="0.0")
            lbl.grid(row=i+1, column=3, sticky="w", padx=(10, 20))
            self.right_labels.append(lbl)

        # 底盘速度
        ttk.Label(data_frame, text="底盘速度").grid(row=0, column=4, sticky="w")
        self.base_x = ttk.Label(data_frame, text="x: 0.00")
        self.base_y = ttk.Label(data_frame, text="y: 0.00")
        self.base_t = ttk.Label(data_frame, text="θ: 0.00")
        self.base_x.grid(row=1, column=4, sticky="w", padx=(10, 0))
        self.base_y.grid(row=2, column=4, sticky="w", padx=(10, 0))
        self.base_t.grid(row=3, column=4, sticky="w", padx=(10, 0))

    def start_teleop(self):
        global running, print_log_enabled
        if running:
            return

        use_dummy = self.use_dummy_var.get()
        fps = self.fps_var.get()
        ip = self.ip_var.get()
        print_log_enabled = self.print_log_var.get()  # 更新全局日志开关

        if not (1 <= fps <= 60):
            messagebox.showerror("参数错误", "FPS 必须在 1~60 之间")
            return

        running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        thread = threading.Thread(
            target=run_teleop,
            args=(use_dummy, fps, ip),
            daemon=True
        )
        thread.start()

    def stop_teleop(self):
        global running
        running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def update_display(self):
        """每 100ms 从 latest_action 读取并更新 GUI"""
        action = latest_action

        # 左臂字段
        left_keys = [
            "arm_left_shoulder_pan.pos",
            "arm_left_shoulder_lift.pos",
            "arm_left_elbow_flex.pos",
            "arm_left_wrist_flex.pos",
            "arm_left_wrist_roll.pos",
            "arm_left_gripper.pos"
        ]
        for i, key in enumerate(left_keys):
            val = action.get(key, 0.0)
            self.left_labels[i].config(text=f"{val:.1f}")

        # 右臂字段
        right_keys = [
            "arm_right_shoulder_pan.pos",
            "arm_right_shoulder_lift.pos",
            "arm_right_elbow_flex.pos",
            "arm_right_wrist_flex.pos",
            "arm_right_wrist_roll.pos",
            "arm_right_gripper.pos"
        ]
        for i, key in enumerate(right_keys):
            val = action.get(key, 0.0)
            self.right_labels[i].config(text=f"{val:.1f}")

        # 底盘速度
        self.base_x.config(text=f"x: {action.get('x.vel', 0.0):.2f}")
        self.base_y.config(text=f"y: {action.get('y.vel', 0.0):.2f}")
        self.base_t.config(text=f"θ: {action.get('theta.vel', 0.0):.2f}")

        # 继续刷新
        self.root.after(100, self.update_display)


# ====== 修复拼写错误 + 启动 ======
if __name__ == "__main__":
    # 修正：KeyboardTeleop 拼写（你原代码是正确的，但上面我误写成 KeyboardTeleOp）
    # 确保导入正确（已正确）

    root = tk.Tk()
    app = LeCarmGUI(root)
    root.mainloop()