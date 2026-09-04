#!/usr/bin/env python3
"""GPIO stepper motor control node.

Adds software-only position moves for a lead-screw axis. The closed-loop motor
driver is still responsible for following STEP/DIR pulses; this node converts a
requested nut travel distance in millimeters into a fixed number of pulses.
"""

import json
import os
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from Hobot import GPIO


class MotorControlNode(Node):
    def __init__(self):
        super().__init__("motor_control_node")

        self.declare_parameter("pin_dir", 31)
        self.declare_parameter("pin_step", 32)
        self.declare_parameter("pin_en", 36)
        self.declare_parameter("pin_button", 37)
        self.declare_parameter("step_delay", 0.001)
        self.declare_parameter("auto_start", False)
        self.declare_parameter("lead_mm_per_rev", 10.0)
        self.declare_parameter("steps_per_rev", 400)
        self.declare_parameter("direction_sign", 1)
        self.declare_parameter("travel_min_mm", 0.0)
        self.declare_parameter("travel_max_mm", 400.0)
        self.declare_parameter(
            "position_state_path",
            "/home/sunrise/cable1/config/motor_position.json",
        )

        self.PIN_DIR = int(self.get_parameter("pin_dir").value)
        self.PIN_STEP = int(self.get_parameter("pin_step").value)
        self.PIN_EN = int(self.get_parameter("pin_en").value)
        self.PIN_BUTTON = int(self.get_parameter("pin_button").value)
        self.base_step_delay = max(0.0001, float(self.get_parameter("step_delay").value))
        self.step_delay = self.base_step_delay
        self.speed_percent = 100.0
        self.auto_start = bool(self.get_parameter("auto_start").value)
        self.lead_mm_per_rev = float(self.get_parameter("lead_mm_per_rev").value)
        self.steps_per_rev = max(1, int(self.get_parameter("steps_per_rev").value))
        self.direction_sign = 1 if int(self.get_parameter("direction_sign").value) >= 0 else -1
        self.travel_min_mm = float(self.get_parameter("travel_min_mm").value)
        self.travel_max_mm = float(self.get_parameter("travel_max_mm").value)
        self.position_state_path = str(
            self.get_parameter("position_state_path").value
        )
        if self.travel_max_mm <= self.travel_min_mm:
            self.travel_max_mm = self.travel_min_mm + 400.0

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.PIN_DIR, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.PIN_STEP, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.PIN_EN, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.PIN_BUTTON, GPIO.IN)
        time.sleep(0.1)

        self.state_lock = threading.RLock()
        self.motor_running = False
        self.motor_thread = None
        self.stop_thread = False
        self.button_pressed = False
        self.motor_direction = "forward"
        self.motion_mode = "idle"
        self.software_position_mm = self.load_software_position()
        self.target_distance_mm = 0.0
        self.target_steps = 0
        self.completed_steps = 0
        self.last_command = ""
        self.last_error = ""

        self.control_sub = self.create_subscription(
            String, "/motor/control", self.control_callback, 10
        )

        self.status_pub = self.create_publisher(String, "/motor/status", 10)
        self.direction_pub = self.create_publisher(String, "/motor/direction", 10)
        self.motion_status_pub = self.create_publisher(String, "/motor/motion_status", 10)
        self.software_position_pub = self.create_publisher(
            Float32, "/motor/software_position_mm", 10
        )
        self.status_timer = self.create_timer(0.5, self.publish_motion_status)

        self.get_logger().info("Motor control node started")
        self.get_logger().info(
            "GPIO: DIR=%s STEP=%s EN=%s BUTTON=%s, lead=%.4f mm/rev, "
            "steps_per_rev=%d, mm_per_step=%.6f, travel=%.1f..%.1f mm, "
            "base_step_delay=%.6f s"
            % (
                self.PIN_DIR,
                self.PIN_STEP,
                self.PIN_EN,
                self.PIN_BUTTON,
                self.lead_mm_per_rev,
                self.steps_per_rev,
                self.mm_per_step,
                self.travel_min_mm,
                self.travel_max_mm,
                self.base_step_delay,
            )
        )

        self.button_thread = threading.Thread(target=self.button_monitor, daemon=True)
        self.button_thread.start()

        if self.auto_start:
            self.start_motor()
        else:
            self.get_logger().info("Motor standby")
            self.publish_legacy_status("stopped")
            self.publish_motion_status()

    @property
    def mm_per_step(self):
        return self.lead_mm_per_rev / float(self.steps_per_rev)

    def load_software_position(self):
        try:
            with open(self.position_state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            position = float(data.get("position_mm", 0.0))
            position = max(self.travel_min_mm, min(self.travel_max_mm, position))
            self.get_logger().info(
                f"Loaded software position: {position:.4f} mm"
            )
            return position
        except FileNotFoundError:
            return 0.0
        except Exception as exc:
            self.get_logger().warn(f"Failed to load software position: {exc}")
            return 0.0

    def save_software_position(self, position=None):
        try:
            if position is None:
                with self.state_lock:
                    position = self.software_position_mm
            directory = os.path.dirname(self.position_state_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            tmp_path = self.position_state_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "position_mm": round(float(position), 6),
                        "updated_at": time.time(),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            os.replace(tmp_path, self.position_state_path)
        except Exception as exc:
            self.get_logger().warn(f"Failed to save software position: {exc}")

    @property
    def travel_span_mm(self):
        return self.travel_max_mm - self.travel_min_mm

    def position_percent(self):
        if self.travel_span_mm <= 0:
            return 0.0
        pct = (self.software_position_mm - self.travel_min_mm) / self.travel_span_mm
        return max(0.0, min(1.0, pct))

    def next_position_mm(self, direction):
        delta = self.mm_per_step * (1 if direction == "forward" else -1)
        delta *= self.direction_sign
        return self.software_position_mm + delta

    def is_position_allowed(self, position_mm):
        eps = self.mm_per_step * 0.51
        return self.travel_min_mm - eps <= position_mm <= self.travel_max_mm + eps

    def set_speed_percent(self, speed_percent=None):
        if speed_percent is None:
            return
        try:
            value = float(speed_percent)
        except Exception:
            self.last_error = f"invalid speed_percent: {speed_percent}"
            self.publish_motion_status()
            return
        value = max(10.0, min(100.0, value))
        with self.state_lock:
            self.speed_percent = value
            self.step_delay = self.base_step_delay / (value / 100.0)

    def effective_step_delay(self):
        with self.state_lock:
            return self.step_delay

    def set_direction_pin(self, direction):
        if direction == "forward":
            GPIO.output(self.PIN_DIR, GPIO.LOW)
        else:
            GPIO.output(self.PIN_DIR, GPIO.HIGH)

    def enable_motor(self):
        GPIO.output(self.PIN_EN, GPIO.LOW)
        time.sleep(0.01)

    def disable_motor(self):
        GPIO.output(self.PIN_EN, GPIO.HIGH)
        time.sleep(0.01)
        GPIO.output(self.PIN_STEP, GPIO.LOW)

    def step_once(self, direction):
        delay = self.effective_step_delay()
        GPIO.output(self.PIN_STEP, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(self.PIN_STEP, GPIO.LOW)
        time.sleep(delay)

        with self.state_lock:
            self.software_position_mm = self.next_position_mm(direction)
            self.completed_steps += 1
            position = self.software_position_mm
            should_save = self.completed_steps % 100 == 0
        if should_save:
            self.save_software_position(position)

    def motor_loop(self):
        with self.state_lock:
            direction = self.motor_direction
            self.motion_mode = "continuous"
            self.completed_steps = 0
            self.target_steps = 0
            self.target_distance_mm = 0.0

        self.set_direction_pin(direction)
        self.get_logger().info(f"Motor continuous run: {direction}")
        time.sleep(0.01)
        self.enable_motor()

        while not self.stop_thread:
            with self.state_lock:
                allowed = self.is_position_allowed(self.next_position_mm(direction))
            if not allowed:
                with self.state_lock:
                    self.last_error = "software travel limit reached"
                self.get_logger().warn("Software travel limit reached; stopping motor")
                break
            self.step_once(direction)

        self.disable_motor()
        self.save_software_position()
        with self.state_lock:
            self.motor_running = False
            self.motion_mode = "idle"
        self.publish_legacy_status("stopped")
        self.publish_motion_status()
        self.get_logger().info("Motor stopped")

    def move_loop(self, distance_mm, steps, direction):
        with self.state_lock:
            self.motion_mode = "move"
            self.motor_direction = direction
            self.target_distance_mm = float(distance_mm)
            self.target_steps = int(steps)
            self.completed_steps = 0

        self.set_direction_pin(direction)
        self.get_logger().info(
            "Motor move: %.4f mm, %d steps, direction=%s"
            % (distance_mm, steps, direction)
        )
        time.sleep(0.01)
        self.enable_motor()

        for _ in range(steps):
            if self.stop_thread:
                break
            with self.state_lock:
                allowed = self.is_position_allowed(self.next_position_mm(direction))
            if not allowed:
                with self.state_lock:
                    self.last_error = "software travel limit reached"
                self.get_logger().warn("Software travel limit reached during move")
                break
            self.step_once(direction)

        interrupted = self.stop_thread
        self.disable_motor()
        self.save_software_position()
        with self.state_lock:
            self.motor_running = False
            self.motion_mode = "idle"
            if interrupted:
                self.last_error = "move interrupted"
            elif self.last_error != "software travel limit reached":
                self.last_error = ""

        self.publish_legacy_status("stopped")
        self.publish_motion_status()
        if interrupted:
            self.get_logger().warn("Motor move interrupted")
        else:
            self.get_logger().info("Motor move complete")

    def publish_legacy_status(self, status):
        msg = String()
        msg.data = status
        try:
            self.status_pub.publish(msg)
        except Exception:
            pass

    def publish_direction(self):
        msg = String()
        msg.data = self.motor_direction
        self.direction_pub.publish(msg)

    def publish_motion_status(self):
        with self.state_lock:
            moving = bool(self.motor_running)
            payload = {
                "status": "running" if moving else "stopped",
                "mode": self.motion_mode,
                "direction": self.motor_direction,
                "position_mm": round(self.software_position_mm, 4),
                "position_percent": round(self.position_percent(), 4),
                "travel_min_mm": self.travel_min_mm,
                "travel_max_mm": self.travel_max_mm,
                "travel_span_mm": self.travel_span_mm,
                "lead_mm_per_rev": self.lead_mm_per_rev,
                "steps_per_rev": self.steps_per_rev,
                "mm_per_step": round(self.mm_per_step, 8),
                "speed_percent": round(self.speed_percent, 2),
                "step_delay": round(self.step_delay, 8),
                "base_step_delay": round(self.base_step_delay, 8),
                "target_distance_mm": round(self.target_distance_mm, 4),
                "target_steps": self.target_steps,
                "completed_steps": self.completed_steps,
                "progress": (
                    round(self.completed_steps / self.target_steps, 4)
                    if self.target_steps
                    else None
                ),
                "last_command": self.last_command,
                "last_error": self.last_error,
            }
            position = self.software_position_mm

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        try:
            self.motion_status_pub.publish(msg)
            self.software_position_pub.publish(Float32(data=float(position)))
        except Exception:
            pass

    def stop_motor(self):
        thread = None
        with self.state_lock:
            if self.motor_running:
                self.stop_thread = True
                thread = self.motor_thread

        if thread:
            thread.join(timeout=2.0)

        with self.state_lock:
            self.motor_running = False
            self.motion_mode = "idle"

        self.disable_motor()
        self.save_software_position()
        GPIO.output(self.PIN_DIR, GPIO.LOW)
        self.publish_legacy_status("stopped")
        self.publish_motion_status()

    def start_motor(self, speed_percent=None):
        self.set_speed_percent(speed_percent)
        with self.state_lock:
            if self.motor_running:
                return
            self.motor_running = True
            self.stop_thread = False
            self.last_command = "start"
            self.last_error = ""
            self.motor_thread = threading.Thread(target=self.motor_loop, daemon=True)
            self.motor_thread.start()

        self.publish_legacy_status("running")
        self.publish_direction()
        self.publish_motion_status()

    def start_move(self, distance_mm, speed_percent=None):
        self.set_speed_percent(speed_percent)
        try:
            distance_mm = float(distance_mm)
        except Exception:
            self.last_error = f"invalid distance: {distance_mm}"
            self.publish_motion_status()
            return False, self.last_error

        if abs(distance_mm) <= 0:
            return False, "distance must be non-zero"
        with self.state_lock:
            target_position = self.software_position_mm + distance_mm
        if not self.is_position_allowed(target_position):
            return (
                False,
                "target %.3f mm exceeds software travel %.3f..%.3f mm"
                % (target_position, self.travel_min_mm, self.travel_max_mm),
            )

        steps = int(round(abs(distance_mm) / self.mm_per_step))
        if steps <= 0:
            return False, "distance is smaller than one step"

        self.stop_motor()
        direction = "forward" if distance_mm > 0 else "reverse"

        with self.state_lock:
            self.motor_running = True
            self.stop_thread = False
            self.last_command = f"move_mm:{distance_mm:g}"
            self.last_error = ""
            self.motor_thread = threading.Thread(
                target=self.move_loop,
                args=(distance_mm, steps, direction),
                daemon=True,
            )
            self.motor_thread.start()

        self.publish_legacy_status("running")
        self.publish_direction()
        self.publish_motion_status()
        return True, f"moving {distance_mm:g} mm ({steps} steps)"

    def start_move_to(self, target_position_mm, speed_percent=None):
        try:
            target_position_mm = float(target_position_mm)
        except Exception:
            self.last_error = f"invalid target position: {target_position_mm}"
            self.publish_motion_status()
            return False, self.last_error

        if not self.is_position_allowed(target_position_mm):
            return (
                False,
                "target %.3f mm exceeds software travel %.3f..%.3f mm"
                % (target_position_mm, self.travel_min_mm, self.travel_max_mm),
            )

        with self.state_lock:
            distance_mm = target_position_mm - self.software_position_mm
        return self.start_move(distance_mm, speed_percent=speed_percent)

    def set_software_position(self, position_mm=0.0):
        position_mm = max(self.travel_min_mm, min(self.travel_max_mm, float(position_mm)))
        with self.state_lock:
            self.software_position_mm = position_mm
            self.completed_steps = 0
            self.target_steps = 0
            self.target_distance_mm = 0.0
            self.last_command = f"set_position_mm:{position_mm:g}"
            self.last_error = ""
            position = self.software_position_mm
        self.save_software_position(position)
        self.publish_motion_status()

    def button_monitor(self):
        self.get_logger().info("Button monitor started")
        while True:
            button_state = GPIO.input(self.PIN_BUTTON)

            if button_state == GPIO.LOW and not self.button_pressed:
                self.button_pressed = True
                time.sleep(0.05)

                if self.motor_running:
                    self.get_logger().info("Button pressed: stop motor")
                    self.stop_motor()
                else:
                    self.get_logger().info("Button pressed: start motor")
                    self.start_motor()

            elif button_state == GPIO.HIGH and self.button_pressed:
                self.button_pressed = False

            time.sleep(0.01)

    def parse_command_options(self, command):
        parts = [part.strip() for part in command.split(";") if part.strip()]
        base = parts[0] if parts else ""
        options = {}
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            options[key.strip().lower()] = value.strip()
        speed_percent = (
            options.get("speed_percent")
            or options.get("speed")
            or options.get("speed_pct")
        )
        return base, speed_percent

    def control_callback(self, msg):
        command = (msg.data or "").strip()
        base_command, speed_percent = self.parse_command_options(command)
        lower = base_command.lower()
        self.get_logger().info(f"Motor command: {command}")

        if lower == "start":
            self.start_motor(speed_percent=speed_percent)
        elif lower == "stop":
            self.last_command = "stop"
            self.stop_motor()
        elif lower == "forward":
            self.motor_direction = "forward"
            self.stop_motor()
            time.sleep(0.1)
            self.start_motor(speed_percent=speed_percent)
        elif lower == "reverse":
            self.motor_direction = "reverse"
            self.stop_motor()
            time.sleep(0.1)
            self.start_motor(speed_percent=speed_percent)
        elif lower.startswith("move_mm:") or lower.startswith("move:"):
            distance = base_command.split(":", 1)[1]
            ok, message = self.start_move(distance, speed_percent=speed_percent)
            if not ok:
                self.get_logger().warn(message)
        elif lower.startswith("move_to_mm:") or lower.startswith("move_to:"):
            target = base_command.split(":", 1)[1]
            ok, message = self.start_move_to(target, speed_percent=speed_percent)
            if not ok:
                self.get_logger().warn(message)
        elif lower in ("zero_position", "zero_position_mm"):
            self.set_software_position(0.0)
        elif lower.startswith("set_position_mm:"):
            try:
                self.set_software_position(float(command.split(":", 1)[1]))
            except ValueError:
                self.last_error = f"invalid set_position_mm command: {command}"
                self.publish_motion_status()
        else:
            self.last_error = f"unknown command: {command}"
            self.get_logger().warn(self.last_error)
            self.publish_motion_status()

    def destroy_node(self):
        self.stop_motor()
        time.sleep(0.5)
        GPIO.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
