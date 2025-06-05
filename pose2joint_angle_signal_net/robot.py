from rotation import DEG

class NAORobot:
    def __init__(self):
        self.joint_names = ["BP", "RK", "LK", "BT", "RMrot", "LMrot", "BLN", "RF", "LF", "BMN", "RSI", "LSI", "BUN", "RS", "LS", "RE", "LE", "RW", "LW"]

NAO_JOINT_LIMITS = [
    [DEG(-119.5), DEG(119.5)],  # LShoulderPitch
    [DEG(-18.0), DEG(76.0)],    # LShoulderRoll
    [DEG(-119.5), DEG(119.5)],  # LElbowYaw
    [DEG(-88.5), DEG(-2.0)],    # LElbowRoll
    [DEG(-104.5), DEG(104.5)],  # LWristYaw
    [DEG(-119.5), DEG(119.5)],  # RShoulderPitch
    [DEG(-76.0), DEG(18.0)],    # RShoulderRoll
    [DEG(-119.5), DEG(119.5)],  # RElbowYaw
    [DEG(2.0), DEG(88.5)],      # RElbowRoll
    [DEG(-104.5), DEG(104.5)],  # RWristYaw
    [DEG(-65.62), DEG(42.44)],  # LHipYawPitch
    [DEG(-21.74), DEG(45.29)],  # LHipRoll
    [DEG(-88.0), DEG(27.73)],   # LHipPitch
    [DEG(-5.29), DEG(121.04)],  # LKneePitch
    [DEG(-68.15), DEG(52.86)],  # LAnklePitch
    [DEG(-22.79), DEG(44.06)],  # LAnkleRoll
    [DEG(-45.29), DEG(21.74)],  # RHipRoll
    [DEG(-88.0), DEG(27.73)],   # RHipPitch
    [DEG(-5.9), DEG(121.47)],   # RKneePitch
    [DEG(-67.97), DEG(53.4)],   # RAnklePitch
    [DEG(-44.06), DEG(22.8)]    # RAnkleRoll
]