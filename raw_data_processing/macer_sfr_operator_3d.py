"""MACER3D 的离线 SFR 算子，采用作者给出的代码单位与热力学公式。

所有场按 (r, theta, phi) 排列。cool_rate 为正值的代码单位体积冷却率；
grav 为静态总径向加速度。默认连续时间极限，不沿用二维有限步长修正。
metallicity 按绝对质量分数解释；这是数据支持的口径，尚待源码最终确认。
"""

from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class SFR3DParameters:
    gamma: float = 5.0 / 3.0
    efficiency: float = 0.03
    temperature_ceiling: float = 4.e4
    number_density_floor: float = 1.0
    rotation_factor: float = 2.0 * np.pi
    gravitational_constant: float = 112.4766
    hydrogen_mass_code: float = 3.364939e-65
    boltzmann_code: float = 2.904099e-67
    length_unit_cm: float = 3.085678e21
    hydrogen_fraction: float = 0.7
    sfr_to_msun_per_yr: float = 0.025
    number_density_mode: str = "total_particles"
    temperature_mode: str = "metallicity"

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, (float, int)) and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{key} 必须有限且为正。")
        if self.gamma <= 1 or self.hydrogen_fraction >= 1 or self.efficiency > 1:
            raise ValueError("gamma、hydrogen_fraction 或 efficiency 超出物理范围。")
        if self.number_density_mode not in ("total_particles", "hydrogen_nuclei", "rho_over_mH"):
            raise ValueError("未知 number_density_mode。")
        if self.temperature_mode not in ("metallicity", "fixed_71.41"):
            raise ValueError("未知 temperature_mode。")


@dataclass(frozen=True)
class SFR3DResult:
    sfr_density: np.ndarray
    cell_sfr: np.ndarray
    total_sfr: float
    temperature: np.ndarray
    number_density: np.ndarray
    tau_cool: np.ndarray
    tau_free_fall: np.ndarray
    tau_rotation: np.ndarray
    tau_star: np.ndarray
    eligible_mask: np.ndarray
    limiter_code: np.ndarray
    parameters: dict


def calculate_macer_sfr_3d(*, density, pressure, metallicity, cooling_rate,
                           cell_volume, radius, radial_acceleration,
                           parameters=None):
    """返回代码 SFR 密度、逐格 SFR、总体积分与时标诊断。

    不读取监督字段 rhodotsf。cooling_rate=0 时冷却时间为无穷大，SFR=0；
    负冷却率被拒绝，调用方不能静默取绝对值。零密度格点不形成恒星。
    limiter_code: 0=阈值外，1=冷却，2=自由落体，3=旋转。
    """
    par = parameters or SFR3DParameters()
    par.validate()
    rho = np.asarray(density, dtype=np.float64)
    if rho.ndim != 3 or rho.size == 0:
        raise ValueError("density 必须是非空三维场 (r, theta, phi)。")

    def field(name, value, *, broadcast=False, positive=False, negative_ok=False):
        a = np.asarray(value, dtype=np.float64)
        if broadcast:
            a = np.broadcast_to(a, rho.shape)
        if a.shape != rho.shape or not np.all(np.isfinite(a)):
            raise ValueError(f"{name} 必须与 density 同形且全部有限。")
        if (positive and np.any(a <= 0)) or (not negative_ok and np.any(a < 0)):
            raise ValueError(f"{name} 包含非法的负值或零值。")
        return a

    rho = field("density", rho)
    p = field("pressure", pressure)
    z = field("metallicity", metallicity)
    if np.any(z > 1):
        raise ValueError("metallicity 应为 [0,1] 内的绝对质量分数。")
    cooling = field("cooling_rate", cooling_rate)
    volume = field("cell_volume", cell_volume, positive=True)
    r = field("radius", radius, broadcast=True, positive=True)
    grav = field("radial_acceleration", radial_acceleration, broadcast=True, negative_ok=True)
    mu_inverse = 2 * par.hydrogen_fraction + .75 * (1 - par.hydrogen_fraction - z) + .5 * z
    # 高金属丰度可能使固定 X 假设失效；此处忠实保留作者公式，并在元数据记录。
    mu = 1 / mu_inverse
    coefficient = mu * par.hydrogen_mass_code / par.boltzmann_code
    if par.temperature_mode == "fixed_71.41":
        coefficient = np.full_like(rho, 71.41)
    temperature = np.divide(p * coefficient, rho, out=np.full_like(rho, np.inf), where=rho > 0)
    number_density = rho / (par.hydrogen_mass_code * par.length_unit_cm**3)
    if par.number_density_mode == "total_particles":
        number_density = number_density / mu
    elif par.number_density_mode == "hydrogen_nuclei":
        number_density = number_density * par.hydrogen_fraction
    tau_cool = np.divide(p / (par.gamma - 1), cooling,
                         out=np.full_like(rho, np.inf), where=cooling > 0)
    ff_squared = np.divide(3 * np.pi, 32 * par.gravitational_constant * rho,
                           out=np.full_like(rho, np.inf), where=rho > 0)
    tau_ff = np.sqrt(ff_squared)
    tau_rot = par.rotation_factor * np.sqrt(np.divide(
        r, np.abs(grav), out=np.full_like(rho, np.inf), where=grav != 0))
    tau_dyn = np.minimum(tau_ff, tau_rot)
    tau_star = np.maximum(tau_cool, tau_dyn)
    eligible = ((rho > 0) & (temperature < par.temperature_ceiling)
                & (number_density > par.number_density_floor))
    rate = np.divide(par.efficiency * rho, tau_star, out=np.zeros_like(rho),
                     where=eligible & np.isfinite(tau_star) & (tau_star > 0))
    cell_rate = rate * volume
    limiter = np.where(eligible, np.where(tau_cool >= tau_dyn, 1,
                        np.where(tau_ff <= tau_rot, 2, 3)), 0).astype(np.int8)
    metadata = asdict(par)
    metadata.update(operator_version=1, time_step_mode="continuous_limit",
                    metallicity_semantics="absolute_mass_fraction_data_supported",
                    cooling_semantics="positive_code_energy_per_volume_per_time",
                    gravity_semantics="static_radial_acceleration_author_confirmed",
                    composition_invalid_cell_count=int(np.count_nonzero(z > 1-par.hydrogen_fraction)))
    return SFR3DResult(rate, cell_rate, float(cell_rate.sum(dtype=np.float64)),
                       temperature, number_density, tau_cool, tau_ff, tau_rot,
                       tau_star, eligible, limiter, metadata)


def validate_gravity_geometry(snapshot, gravity_snapshot):
    """逐轴核对中心与边界，防止同形状但不同网格的静态引力错配。"""
    for kind in ("centers", "edges"):
        for a, b in zip(getattr(snapshot.geometry, kind), getattr(gravity_snapshot.geometry, kind)):
            if a.shape != b.shape or not np.array_equal(a, b):
                raise ValueError(f"静态引力与待计算 snapshot 的 {kind} 不一致。")
