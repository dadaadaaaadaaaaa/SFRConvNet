"""原生 (r,theta,phi) 网格上的三维守恒空间降采样。

momentum_x/y/z 是单元积分动量 M*v，不是动量密度。
velocity_x/y/z 是固定笛卡尔基底上的质量加权速度。
vel1/2/3 是同一速度在 coarse 中心球坐标基底上的投影，不能直接平均 fine 球坐标分量。
"""

from dataclasses import dataclass
import numpy as np

from raw_data_processing.coarsening_common import block_reduce_sum


def spherical_volume(edges):
    r, t, p = [np.asarray(e, dtype=np.float64) for e in edges]
    if any(e.ndim != 1 or len(e) < 2 or not np.isfinite(e).all() or np.any(np.diff(e) <= 0) for e in (r,t,p)):
        raise ValueError("三维边界必须有限、严格递增且为一维。")
    if r[0] <= 0 or t[0] < 0 or t[-1] > np.pi:
        raise ValueError("球坐标边界超出物理范围。")
    return np.diff(r**3)[:,None,None]/3 * (-np.diff(np.cos(t)))[None,:,None] * np.diff(p)[None,None,:]


def spherical_to_cartesian(vr, vt, vp, axes):
    """将每格局部球坐标速度转换到全域共用的笛卡尔基底。"""
    t, p = axes[1][None,:,None], axes[2][None,None,:]
    return (vr*np.sin(t)*np.cos(p)+vt*np.cos(t)*np.cos(p)-vp*np.sin(p),
            vr*np.sin(t)*np.sin(p)+vt*np.cos(t)*np.sin(p)+vp*np.cos(p),
            vr*np.cos(t)-vt*np.sin(t))


def cartesian_to_spherical(vx, vy, vz, axes):
    """投影到指定单元中心的球坐标基底，不进行第二次平均。"""
    t, p = axes[1][None,:,None], axes[2][None,None,:]
    return (vx*np.sin(t)*np.cos(p)+vy*np.sin(t)*np.sin(p)+vz*np.cos(t),
            vx*np.cos(t)*np.cos(p)+vy*np.cos(t)*np.sin(p)-vz*np.sin(t),
            -vx*np.sin(p)+vy*np.cos(p))


def derive_fields(fields, volume, axes, parameters, *, cartesian=False):
    """由守恒量生成冗余物理字段；温度和内能使用三维公式。"""
    f = dict(fields)
    rho, pressure, z = (f[k] for k in ('density','pressure','metallicity'))
    if np.any(rho <= 0):
        raise ValueError("当前三维预处理要求正密度，真空单元的温度约定尚未定义。")
    mu = 1/(2*parameters.hydrogen_fraction+.75*(1-parameters.hydrogen_fraction-z)+.5*z)
    f['gas_mass'] = rho*volume
    f['gas_energy'] = pressure/(parameters.gamma-1)  # 内能密度，非单元总内能。
    f['temperature'] = pressure/rho*mu*parameters.hydrogen_mass_code/parameters.boltzmann_code
    if parameters.temperature_mode == 'fixed_71.41':
        f['temperature'] = pressure/rho*71.41
    if cartesian:
        xyz = tuple(f['velocity_'+a] for a in 'xyz')
        for name, values in zip(('vel1','vel2','vel3'), cartesian_to_spherical(*xyz, axes)):
            f[name] = values
    else:
        xyz = spherical_to_cartesian(*(f[k] for k in ('vel1','vel2','vel3')), axes)
    for a, velocity in zip('xyz', xyz):
        f['velocity_'+a] = velocity
        f['momentum_'+a] = velocity*f['gas_mass']
    if 'cooling_rate_density' in f:
        f['cooling_time'] = np.divide(f['gas_energy'], f['cooling_rate_density'],
            out=np.full_like(rho, np.inf), where=f['cooling_rate_density'] > 0)
    return f


def validate_state(fields, volume, axes, edges, truth, parameters):
    """拒绝错误几何、未知字段、冗余字段不一致或非物理源状态。"""
    parameters.validate()
    shape = volume.shape
    if len(shape) != 3 or len(axes) != 3 or len(edges) != 3:
        raise ValueError("需要三维 (r,theta,phi) 数据。")
    expected_volume = spherical_volume(edges)
    if expected_volume.shape != shape or not np.isfinite(volume).all() or np.any(volume <= 0):
        raise ValueError("体积形状错误或非正/非有限。")
    if not np.allclose(volume, expected_volume, rtol=1e-10, atol=0):
        raise ValueError("存储体积与权威 cell faces 不一致。")
    for n, a, e in zip(shape, axes, edges):
        if a.shape != (n,) or len(e) != n+1 or not np.isfinite(a).all() or np.any(a <= e[:-1]) or np.any(a >= e[1:]):
            raise ValueError("单元中心与边界不一致。")
    required = {'density','pressure','metallicity','gas_energy','gas_mass','temperature',
                'vel1','vel2','vel3',*(f'velocity_{a}' for a in 'xyz'),*(f'momentum_{a}' for a in 'xyz')}
    supported = required | {'cooling_rate_density','cooling_time','radial_acceleration'}
    if required-set(fields) or set(fields)-supported:
        raise ValueError(f"缺失或未知三维字段：{required-set(fields)}, {set(fields)-supported}")
    for name, a in {**fields, 'sfr_density_true': truth}.items():
        a = np.asarray(a)
        finite = np.isfinite(a) | ((name == 'cooling_time') & np.isposinf(a))
        if a.shape != shape or not finite.all():
            raise ValueError(f"字段 {name} 形状错误或包含非法非有限值。")
    for name in ('density','pressure','gas_energy','gas_mass','temperature','metallicity','cooling_rate_density'):
        if name in fields and np.any(fields[name] < 0):
            raise ValueError(f"字段 {name} 不允许为负。")
    if np.any(truth < 0) or np.any(fields['metallicity'] > 1):
        raise ValueError("SFR 或绝对金属质量分数越界。")
    reconstructed = derive_fields(fields, volume, axes, parameters)
    for name in ('gas_mass','gas_energy','temperature','cooling_time',*(f'velocity_{a}' for a in 'xyz'),*(f'momentum_{a}' for a in 'xyz')):
        if name in fields and not np.allclose(fields[name], reconstructed[name], rtol=1e-10, atol=1e-12):
            raise ValueError(f"冗余字段 {name} 与主物理量不一致。")


@dataclass
class Spatial3DResult:
    fields: dict
    volume: np.ndarray
    axes: tuple
    edges: tuple
    truth: np.ndarray
    diagnostics: dict


def coarsen_spatial_3d(fields, volume, axes, edges, truth, parameters,
                       new_shape=(32,16,32), *, mode='direct', rtol=1e-10, atol=1e-12):
    """Direct 与 Residual 共用守恒聚合；Residual 额外聚合冷却率与有效径向引力。"""
    if mode not in ('direct','residual'):
        raise ValueError("mode 必须是 direct 或 residual。")
    if len(new_shape) != 3 or any(isinstance(n, bool) or int(n) != n or n <= 0 for n in new_shape):
        raise ValueError("new_shape 必须为三个正整数。")
    if not np.isfinite([rtol,atol]).all() or rtol < 0 or atol < 0:
        raise ValueError("守恒容差必须有限且非负。")
    validate_state(fields, volume, axes, edges, truth, parameters)
    shape = tuple(int(n) for n in new_shape)
    def total(a): return block_reduce_sum(a, shape)
    cv = total(volume)
    factors = tuple(n//c for n,c in zip(volume.shape,shape))
    ce = tuple(e[::k].copy() for e,k in zip(edges,factors))
    ca = (np.sqrt(ce[0][:-1]*ce[0][1:]), (ce[1][:-1]+ce[1][1:])/2, (ce[2][:-1]+ce[2][1:])/2)
    mass = fields['density']*volume
    cm = total(mass)
    cf = dict(density=cm/cv, pressure=total(fields['pressure']*volume)/cv,
              metallicity=total(fields['metallicity']*mass)/cm)
    for a in 'xyz':
        cf['velocity_'+a] = total(fields['momentum_'+a])/cm
    if mode == 'residual':
        for name in ('cooling_rate_density','radial_acceleration'):
            if name not in fields:
                raise ValueError(f"Residual 需要 fine 字段 {name}，请先生成 residual fine。")
        cf['cooling_rate_density'] = total(fields['cooling_rate_density']*volume)/cv
        # 这是有效径向标量，不是完整笛卡尔力矢量的守恒投影。
        cf['radial_acceleration'] = total(fields['radial_acceleration']*mass)/cm
    cf = derive_fields(cf, cv, ca, parameters, cartesian=True)
    ct = total(truth*volume)/cv
    validate_state(cf, cv, ca, ce, ct, parameters)
    diagnostics = {}
    def audit(name, fine_extensive, coarse_extensive):
        before, after = float(np.sum(fine_extensive)), float(np.sum(coarse_extensive))
        scale = float(np.sum(np.abs(fine_extensive)))
        error = abs(after-before)
        if error > atol+rtol*scale:
            raise ValueError(f"{name} 不守恒：fine={before}, coarse={after}")
        diagnostics[name] = dict(fine=before, coarse=after, absolute_error=error,
                                 scaled_relative_error=error/scale if scale else 0.)
    audit('volume',volume,cv)
    audit('gas_mass',mass,cm)
    audit('internal_energy',fields['gas_energy']*volume,cf['gas_energy']*cv)
    audit('metal_mass',fields['metallicity']*mass,cf['metallicity']*cm)
    audit('sfr',truth*volume,ct*cv)
    for a in 'xyz': audit('momentum_'+a,fields['momentum_'+a],cf['momentum_'+a])
    if mode == 'residual':
        audit('cooling_power',fields['cooling_rate_density']*volume,cf['cooling_rate_density']*cv)
        audit('radial_force_scalar',fields['radial_acceleration']*mass,cf['radial_acceleration']*cm)
    # 平均速度会损失已解析动能；差额是未解析速度弥散，不转入热能。
    ke_fine = total(.5*mass*sum(fields['velocity_'+a]**2 for a in 'xyz'))
    ke_coarse = .5*cm*sum(cf['velocity_'+a]**2 for a in 'xyz')
    if np.any(ke_coarse-ke_fine > atol+rtol*np.abs(ke_fine)):
        raise ValueError("粗粒化不应增加已解析动能。")
    diagnostics['unresolved_kinetic_energy'] = float(np.sum(ke_fine-ke_coarse))
    diagnostics['conservation_passed'] = True
    return Spatial3DResult(cf, cv, ca, ce, ct, diagnostics)
