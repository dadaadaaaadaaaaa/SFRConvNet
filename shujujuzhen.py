from pyhdf.SD import SD, SDC
import numpy as np
import os

# 物理常量定义
LU = 3.08568025e21  # 长度单位 (cm)
TU = 3.15360000e16  # 时间单位 (s)
MU = 4.97230000e40  # 质量单位 (g)
Kb = 1.38064900e-16  # 玻尔兹曼常数 (erg/K)
gamma = 5 / 3  # 绝热指数

# 物理量中英文名称映射
PHYSICAL_QUANTITIES = {
    "v1": {"en": "Radial Velocity", "cn": "径向速度"},#km/s
    "v2": {"en": "Angular Velocity", "cn": "角向速度"},#km/s
    "v3": {"en": "Azimuthal Velocity", "cn": "方位角速度"},#km/s
    "density": {"en": "Gas Density", "cn": "气体密度"},#cm^-3
    "temperature": {"en": "Gas Temperature", "cn": "气体温度"},#K
    "gp_bg": {"en": "Gravitational Potential BG", "cn": "引力势背景"},
    "a1": {"en": "Acceleration Component 1", "cn": "加速度分量1"},
    "a2": {"en": "Acceleration Component 2", "cn": "加速度分量2"},
    "energy": {"en": "Internal Energy", "cn": "内能"},#数值93*cm^-3*K
    "sfr": {"en": "Star Formation Rate", "cn": "恒星形成率"},#M_sun/yr
    "mnewstar": {"en": "Mass of New Stars", "cn": "新恒星质量"},
    "dstar": {"en": "Stellar Density", "cn": "恒星密度"},
    "Mstar_i": {"en": "Initial Stellar Mass", "cn": "初始恒星质量"},
    "vol": {"en": "Volume", "cn": "体积"},#kpc³
    "mgas": {"en": "Gas Mass", "cn": "气体质量"},
    "coolt": {"en": "Cooling Time", "cn": "冷却时间"},#yr
    "abun_ism": {"en": "ISM Abundance", "cn": "星际介质丰度"},
    "abun_wind": {"en": "Wind Abundance", "cn": "风丰度"},#0到1
    "abun_jet": {"en": "Jet Abundance", "cn": "喷流丰度"},
    "abun_cold": {"en": "Cold Gas Abundance", "cn": "冷气体丰度"},
    "abun_hot": {"en": "Hot Gas Abundance", "cn": "热气体丰度"},
    "abun_init": {"en": "Initial Abundance", "cn": "初始丰度"},
    "velocity_magnitude": {"en": "Velocity Magnitude", "cn": "速度大小"},
    "temp_calculated": {"en": "Calculated Temperature", "cn": "计算温度"},
    "entropy": {"en": "Entropy", "cn": "熵"},
    "density_cgs": {"en": "Density (g/cm³)", "cn": "密度 (g/cm³)"},
    "r": {"en": "Radial Coordinate", "cn": "径向坐标"},
    "theta": {"en": "Angular Coordinate", "cn": "角向坐标"},
    "x": {"en": "Cartesian X Coordinate", "cn": "笛卡尔X坐标"},
    "y": {"en": "Cartesian Y Coordinate", "cn": "笛卡尔Y坐标"},
}


class HDFDataExtractor:
    """
    高效提取HDF模拟数据的工具类，使用pyhdf库访问数据

    参数:
        file_path (str): HDF文件所在目录
        dump_num (int): 数据dump编号
    """

    def __init__(self, file_path, dump_num):
        # 构建完整文件名
        self.filename = os.path.join(file_path, f"hdfra.{dump_num:03d}")
        print(f"加载文件: {self.filename}")

        try:
            # 打开HDF文件
            self.hdffile = SD(self.filename, SDC.READ)

            # 获取所有数据集信息
            self.datasets = self.hdffile.datasets()
            print(f"可用数据集: {list(self.datasets.keys())}")

            # 加载网格坐标
            self.r_coords = self.hdffile.select("fakeDim2")[:]  # 径向坐标
            self.theta_coords = self.hdffile.select("fakeDim1")[:]  # 角向坐标

            # 创建物理量映射字典
            self.quantity_map = {
                "v1": "Data-Set-2",#4*(Data-Set-数字)-1=np.array(file.select(数字).get())
                "v2": "Data-Set-3",
                "v3": "Data-Set-4",
                "density": "Data-Set-5",
                "temperature": "Data-Set-6",
                "gp_bg": "Data-Set-7",
                "a1": "Data-Set-8",
                "a2": "Data-Set-9",
                "energy": "Data-Set-10",
                "sfr": "Data-Set-11",
                "mnewstar": "Data-Set-12",
                "dstar": "Data-Set-13",
                "Mstar_i": "Data-Set-14",
                "vol": "Data-Set-15",
                "mgas": "Data-Set-16",
                "coolt": "Data-Set-17",
                "abun_ism": "Data-Set-18",
                "abun_wind": "Data-Set-19",
                "abun_jet": "Data-Set-20",
                "abun_cold": "Data-Set-21",
                "abun_hot": "Data-Set-22",
                "abun_init": "Data-Set-23",
            }

            # 计算笛卡尔坐标
            self._compute_cartesian_coordinates()

            print(f"成功加载数据 - 径向网格: {len(self.r_coords)}, 角向网格: {len(self.theta_coords)}")

        except Exception as e:
            print(f"文件加载失败: {str(e)}")
            raise

    def _compute_cartesian_coordinates(self):
        """计算笛卡尔坐标系(x, y)"""
        # 创建网格
        R, Theta = np.meshgrid(self.r_coords, self.theta_coords)

        # 转换为笛卡尔坐标
        self.x_grid = np.sin(Theta) * R
        self.y_grid = np.cos(Theta) * R

        # 处理y坐标为负值的情况
        self.y_grid = np.where(self.y_grid < 0, -np.abs(self.y_grid), self.y_grid)

    def _load_quantity(self, dataset_name):
        """安全加载指定物理量数据集"""
        try:
            # 使用数据集名称加载数据
            dataset = self.hdffile.select(dataset_name)
            return dataset.get()
        except Exception as e:
            raise RuntimeError(f"加载数据集'{dataset_name}'失败: {str(e)}")

    def _find_closest_index(self, values, target):
        """在数组中查找最接近目标值的索引"""
        return np.argmin(np.abs(values - target))

    def get_quantities_by_coords(self, r, theta):
        """
        通过物理坐标获取所有物理量

        参数:
            r (float): 径向坐标
            theta (float): 角向坐标(弧度)

        返回:
            dict: 包含该位置所有物理量的字典
        """
        # 查找最近的网格索引
        r_idx = self._find_closest_index(self.r_coords, r)
        theta_idx = self._find_closest_index(self.theta_coords, theta)

        # 使用索引获取数据
        return self.get_quantities_by_indices(r_idx, theta_idx)

    def get_quantities_by_indices(self, r_idx, theta_idx):
        """
        通过网格索引获取所有物理量

        参数:
            r_idx (int): 径向索引
            theta_idx (int): 角向索引

        返回:
            dict: 包含该位置所有物理量的字典
        """
        # 验证索引有效性
        if not (0 <= r_idx < len(self.r_coords)):
            raise ValueError(f"无效径向索引{r_idx}, 有效范围0-{len(self.r_coords) - 1}")

        if not (0 <= theta_idx < len(self.theta_coords)):
            raise ValueError(f"无效角向索引{theta_idx}, 有效范围0-{len(self.theta_coords) - 1}")

        results = {}

        # 首先加载基本物理量
        for quantity_name, dataset_name in self.quantity_map.items():
            try:
                data = self._load_quantity(dataset_name)
                # 确保数据是三维的 (time, theta, r)
                if data.ndim == 3:
                    # 取第一个时间步的数据
                    results[quantity_name] = data[0, theta_idx, r_idx]
                elif data.ndim == 2:
                    results[quantity_name] = data[theta_idx, r_idx]
                else:
                    print(f"警告: 数据集 '{dataset_name}' 维度异常: {data.shape}")
                    results[quantity_name] = None
            except Exception as e:
                print(f"警告: 无法加载{quantity_name}({dataset_name}): {str(e)}")
                results[quantity_name] = None

        # 添加坐标信息
        results["r"] = self.r_coords[r_idx]
        results["theta"] = self.theta_coords[theta_idx]
        results["x"] = self.x_grid[theta_idx, r_idx]
        results["y"] = self.y_grid[theta_idx, r_idx]

        # 添加矩阵位置信息
        results["matrix_position"] = (theta_idx, r_idx)

        # 计算派生量
        if results.get("energy") is not None and results.get("density") is not None:
            # 计算温度 (使用原始代码中的转换系数)
            results["temp_calculated"] = 93 * results["energy"] / results["density"]

            # 计算速度大小
            if results.get("v1") is not None and results.get("v2") is not None:
                results["velocity_magnitude"] = np.sqrt(
                    results["v1"] ** 2 + results["v2"] ** 2
                )

            # 计算熵
            if results.get("temperature") is not None:
                results["entropy"] = (
                        results["temperature"] * 8.625e-8 /
                        (results["density"] ** (2 / 3))
                )

            # 计算密度(g/cm³)
            if results.get("density") is not None:
                results["density_cgs"] = results["density"] * MU / LU ** 3

        return results

    def get_grid_shape(self):
        """返回网格尺寸"""
        return (len(self.r_coords), len(self.theta_coords))

    def get_coordinate_ranges(self):
        """返回坐标范围"""
        return {
            "r": (np.min(self.r_coords), np.max(self.r_coords)),
            "theta": (np.min(self.theta_coords), np.max(self.theta_coords)),
            "x": (np.min(self.x_grid), np.max(self.x_grid)),
            "y": (np.min(self.y_grid), np.max(self.y_grid))
        }

    def __del__(self):
        """确保文件在对象销毁时关闭"""
        if hasattr(self, 'hdffile') and self.hdffile:
            self.hdffile.end()


def print_physical_quantities(data):
    """
    打印所有物理量的中英文名称、数值和矩阵位置

    参数:
        data (dict): 包含物理量数据的字典
    """
    # 打印矩阵位置
    if "matrix_position" in data:
        theta_idx, r_idx = data["matrix_position"]
        print(f"\n矩阵位置: (θ索引={theta_idx}, r索引={r_idx})")

    # 打印坐标信息
    if "r" in data:
        print(f"径向坐标 (r): {data['r']:.6f} kpc")
    if "theta" in data:
        print(f"角向坐标 (θ): {data['theta']:.6f} rad")
    if "x" in data:
        print(f"笛卡尔X坐标: {data['x']:.6f} kpc")
    if "y" in data:
        print(f"笛卡尔Y坐标: {data['y']:.6f} kpc")

    print("\n物理量数值:")
    # 打印所有物理量
    for key, value in data.items():
        # 跳过坐标和位置信息
        if key in ["r", "theta", "x", "y", "matrix_position"]:
            continue

        # 获取中英文名称
        name_info = PHYSICAL_QUANTITIES.get(key, {"en": key, "cn": key})
        en_name = name_info["en"]
        cn_name = name_info["cn"]

        # 格式化数值
        if value is None:
            value_str = "N/A"
        elif isinstance(value, float):
            # 根据数值大小选择合适的格式
            if abs(value) < 1e-4 or abs(value) > 1e4:
                value_str = f"{value:.4e}"
            else:
                value_str = f"{value:.6f}"
        else:
            value_str = str(value)

        # 打印物理量信息
        print(f"{en_name} ({cn_name}): {value_str}")


# 使用示例
if __name__ == "__main__":
    # 配置文件路径
    file_path = "/Users/dingsensen/Downloads/document/paper/torus-AGN/simulaton1/torus_100yr_1"
    dump_num = 450  # 示例dump编号

    try:
        # 初始化提取器
        extractor = HDFDataExtractor(file_path, dump_num)

        # 获取网格信息
        grid_shape = extractor.get_grid_shape()
        coord_ranges = extractor.get_coordinate_ranges()

        print(f"\n网格尺寸: {grid_shape[0]}×{grid_shape[1]}")
        print(f"径向(r)范围: [{coord_ranges['r'][0]:.4f}, {coord_ranges['r'][1]:.4f}] kpc")
        print(f"角向(θ)范围: [{coord_ranges['theta'][0]:.4f}, {coord_ranges['theta'][1]:.4f}] rad")
        print(f"笛卡尔x范围: [{coord_ranges['x'][0]:.4f}, {coord_ranges['x'][1]:.4f}] kpc")
        print(f"笛卡尔y范围: [{coord_ranges['y'][0]:.4f}, {coord_ranges['y'][1]:.4f}] kpc")

        # 示例1: 通过坐标查询
        target_r = 0.026
        target_theta = 1
        print(f"\n查询坐标: r={target_r}, theta={target_theta}")
        data_by_coords = extractor.get_quantities_by_coords(target_r, target_theta)

        # 打印所有物理量信息
        print_physical_quantities(data_by_coords)

        # 示例2: 通过索引查询
        print("\n通过索引查询: r_idx=100, theta_idx=30")
        data_by_indices = extractor.get_quantities_by_indices(279, 36)

        # 打印所有物理量信息
        print_physical_quantities(data_by_indices)

    except Exception as e:
        print(f"发生错误: {str(e)}")