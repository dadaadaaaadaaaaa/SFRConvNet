
from pyhdf.SD import SD
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class PhysicalUnit():
    """
    This class defines several commonly used physical constants and code units.

    Simulation base units:
    - time   : Gyr
    - length : kpc
    - mass   : 2.5e7 solar masses

    Notes
    -----
    These constants are kept consistent with the original internal analysis script.
    """

    def __init__(self):
        self.mass = 4.91534e37 # 超大质量黑洞的质量
        self.solarmass = 2e30
        self.kpc = 3.08419e19
        self.pc = self.kpc * 1e-3
        self.Gyr = 3.1536e16
        self.c = 3e8
        self.yr = 3.1536e7
        self.G = 6.67e-11
        self.sigmaT = 6.652e-29 # 汤姆逊散射截面，即光子被自由电子散射的概率
        self.mp = 1.67246e-27 # 质子静止质量
        self.ledd = 1.2581e31 # 单位太阳质量的爱丁顿光度，即当辐射压力和引力达到平衡时的最大光度
        self.mdotedd = 1.39788e15 # 单位太阳质量的爱丁顿吸积率
        self.k = 1.38e-23 # 玻尔兹曼常数

        # NOTE:
        # The X-ray related helper functions in the original private workflow
        # are not included here, because they depend on additional external
        # tables/variables that are not part of this public data release.


class HDFdata():
    """
    This class reads one HDF4 snapshot and stores the main simulation fields.

    Parameters
    ----------
    dump : int
        Snapshot index. For example, dump=45 reads 'hdfra.045'.

    file_path : str, optional
        Path to the directory containing the snapshot files.
        Example:
            file_path = "/your/path/to/snapshots/"
        Default is the current directory ("").

    Main loaded quantities
    ----------------------
    Coordinates:
        r, theta

    Velocity fields:
        v1 : radial velocity
        v2 : theta-direction velocity
        v3 : phi-direction velocity

    Gas properties:
        d      : gas number density
        T_file : gas temperature stored directly in the HDF file
        e      : 气体内能密度（Data-Set-10）
        T      : temperature recomputed from e and d using the original script
        K      : entropy-like quantity

    Star / stellar quantities:
        sfr      : star formation rate
        mnewstar : mass of newly formed stars
        dstar    : stellar density
        Mstar_i  : initial stellar mass

    Grid and gas content:
        vol   : cell volume
        mgas  : gas mass
        coolt : gas cooling time

    Tracer / abundance fields:
        abun_wind : AGN wind tracer
        abun_jet  : AGN jet tracer
        abun_cold : cold-mode cosmological inflow tracer
        abun_hot  : hot-mode cosmological inflow tracer
        abun_init : initial gas tracer

    Additional fields kept for completeness:
        GP_BG, a1, a2, abun_ism
    These are retained from the original file structure, but are not used in
    our standard public analysis workflow.
    """

    def __init__(self, dump, file_path=""):
        """Initialize the data dump and filename."""
        self.filename = file_path + "hdfra." + "%03d" % (dump)
        self.hdffile = SD(self.filename)

        # Velocity components
        self.v1 = self.hdffile.select("Data-Set-2")[0, :, :]   # radial velocity
        self.v2 = self.hdffile.select("Data-Set-3")[0, :, :]   # theta-direction velocity
        self.v3 = self.hdffile.select("Data-Set-4")[0, :, :]   # phi-direction velocity

        # Gas thermodynamic quantities
        self.d = self.hdffile.select("Data-Set-5")[0, :, :]    # gas number density
        self.T_file = self.hdffile.select("Data-Set-6")[0, :, :]  # temperature stored in the HDF file

        # Additional fields kept for completeness; not used in the standard analysis
        self.GP_BG = self.hdffile.select("Data-Set-7")[0, :, :]   # auxiliary field, not used
        self.a1 = self.hdffile.select("Data-Set-8")[0, :, :]      # auxiliary field, not used
        self.a2 = self.hdffile.select("Data-Set-9")[0, :, :]      # auxiliary field, not used

        self.e = self.hdffile.select("Data-Set-10")[0, :, :]   # 气体内能密度

        # Star formation and stellar quantities
        self.sfr = self.hdffile.select("Data-Set-11")[0, :, :]      # star formation rate
        self.mnewstar = self.hdffile.select("Data-Set-12")[0, :, :] # mass of newly formed stars
        self.dstar = self.hdffile.select("Data-Set-13")[0, :, :]    # stellar density
        self.Mstar_i = self.hdffile.select("Data-Set-14")[0, :, :]  # initial stellar mass

        # Grid / gas properties
        self.vol = self.hdffile.select("Data-Set-15")[0, :, :]    # grid cell volume
        self.mgas = self.hdffile.select("Data-Set-16")[0, :, :]   # gas mass in each cell
        self.coolt = self.hdffile.select("Data-Set-17")[0, :, :]  # gas cooling time

        # Tracer / abundance fields
        self.abun_ism = self.hdffile.select("Data-Set-18")[0, :, :]   # auxiliary abundance field, not used
        self.abun_wind = self.hdffile.select("Data-Set-19")[0, :, :]  # abundance of AGN wind
        self.abun_jet = self.hdffile.select("Data-Set-20")[0, :, :]   # abundance of AGN jet
        self.abun_cold = self.hdffile.select("Data-Set-21")[0, :, :]  # abundance of cold-mode cosmological inflow
        self.abun_hot = self.hdffile.select("Data-Set-22")[0, :, :]   # abundance of hot-mode cosmological inflow
        self.abun_init = self.hdffile.select("Data-Set-23")[0, :, :]  # abundance of the initial setup gas

        # Grid coordinates
        self.r = self.hdffile.select("fakeDim2")[:]       # radial coordinate
        self.theta = self.hdffile.select("fakeDim1")[:]   # polar angle coordinate

        self.lentheta = len(self.theta)
        self.lenr = len(self.r)
        self.hdffile.end()

        # Grid spacing
        self.dr = np.zeros(self.lenr)
        self.dtheta = np.zeros(self.lentheta)

        for i in range(self.lenr - 1):
            self.dr[i] = self.r[i + 1] - self.r[i]
        self.dr[-1] = self.dr[-2]

        for i in range(self.lentheta):
            self.dtheta[i] = 9 / 10 * np.pi / self.lentheta

        # Keep the original script convention:
        # 按 MACER 恒等式由内能密度和气体密度重新计算温度。
        with np.errstate(divide='ignore', invalid='ignore'):
            self.T = 93 * self.e / self.d
            self.K = self.T * 8.625e-8 / self.d ** (2 / 3)
