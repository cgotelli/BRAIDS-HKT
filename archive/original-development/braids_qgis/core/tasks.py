"""QGIS background tasks for BRAIDS preparation and result conversion."""
from qgis.core import QgsTask

from braids_core.boundaries.domain import PolygonDomain
from braids_core.boundaries.cut_cells import SolidObstacle
from braids_core.boundaries.inlet_io import NativeGrid

from .mesh_visualization import write_initial_mesh,write_adaptive_initial_mesh
from .project import prepare_run
from .results import materialize_results
from .gpu_preparation import prepare_gpu_run


class PrepareBraidsRunTask(QgsTask):
    def __init__(self,project_path):
        super().__init__("Prepare BRAIDS polygon run",QgsTask.Flag.CanCancel)
        self.project_path=project_path;self.run_root=None;self.error=None

    def run(self):
        try:
            self.run_root=prepare_run(self.project_path,progress=self.setProgress,cancelled=self.isCanceled);return True
        except InterruptedError:return False
        except Exception as exc:
            self.run_root=getattr(exc,"run_root",None)
            # Retaining a traceback can also retain GDAL datasets and their
            # Windows file locks after a task has failed.
            self.error=RuntimeError(str(exc));return False


class PrepareBraidsGpuRunTask(QgsTask):
    """Prepare an authenticated external GPU input without importing CUDA."""

    def __init__(self, project_path, *, worker_root, expected_manifest_sha256,
                 python_executable, expected_python_sha256,
                 gpu_integrator="ssp43", gpu_launch_mode="graph",
                 gpu_numerical_variant="auto", gpu_amr_evolving_banks=False,
                 gpu_amr_selector_policy=None):
        super().__init__("Prepare BRAIDS GPU run", QgsTask.Flag.CanCancel)
        self.project_path = project_path
        self.worker_root = worker_root
        self.expected_manifest_sha256 = expected_manifest_sha256
        self.python_executable = python_executable
        self.expected_python_sha256 = expected_python_sha256
        self.gpu_integrator = gpu_integrator
        self.gpu_launch_mode = gpu_launch_mode
        self.gpu_numerical_variant = gpu_numerical_variant
        self.gpu_amr_evolving_banks = gpu_amr_evolving_banks
        self.gpu_amr_selector_policy = gpu_amr_selector_policy
        self.run_root = None
        self.prepared_root = None
        self.error = None

    def run(self):
        try:
            self.run_root, self.prepared_root = prepare_gpu_run(
                self.project_path, worker_root=self.worker_root,
                expected_manifest_sha256=self.expected_manifest_sha256,
                python_executable=self.python_executable,
                expected_python_sha256=self.expected_python_sha256,
                gpu_integrator=self.gpu_integrator,
                gpu_launch_mode=self.gpu_launch_mode,
                gpu_numerical_variant=self.gpu_numerical_variant,
                gpu_amr_evolving_banks=self.gpu_amr_evolving_banks,
                gpu_amr_selector_policy=self.gpu_amr_selector_policy,
                progress=self.setProgress, cancelled=self.isCanceled,
                native_ready=lambda root:setattr(self,"run_root",root))
            return True
        except InterruptedError:
            return False
        except Exception as exc:
            self.error = RuntimeError(str(exc))
            return False


class PrepareBraidsResultsTask(QgsTask):
    def __init__(self,run_root,animated_mesh=False,prepared_root=None):
        super().__init__("Prepare BRAIDS raster results",QgsTask.Flag.CanCancel)
        self.run_root=run_root;self.animated_mesh=bool(animated_mesh)
        self.prepared_root=prepared_root;self.manifest=None;self.error=None

    def run(self):
        try:
            if self.isCanceled():return False
            self.manifest=materialize_results(self.run_root,animated_mesh=self.animated_mesh,
                                              progress=self.setProgress,cancelled=self.isCanceled,
                                              prepared_root=self.prepared_root)
            self.setProgress(100);return True
        except InterruptedError:return False
        except Exception as exc:
            self.error=RuntimeError(str(exc));return False


class PrepareBraidsMeshTask(QgsTask):
    def __init__(self,domain_data,grid_data,minimum_area_fraction,output_path,obstacle_data=(),adaptive_project=None):
        super().__init__("Build BRAIDS initial mesh preview",QgsTask.Flag.CanCancel)
        self.domain_data=domain_data;self.grid_data=grid_data
        self.minimum_area_fraction=minimum_area_fraction;self.output_path=output_path
        self.obstacle_data=tuple(obstacle_data)
        self.adaptive_project=adaptive_project
        self.metadata=None;self.error=None

    def run(self):
        try:
            if self.isCanceled():return False
            if self.adaptive_project is not None:
                self.metadata=write_adaptive_initial_mesh(self.output_path,self.adaptive_project,
                    progress=self.setProgress,cancelled=self.isCanceled)
                return True
            domain=PolygonDomain(self.domain_data["domain_id"],self.domain_data["vertices_xy"],
                                 self.domain_data["crs_wkt"])
            grid=NativeGrid(**self.grid_data)
            obstacles=tuple(SolidObstacle(row["obstacle_id"],row["vertices_xy"])
                            for row in self.obstacle_data)
            self.metadata=write_initial_mesh(self.output_path,domain,grid,self.minimum_area_fraction,obstacles)
            self.setProgress(100);return True
        except Exception as exc:
            self.error=RuntimeError(str(exc));return False
