#!/usr/bin/env python
import numpy as np

from hardware.calibrate_camera import Calibration

if __name__ == '__main__':
    calibration = Calibration(
        cam_id=830112070066,
        calib_grid_step=0.05,
        checkerboard_offset_from_tool=[0.0, 0.114, 0.0],
        workspace_limits=np.asarray([[-0.2, 0.2], [0.2, 0.4], [0.02, 0.2]])
    )
    calibration.run()
