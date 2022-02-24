#!/usr/bin/env python
import numpy as np

from hardware.calibrate_camera import Calibration

if __name__ == '__main__':
    calibration = Calibration(
        cam_id=830112070066,
        calib_grid_step=0.1,
        checkerboard_offset_from_tool=[0.0, 0.075, 0.16],
        workspace_limits=np.asarray([[0.0, 0.2], [0.2, 0.4], [0.0, 0.2]])
    )
    calibration.run()
