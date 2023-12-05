import os
import sys
from harvesters.core import Harvester
import numpy as np
# PhotoneoTL_DEV_<ID>
device_id = "PhotoneoTL_DEV_2019-09-051-LC3"
if len(sys.argv) == 2:
    device_id = "PhotoneoTL_DEV_" + sys.argv[1]
print("--> device_id: ", device_id)

cti_file_path = "/opt/PhoXiControl-1.9.x/API/lib/photoneo.cti"
print("--> cti_file_path: ", cti_file_path)

with Harvester() as h:
    h.add_file(cti_file_path, True, True)
    h.update()

    # Print out available devices
    print()
    print("Name : ID")
    print("---------")
    for item in h.device_info_list:
        print(item.property_dict['serial_number'], ' : ', item.property_dict['id_'])
    print()

    with h.create({'id_': device_id}) as ia:
        features = ia.remote_device.node_map

        ## General settings
        # ReadOnly
        is_phoxi_control_running = features.IsPhoXiControlRunning.value
        api_version = features.PhotoneoAPIVersion.value
        id = features.PhotoneoDeviceID.value
        type = features.PhotoneoDeviceType.value
        is_acquiring = features.IsAcquiring.value
        is_connected = features.IsConnected.value
        device_firmware_version = features.PhotoneoDeviceFirmwareVersion.value
        device_variant = features.PhotoneoDeviceVariant.value
        device_features = features.PhotoneoDeviceFeatures.value

        # `Freerun` or `Software`
        trigger_mode = features.PhotoneoTriggerMode.value
        features.PhotoneoTriggerMode.value = 'Freerun'
        # True or False
        wait_for_grabbing_end = features.WaitForGrabbingEnd.value
        features.WaitForGrabbingEnd.value = False
        # Timeout in ms, or values 0 (ZeroTimeout) and -1 (Infinity)
        get_frame_timeout = features.GetFrameTimeout.value
        features.GetFrameTimeout.value = 5000
        # True or False
        logout_after_disconnect = features.LogoutAfterDisconnect.value
        features.LogoutAfterDisconnect.value = False
        # True or False
        stop_acquisition_after_disconnect = features.StopAcquisitionAfterDisconnect.value
        features.StopAcquisitionAfterDisconnect.value = False


        ## Capturing settings
        # <1, 20>
        shutter_multiplier = features.ShutterMultiplier.value
        features.ShutterMultiplier.value = 5
        # <1, 20>
        scan_multiplier = features.ScanMultiplier.value
        features.ScanMultiplier.value = 5
        # `Res_2064_1544` or `Res_1032_772`
        resolution = features.Resolution.value
        features.Resolution.value = 'Res_1032_772'
        # True or False
        camera_only_mode = features.CameraOnlyMode.value
        features.CameraOnlyMode.value = False
        # True or False
        ambient_light_suppression = features.AmbientLightSuppression.value
        features.AmbientLightSuppression.value = False
        # `Normal` or `Interreflections`
        coding_strategy = features.CodingStrategy.value
        features.CodingStrategy.value = 'Normal'
        # `Fast`, `High` or `Ultra`
        coding_quality = features.CodingQuality.value
        features.CodingQuality.value = 'Ultra'
        # `Computed`, `LED`, `Laser` or `Focus`
        # texture_source = features.TextureSource.value
        features.TextureSource.value = 'Laser'
        # <10.24, 100.352>
        single_pattern_exposure = features.SinglePatternExposure.value
        features.SinglePatternExposure.value = 10.24
        # <0.0, 100.0>
        maximum_fps = features.MaximumFPS.value
        features.MaximumFPS.value = 25
        # <1, 4095>
        laser_power = features.LaserPower.value
        features.LaserPower.value = 2000
        # <0, 512>
        projection_offset_left = features.ProjectionOffsetLeft.value
        features.ProjectionOffsetLeft.value = 50
        # <0, 512>
        projection_offset_right = features.ProjectionOffsetRight.value
        features.ProjectionOffsetRight.value = 50
        # <1, 4095>
        led_power = features.LEDPower.value
        features.LEDPower.value = 2000
        # True or False
        hardware_trigger = features.HardwareTrigger.value
        features.HardwareTrigger.value = True
        # `Falling`, `Rising` or `Both`
        hardware_trigger_signal = features.HardwareTriggerSignal.value
        features.HardwareTriggerSignal.value = 'Both'


        ## Processing settings
        # <0.0, 100.0>
        max_inaccuracy = features.MaxInaccuracy.value
        features.MaxInaccuracy.value = 3.5
        # `MinX`, `MinY`, `MinZ`, `MaxX`, `MaxY` or `MaxZ`
        camera_space_selector = features.CameraSpaceSelector.value
        features.CameraSpaceSelector.value = 'MinZ'
        # <-999999.0, 999999.0>
        camera_space_value = features.CameraSpaceValue.value
        features.CameraSpaceValue.value = 100.5
        # `MinX`, `MinY`, `MinZ`, `MaxX`, `MaxY` or `MaxZ`
        point_cloud_space_selector = features.PointCloudSpaceSelector.value
        features.PointCloudSpaceSelector.value = 'MaxY'
        # <-999999.0, 999999.0>
        point_cloud_space_value = features.PointCloudSpaceValue.value
        features.PointCloudSpaceValue.value = 200.5
        # <0.0, 90.0>
        max_camera_angle = features.MaxCameraAngle.value
        features.MaxCameraAngle.value = 10
        # <0.0, 90.0>
        max_projector_angle = features.MaxProjectorAngle.value
        features.MaxProjectorAngle.value = 15
        # <0.0, 90.0>
        min_halfway_angle = features.MinHalfwayAngle.value
        features.MinHalfwayAngle.value = 20
        # <0.0, 90.0>
        max_halfway_angle = features.MaxHalfwayAngle.value
        features.MaxHalfwayAngle.value = 25
        # True or False
        calibration_volume_only = features.CalibrationVolumeOnly.value
        features.CalibrationVolumeOnly.value = True
        # `Sharp`, `Normal` or `Smooth`
        # surface_smoothness = features.SurfaceSmoothness.value
        features.SurfaceSmoothness.value = 'Normal'
        # <1, 4>
        normals_estimation_radius = features.NormalsEstimationRadius.value
        features.NormalsEstimationRadius.value = 1
        # True or False
        interreflections_filtering = features.InterreflectionsFiltering.value
        features.InterreflectionsFiltering.value = True
        # <0.01, 0.99>
        interreflections_filtering_strength = features.InterreflectionFilterStrength.value
        features.InterreflectionFilterStrength.value = 0.50


        ## Coordinates settings
        # `CameraSpace`, `MarkerSpace`, `RobotSpace` or `CustomSpace`
        camera_space = features.CoordinateSpace.value
        features.CoordinateSpace.value = 'MarkerSpace'
        # `Custom` or `Robot`
        transformation_space_selector = features.TransformationSpaceSelector.value
        features.TransformationSpaceSelector.value = 'Robot'
        # `Row0Col0`, `Row0Col1`, `Row0Col2`, `Row1Col0`, .. , `Row2Col2`
        transformation_rotation_matrix_selector = features.TransformationRotationMatrixSelector
        features.TransformationRotationMatrixSelector.value = 'Row0Col1'
        # <-999999.0, 999999.0>
        transformation_rotation_matrix_value = features.TransformationRotationMatrixValue
        features.TransformationRotationMatrixValue.value = 150.25
        # `X`, `Y` or `Z`
        transformation_translation_vector_selector = features.TransformationTranslationVectorSelector
        features.TransformationTranslationVectorSelector.value = 'Z'
        # <-999999.0, 999999.0>
        transformation_translation_vector_value = features.TransformationTranslationVectorValue
        features.TransformationTranslationVectorValue.value = 225.50
        # True or False
        recognize_markers = features.RecognizeMarkers
        features.RecognizeMarkers.value = True
        # <-999999.0, 999999.0>
        marker_scale_width = features.MarkerScaleWidth
        features.MarkerScaleWidth.value = 0.50
        # <-999999.0, 999999.0>
        marker_scale_height = features.MarkerScaleHeight
        features.MarkerScaleHeight.value = 0.50

        intrinsic_matrix=np.zeros((3,3))
        ## Calibration settings
        # `Row0Col0`, `Row0Col1`, `Row0Col2`, `Row1Col0`, .. , `Row2Col2`

        # camera_matrix_selector = features.CameraMatrixSelector.value
        # print(camera_matrix_selector)
        # for i in range(3):
        #     for j in range(3):
        #         features.CameraMatrixSelector.value ='Row'+str(i)+'Col'+str(j)
        #         intrinsic_matrix[i][j]=features.CameraMatrixValue.value
        # # features.CameraMatrixSelector.value = 'Row0Col0'
        # # camera_matrix_value = features.CameraMatrixValue.value
        # print(intrinsic_matrix)




        # ReadOnly
        # <0, 13>
        distortion_coefficient_selector = features.DistortionCoefficientSelector
        distortion_coefficient_value = []
        # ReadOnly
        for i in range(4):
            features.DistortionCoefficientSelector.value = i
            distortion_coefficient_value.append(features.DistortionCoefficientValue.value)
        print('distortion value:', distortion_coefficient_value)
        focus_length = features.FocusLength.value
        pixel_length_width = features.PixelSizeWidth.value
        pixel_length_height = features.PixelSizeHeight.value


        ## FrameOutput settings
        # Enable/Disable transfer of spefific images (True or False)
        features.SendPointCloud.value = True
        features.SendNormalMap.value = True
        features.SendDepthMap.value = True
        features.SendConfidenceMap.value = True
        features.SendTexture.value = True

        # The ia object will automatically call the destroy method
        # once it goes out of the block.

    # The h object will automatically call the reset method
    # once it goes out of the block.
