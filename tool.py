import nibabel as nib
import numpy as np
import os
import scipy.ndimage.morphology
import shutil

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.direction import ProbabilisticDirectionGetter

from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.dsi import DiffusionSpectrumModel
from dipy.reconst.dti import TensorModel, fractional_anisotropy

from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response_ssst)
from dipy.segment.mask import median_otsu
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamlinespeed import length

# AnalysisContext documentation: https://docs.qmenta.com/sdk/sdk.html
def run(context):
    
    ####################################################
    # Get the path to input files  and other parameter #
    ####################################################
    analysis_data       = context.fetch_analysis_data()
    settings            = analysis_data['settings']
    postprocessing      = settings['postprocessing']
    numberOfStreamlines = settings['numberOfStreamlines']
    dataSupportExponent = settings['dataSupportExponent']
    dataset             = settings['dataset']

    hcpl_dwi_file_handle = context.get_files('input', modality='HARDI')[0]
    hcpl_dwi_file_path = hcpl_dwi_file_handle.download('/root/')

    hcpl_bvalues_file_handle = context.get_files(
        'input', reg_expression='.*prep.bvalues.hcpl.txt')[0]
    hcpl_bvalues_file_path = hcpl_bvalues_file_handle.download('/root/')
    hcpl_bvecs_file_handle = context.get_files(
        'input', reg_expression='.*prep.gradients.hcpl.txt')[0]
    hcpl_bvecs_file_path = hcpl_bvecs_file_handle.download('/root/')

    dwi_file_handle = context.get_files('input', modality='DSI')[0]
    dwi_file_path = dwi_file_handle.download('/root/')
    bvalues_file_handle = context.get_files(
        'input', reg_expression='.*prep.bvalues.txt')[0]
    bvalues_file_path = bvalues_file_handle.download('/root/')
    bvecs_file_handle = context.get_files(
        'input', reg_expression='.*prep.gradients.txt')[0]
    bvecs_file_path = bvecs_file_handle.download('/root/')

    inject_file_handle = context.get_files(
        'input', reg_expression='.*prep.inject.nii.gz')[0]
    inject_file_path = inject_file_handle.download('/root/')

    VUMC_ROIs_file_handle = context.get_files(
        'input', reg_expression='.*VUMC_ROIs.nii.gz')[0]
    VUMC_ROIs_file_path = VUMC_ROIs_file_handle.download('/root/')
    
    
    if dataset == "HCPL":
        acqType                 = 'hcpl'
        dwi                     = hcpl_dwi_file_path
        input_bval_file_name    = hcpl_bvalues_file_path
        input_bvec_file_name    = hcpl_bvecs_file_path
    elif dataset == "DSI":
        acqType = 'dwi'
        dwi                     = dwi_file_path
        input_bval_file_name    = bvalues_file_path
        input_bvec_file_name    = bvecs_file_path
    else:
        context.set_progress(message='Wrong dataset parameter')
    
    #########################################
    # Convert bvals and bvecs to FSL format #
    #########################################
   
    # Use 1/4 of provided bvals        
    bval    = '/root/local_exec_output/prep.' + acqType + '.bval'
    bvec    = '/root/local_exec_output/prep.' + acqType + '.bvec'
    
    tmp     = np.loadtxt(input_bval_file_name)
    tmp     = np.reshape(tmp,(-1,len(tmp)))/4
    np.savetxt(bval, tmp, fmt='%.0f', delimiter=' ')
    
    tmp=np.loadtxt(input_bvec_file_name).transpose()
    np.savetxt(bvec, tmp, fmt='%.5f', delimiter=' ')
    
  
    ######################
    # Extract brain mask #
    ######################
    
    os.system("/miniconda/bin/dwiextract -bzero " +
              "-fslgrad " + 
              bvec + " " + bval + 
              " -force " + 
              dwi + 
              " /root/local_exec_output/mask.nii.gz")
    
    os.system("/miniconda/bin/mrfilter -force /root/local_exec_output/mask.nii.gz median /root/local_exec_output/mask.nii.gz")
    os.system("/miniconda/bin/mrfilter -force /root/local_exec_output/mask.nii.gz smooth /root/local_exec_output/mask.nii.gz")
    os.system("/miniconda/bin/mrthreshold -force /root/local_exec_output/mask.nii.gz /root/local_exec_output/mask.nii.gz")    
    
    #############################
    # Compute response function #
    #############################
    
    os.system("/miniconda/bin/dwi2response dhollander " +
               "-fslgrad " + 
               bvec + " " + bval + 
               " -force " + 
               " -mask /root/local_exec_output/mask.nii.gz " + dwi +
               " /root/local_exec_output/wm_response.txt" + 
               " /root/local_exec_output/gm_response.txt" + 
               " /root/local_exec_output/csf_response.txt" +
               " -voxels /root/local_exec_output/responSel.nii.gz")
    
    #####################
    # Compute FOD image #
    #####################
    
    os.system("/miniconda/bin/dwi2fod " +
              "-fslgrad " + 
               bvec + " " + bval + 
               " -force " + 
               " -mask /root/local_exec_output/mask.nii.gz msmt_csd " + dwi +
               " /root/local_exec_output/wm_response.txt /root/local_exec_output/wmfod.nii.gz " +
               " /root/local_exec_output/gm_response.txt /root/local_exec_output/gm.nii.gz " +
               " /root/local_exec_output/csf_response.txt /root/local_exec_output/csf.nii.gz")


    ################
    # Tractography #
    ################

    # Get tractogram using Trekker (.vtk output) 
    os.system("./trekker_linux_x64_v0.7" +
               " -fod /root/local_exec_input/wmfod.nii.gz" + 
               " -seed_image /root/local_exec_output/mask.nii.gz" + 
               " -pathway=stop_at_exit /root/local_exec_output/mask.nii.gz" +
               " -pathway=require_entry " + inject_file_path +
               " -seed_count " + numberOfStreamlines +
               " -dataSupportExponent " + dataSupportExponent +
               " -minFODamp 0.05" +
               " -minRadiusOfCurvature 0.1" +
               " -probeLength 0.025" +
               " -writeInterval 40" +
               " -verboseLevel 0" +
               " -output /root/local_exec_output/tractogram.vtk")
    
    ################################
    # Tractogram format conversion #
    ################################
    
    # Convert .vtk to .trk (the long way, in order to have a smaller docker image)
    os.system("/miniconda/bin/tckconvert -force /root/local_exec_output/tractogram.vtk /root/local_exec_output/tractogram.tck")
    streamlines = nib.streamlines.load('/root/local_exec_output/tractogram.tck').streamlines
    nii         = nib.load('/root/local_exec_output/mask.nii.gz')
    affine      = nii.affine
    

    ##################
    # Postprocessing #
    ##################

    if postprocessing in ["EPFL", "ALL"]:
        context.set_progress(message='Processing density map (EPFL)')
        volume_folder = "/root/vol_epfl"
        output_epfl_zip_file_path = "/root/X-link_EPFL.zip"
        os.mkdir(volume_folder)
        lengths = length(streamlines)
        streamlines = streamlines[lengths > 1]
        density = utils.density_map(streamlines, affine, nii.shape)
        density = scipy.ndimage.gaussian_filter(density.astype("float32"), 0.5)

        log_density = np.log10(density + 1)
        max_density = np.max(log_density)
        for i, t in enumerate(np.arange(0, max_density, max_density / 200)):
            nbr = str(i)
            nbr = nbr.zfill(3)
            mask = log_density >= t
            vol_filename = os.path.join(volume_folder,
                                        "vol" + nbr + "_t" + str(t) + ".nii.gz")
            nib.Nifti1Image(mask.astype("int32"), affine,
                            nii.header).to_filename(vol_filename)
        shutil.make_archive(output_epfl_zip_file_path[:-4], 'zip', volume_folder)

    if postprocessing in ["VUMC", "ALL"]:
        context.set_progress(message='Processing density map (VUMC)')
        ROIs_img = nib.load(VUMC_ROIs_file_path)
        volume_folder = "/root/vol_vumc"
        output_vumc_zip_file_path = "/root/X-link_VUMC.zip"
        os.mkdir(volume_folder)
        lengths = length(streamlines)
        streamlines = streamlines[lengths > 1]

        rois = ROIs_img.get_fdata().astype(int)
        _, grouping = utils.connectivity_matrix(streamlines, affine, rois,
                                                inclusive=True,
                                                return_mapping=True,
                                                mapping_as_streamlines=False)
        streamlines = streamlines[grouping[(0, 1)]]

        density = utils.density_map(streamlines, affine, nii.shape)
        density = scipy.ndimage.gaussian_filter(density.astype("float32"), 0.5)

        log_density = np.log10(density + 1)
        max_density = np.max(log_density)
        for i, t in enumerate(np.arange(0, max_density, max_density / 200)):
            nbr = str(i)
            nbr = nbr.zfill(3)
            mask = log_density >= t
            vol_filename = os.path.join(volume_folder,
                                        "vol" + nbr + "_t" + str(t) + ".nii.gz")
            nib.Nifti1Image(mask.astype("int32"), affine,
                            nii.header).to_filename(vol_filename)
        shutil.make_archive(output_vumc_zip_file_path[:-4], 'zip', volume_folder)

    ###################
    # Upload the data #
    ###################
    context.set_progress(message='Uploading results...')
    #context.upload_file(fa_file_path, 'fa.nii.gz')
    # context.upload_file(fod_file_path, 'fod.nii.gz')
    # context.upload_file(streamlines_file_path, 'streamlines.trk')
    if postprocessing in ["EPFL", "ALL"]:
        context.upload_file(output_epfl_zip_file_path,
                            'X-link_' + dataset +'_EPFL.zip')
    if postprocessing in ["VUMC", "ALL"]:
        context.upload_file(output_vumc_zip_file_path,
                            'X-link_' + dataset +'_VUMC.zip')
    
    
    
              
    
    
    
