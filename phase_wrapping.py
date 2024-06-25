import hdf5storage
import numpy as np
from tqdm import tqdm
import os

#%% funciton
def phase_wrapping_main(path_file,name_,unwrpping_):
    # %% step0: initialization
    from Step0_TWE_init import TWE_init
    for name in name_:
        opt = TWE_init()
        # %%
        opt['mat_file'] = path_file+'U_'+name+'.mat'  # path for U
        opt['unwrap_mat_path'] = path_file+'W_'+name+'.mat'  # path for W
        #%% step2: displacemnet extraction
        from Step2_displacement_extraction import Displacement_extraction
        opt['lr_unwrap']=0.01/2
        opt['iter_max_unwrap']=1000
        opt['gradientDC_coff_unwrap']=100
        opt['show_flag_unwrap']= False  # flag for show pic in unwrapping
        opt['show_gap_num_unwrap']= 1  # show gap num for unwrapping

        opt['lr_unwrap'] = unwrpping_[0]
        opt['iter_max_unwrap'] = unwrpping_[1]
        opt['gradientDC_coff_unwrap'] = unwrpping_[2]

        if os.path.exists(opt['unwrap_mat_path']):
            data = hdf5storage.loadmat(opt['unwrap_mat_path'])
            opt['W'] = data['W']
            while len(opt['W'].shape) < 5:
                opt['W'] = opt['W'][..., np.newaxis]
            opt['dxyz'] = np.squeeze(data['dxyz'])
            opt['Freq_list'] = np.squeeze(np.array(data['Freq_list']))
            assert len(opt['Freq_list']) == opt['W'].shape[4], 'Freq parameter mismatch!'
        else:
            # %% step1: load data
            from Step1_data_loader import Rawdata_loader
            opt = Rawdata_loader(opt)
            assert len(opt['Freq_list']) == opt['U'].shape[5], 'Freq parameter mismatch!'
            assert len(opt['Phase_point']) == opt['U'].shape[3], 'Sample time parameter mismatch!'

            U=opt['U']
            x, y, z, Sample_num, Direction_num, Freq_num = U.shape[0], \
                                                               U.shape[1], \
                                                               U.shape[2], \
                                                               U.shape[3], \
                                                               U.shape[4], \
                                                               U.shape[5]
            for f_i in tqdm(range(U.shape[2]), desc='unwrapping'):
                opt['U']=np.reshape(U[:,:,f_i,:,:,:],[x,y,1,Sample_num, Direction_num, Freq_num])
                opt=Displacement_extraction(opt)
                if f_i==0:
                    WW=opt['W']
                    pdx=opt['phasedx']
                    pdy=opt['phasedy']
                else:
                    WW=np.concatenate((WW,opt['W']),axis=2)
                    pdx = np.concatenate((pdx, opt['phasedx']), axis=2)
                    pdy = np.concatenate((pdy, opt['phasedy']), axis=2)
            opt['U']=U
            opt['W']=WW
            if not os.path.exists(opt['unwrap_mat_path']):
                hdf5storage.savemat(opt['unwrap_mat_path'],
                                    {
                                        # 'pdx': pdx,
                                        # 'pdy': pdy,
                                        'W': WW,
                                        'dxyz': opt['dxyz'],
                                        'Freq_list': opt['Freq_list'],
                                    },
                                    format='7.3', matlab_compatible=True)


