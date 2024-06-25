import hdf5storage
import numpy as np

def Rawdata_loader(opt):
    '''
    Function: load raw data

    # Perset params:
        :param mat_file: Path of the raw data mat file.
        :param logger: logger instance
    # Derived parameters
        :param U: raw complex data
                % | img_x | img_y | img_z | time_sample | dirction | freq |
                % | ----- | ----- | ----- | ----------- | -------- | ---- |
                % | 1     | 2     | 3     | 4           | 5        | 6    |
        :param dxyz:  Resolution (m), list, double
        :param Freq_list:  The list of  frequency (Hz), list, double
        :param Phase_point: Phase for each sample time (rad, +,-means the MEG polarity),list, double
    :Note:
        The mat file must contain U,dxyz,Freq_list,Phase_point vars.
    '''
    #%% 1. load data
    data = hdf5storage.loadmat(opt['mat_file'])
    opt['U'] = data['U']
    opt['dxyz'] = np.squeeze(data['dxyz'])
    opt['Freq_list'] =  np.squeeze(data['Freq_list'])
    opt['Phase_point'] = np.squeeze(data['Phase_point'])
    #%% 2. exception handling
    while len(opt['U'].shape)<6:
        opt['U']=opt['U'][...,np.newaxis]


    #%% 3. info print
    opt['logger'].info("Step1: data_load--load rawdata.")
    opt['logger'].info("U's shape is x:%d,y:%d,z:%d,t:%d,d:%d,f:%d"%(opt['U'].shape[0],
                                                 opt['U'].shape[1],
                                                 opt['U'].shape[2],
                                                 opt['U'].shape[3],
                                                 opt['U'].shape[4],
                                                 opt['U'].shape[5],))
    opt['logger'].info("dxyz is (%.5f,%.5f,%.5f)"%(opt['dxyz'][0],opt['dxyz'][1],opt['dxyz'][2]))
    opt['logger'].info("Freq_list is %s"%(str(opt['Freq_list'])))
    opt['logger'].info("Phase_point is pi*(%s)\n"%(str(opt['Phase_point']/np.pi)))
    return opt