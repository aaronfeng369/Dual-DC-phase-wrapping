import numpy as np
import torch
#import matplotlib.pyplot as plt
from tqdm import tqdm

def Displacement_extraction(opt):
    '''
    Function: extract the displacement
    # Perset params:
        :param U: raw complex data
                % | img_x | img_y | img_z | time_sample | dirction | freq |
                % | ----- | ----- | ----- | ----------- | -------- | ---- |
                % | 1     | 2     | 3     | 4           | 5        | 6    |
        :param Phase_point: Phase for each sample time (rad, +,-means the MEG polarity),list, double

        :param lr_unwrap: learning rate for unwrapping
        :param iter_max_unwrap: max iter num for unwrapping
        :param gradientDC_coff_unwrap: coff for gradientDC

        :param show_flag_unwrap: flag for show pic in unwrapping
        :param show_gap_num_unwrap: show gap num for unwrapping

        :param device: device choise
        :param logger: logger instance
    # Derived parameters
        :param phasedx: dphase/dx
                % | img_x | img_y | img_z  | dirction | freq |
                % | ----- | ----- | -----  | -------- | ---- |
                % | 1     | 2     | 3      | 4        | 5    |
        :param phasedy: dphase/dy
                % | img_x | img_y | img_z  | dirction | freq |
                % | ----- | ----- | -----  | -------- | ---- |
                % | 1     | 2     | 3      | 4        | 5    |
        :param W: unwrapping complex displacement
                % | img_x | img_y | img_z  | dirction | freq |
                % | ----- | ----- | -----  | -------- | ---- |
                % | 1     | 2     | 3      | 4        | 5    |
    :Note:

    :return:
    '''

    logger = opt['logger']
    device = opt['device']

    U = opt['U']
    Phase_point = opt['Phase_point']


    #%% 1. gradient phase
    x, y, z, Sample_num, Direction_num, Freq_num = U.shape[0], \
                                                   U.shape[1], \
                                                   U.shape[2], \
                                                   U.shape[3], \
                                                   U.shape[4], \
                                                   U.shape[5]

    polarity_point = np.sign(Phase_point)
    Phase_point_ = np.abs(Phase_point)
    polarity_point[polarity_point == 0] = 1
    Coff_matrix = torch.from_numpy(np.stack((np.multiply(polarity_point, np.cos(Phase_point_)),
                              -np.multiply(polarity_point, np.sin(Phase_point_))),
                             axis=1)).to(torch.cfloat).to(device)

    U_ = torch.from_numpy(U).to(torch.cfloat).to(device)

    U = U + 1e-6 + 1j * 1e-6
    U_[torch.abs(U_) == 0] = torch.nan
    U_=(U_/torch.abs(U_)).permute([0,1,2,4,5,3])

    Udx=-1j*(U_[1:,0:-1,...]-U_[0:-1,0:-1,...])/U_[0:-1,0:-1,...]
    Udx=Udx.contiguous().view((x-1),(y-1),z, Direction_num, Freq_num,Sample_num,1).real
    Udy = -1j*(U_[ 0:-1,1:, ...] - U_[0:-1, 0:-1, ...])/U_[0:-1,0:-1,...]
    Udy = Udy.contiguous().view((x-1),(y-1),z, Direction_num, Freq_num,Sample_num,1).real


    tmp=torch.matmul(torch.transpose(Coff_matrix, 0, 1), Coff_matrix)
    tmp=torch.inverse(tmp)
    tmp=torch.matmul(tmp,torch.transpose(Coff_matrix, 0, 1))
    inv_matrix=tmp.view(1,1,1,1,1,2,Sample_num).real

    phasedx=torch.matmul(inv_matrix,Udx)
    # phasedx=phasedx[:,:,:,:,:,0,:]+1j*phasedx[:,:,:,:,:,1,:]
    phasedx=phasedx.view(x-1,y-1,z,Direction_num,Freq_num,2)
    phasedy = torch.matmul(inv_matrix, Udy)
    # phasedy = phasedy[:, :, :, :, :, 0] + 1j * phasedy[:, :, :, :, :, 1]
    phasedy = phasedy.view(x-1,y-1,z,Direction_num,Freq_num,2)

    #%% 2. unwrapping phase
    #  cross wave
    index_tri = np.arange(Sample_num ** 2).reshape(Sample_num, Sample_num)[np.triu_indices(Sample_num, 1)]
    # U[U == 0] = np.nan
    U=U+1e-6+1j*1e-6
    U = U / np.abs(U)
    wave_n = np.reshape(U[:, :, :, np.newaxis, :, :, :] / U[:, :, :, :, np.newaxis, :, :]
                        , [x, y, z, Sample_num ** 2, Direction_num, Freq_num])
    cross_wave = wave_n[:, :, :, index_tri, :, :]
    cross_wave = np.stack((np.real(cross_wave), np.imag(cross_wave)), axis=6).transpose([0, 1, 2, 4, 5, 3, 6])
    #  cross offset
    polarity_point = np.sign(Phase_point)
    Phase_point = np.abs(Phase_point)
    polarity_point[polarity_point == 0] = 1
    Offset_matrix = np.stack((np.multiply(polarity_point, np.cos(Phase_point)),
                              -np.multiply(polarity_point, np.sin(Phase_point))),
                             axis=1)
    cross_Offset_matrix = (Offset_matrix[np.newaxis, :, :] - Offset_matrix[:, np.newaxis, :]) \
                              .reshape(Sample_num ** 2, 2)[index_tri, :]
    #  Data To cuda
    cross_wave_cuda = torch.from_numpy(cross_wave).to(torch.float).to(device)
    cross_Offset_matrix_cuda = torch.from_numpy(cross_Offset_matrix).to(torch.float).to(device)
    Phi_U = torch.zeros([x, y, z, Direction_num, Freq_num, 2]).to(torch.float).to(device)
    Phi_U.requires_grad = True

    # iteration
    loss_dc = lambda x, y: torch.nanmean(torch.abs(x - y))
    loss_dc_2 = lambda x, y: (torch.nanmean(torch.abs(x - y)*torch.abs(x - y)))
    optim = torch.optim.Adam([Phi_U],
                             lr=opt['lr_unwrap'],
                             betas=(0.9, 0.999),
                             eps=1e-08,
                             )


    iter_max = opt['iter_max_unwrap']
    for ii in range(iter_max):
        phi_i = torch.mm(Phi_U.view(-1, 2), cross_Offset_matrix_cuda.transpose(1, 0)) \
            .reshape([x, y, z, Direction_num, Freq_num, -1])
        R_eq = torch.stack((torch.cos(phi_i), torch.sin(phi_i)), -1)
        Loss_dc_val = loss_dc_2(R_eq, cross_wave_cuda)

        if ii==iter_max:
            optim = torch.optim.Adam([Phi_U],
                                         lr=opt['lr_unwrap']/100,
                                         betas=(0.9, 0.999),
                                         eps=1e-08,
                                         )

        if ii<iter_max:
            pdx = Phi_U[1:, 0:-1, ...] - Phi_U[0:-1, 0:-1, ...]
            pdy = Phi_U[0:-1, 1:, ...] - Phi_U[0:-1, 0:-1, ...]
            Loss_dc_dxy_val = loss_dc(pdx, phasedx) + loss_dc(pdy, phasedy)
            Loss = Loss_dc_val+Loss_dc_dxy_val*opt['gradientDC_coff_unwrap']
        else:
            Loss = Loss_dc_val


        optim.zero_grad()
        Loss.backward(retain_graph=True)
        optim.step()
        if opt['show_flag_unwrap'] and ii % opt['show_gap_num_unwrap'] == 0:
            IM = torch.squeeze(Phi_U[:, :, 0, 0, 0, 0]).clone().detach().cpu().numpy()
            IM2 = torch.squeeze(Phi_U[:, :, 0, 0, 0, 1]).clone().detach().cpu().numpy()
            IM_ = np.angle(U[:, :, 0, 0, 0, 0])
            IM2_ = np.angle(U[:, :, 0, 4, 0, 0])
            im = np.concatenate([IM, IM2], axis=0)
            im2 = np.concatenate([IM_, IM2_], axis=0)
            t = np.concatenate([im, im2], axis=1)


    #%% -1. output

    # phase gradient
    phasedx = phasedx[:, :, :, :, :, 0] + 1j * phasedx[:, :, :, :, :, 1]
    phasedy = phasedy[:, :, :, :, :, 0] + 1j * phasedy[:, :, :, :, :, 1]
    opt['phasedx'] = phasedx.detach().cpu().numpy()
    opt['phasedy'] = phasedy.detach().cpu().numpy()

    # unwrapping
    W = Phi_U[:, :, :, :, :, 0] + 1j * Phi_U[:, :, :, :, :, 1]
    W = W.detach().cpu().numpy()
    opt['W'] = W
    return opt