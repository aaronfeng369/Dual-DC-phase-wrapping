## Dual-DC phase Unwrapping



1. Dependent environment （dual-dc.yaml）

2. Dual-DC pytorch function  for 2D MRE phase unwrapping (phase_wrapping.py)

    Contains 5 parameters:
    '''
    % path_file: file path (Dual-DC and raw MRE data) (end with /)
    % name_: data name (U_<name>.mat)
    % unwrpping_:[lr_unwrap,iter_max_unwrap,gradientDC_coff_unwrap]
    %     lr_unwrap: 0.005,
    %     iter_max_unwrap:4000,
    %     gradientDC_coff_unwrap:1000
    '''

    Input file:  U_{dataset name}.mat

    Contains three variables

    Freq_list:  list of Frequency(Hz), Such as [30,40,50,60]

    U:    6D-complex wavefiled. (x,y,z,timepoint,MEG encoding direction,Frequency)

    dxyz: [dx,dy,dz] (m)

    Phase_point: Phase offset for each sample time (rad, +,-means the MEG polarity),list, double

    Output file : W_{dataset name}.mat

    Contains one variable:

    W: (x,y,z,MEG encoding direction,Frequency) unwrapped complexed displacement
    (If  W_{dataset name}.mat already exists, it will not be calculated again.)

3. demo (Dual_DC.py)

4. show results (show_phase.m)

5. Demo dataset (U_0.4.mat, U_p0.mat) 



Note: There may be a little difference between the version and the paper of Dual-DC. If you have any questions, please contact me.
By Shengyuan Ma (shengyuanma@sjtu.edu.cn,fengyuan@sjtu.edu.cn)


Cite: Ma, S.; Wang, R.; Qiu, S.; Li, R.; Yue, Q.; Sun, Q.; Chen, L.; Yan, F.; Yang, G.-Z.; Feng, Y. MR Elastography With Optimization-Based Phase Unwrapping and Traveling Wave Expansion-Based Neural Network (TWENN). IEEE Trans. Med. Imaging 2023, 42 (9), 2631–2642. https://doi.org/10.1109/TMI.2023.3261346.
