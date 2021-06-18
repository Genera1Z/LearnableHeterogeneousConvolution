import os
import pickle
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras.models as KM

from src.layer import LhcProbe

print(KM)


def std_count_shapes(kmdl, ratio=0.1):
    for layer in kmdl.layers:
        if hasattr(layer, 'kernel'):
            kernels = layer.kernel
            kernels_bin = np.where(kernels > np.max(kernels, (0, 1, 2), keepdims=True) * ratio, 1, 0)
            nshape_tuple = LhcProbe.count_shapes(kernels_bin)

            if nshape_tuple is None:
                continue

            print(nshape_tuple)
            # XXX arrange in sparsity-ascending order
            #   try differen scales: (0,1)/(0,1,2)/(:), namely, slice/kernel/kernels
            #   try different ratios: 0.01~0.1;
            plt.plot(np.arange(len(nshape_tuple[1])), nshape_tuple[1], marker='.')
            plt.show()


def lhc_count_shapes(kmdl, save_dir):
    nwght_ttl_all, nwght_vld_all = list(), list()
    ncmpt_ttl_all, ncmpt_vld_all = list(), list()
    log_strs = list()
    for cnt, layer in enumerate(kmdl.layers):
        if hasattr(layer, 'masks'):
            masks = LhcProbe.calc_masks(layer.effects, layer.ci, layer.co, layer.cgi, layer.cgo)

            nshape_list = LhcProbe.count_shapes(masks)
            pickle.dump(nshape_list, open(f'{save_dir}layer{cnt}.pkl', 'wb'))
            # shape_ratios = np.array(nshape_list) / np.sum(nshape_list)
            # print('shape_ratios:', shape_ratios)

            plt.plot(nshape_list)
            plt.grid()
            # plt.show()
            plt.savefig(f'{save_dir}layer{cnt}.png')

            nw_ttl, nw_vld = LhcProbe.count_weights(masks)
            nwght_ttl_all.append(nw_ttl)
            nwght_vld_all.append(nw_vld)

            nc_ttl = np.prod(layer.output.shape[1:3]) * nw_ttl
            nc_vld = np.prod(layer.output.shape[1:3]) * nw_vld
            ncmpt_ttl_all.append(nc_ttl)
            ncmpt_vld_all.append(nc_vld)

            log_strs.append(f'nw: {nw_ttl}, {nw_vld}, {nw_vld / nw_ttl}')
            log_strs.append(f'nc: {nc_ttl}, {nc_vld}, {nc_vld / nc_ttl}')

    nw_ttl_sum = sum(nwght_ttl_all)
    nw_vld_sum = sum(nwght_vld_all)
    log_strs.append(f'total nw: {nw_ttl_sum}, {nw_vld_sum}, {nw_vld_sum / nw_ttl_sum}')

    nc_ttl_sum = sum(ncmpt_ttl_all)
    nc_vld_sum = sum(ncmpt_vld_all)
    log_strs.append(f'total nc: {nc_ttl_sum}, {nc_vld_sum}, {nc_vld_sum / nc_ttl_sum}')

    with open(f'{save_dir}sparsity.txt', 'w') as f:
        f.writelines(log_strs)

    pprint(log_strs)


def lhc_calc_mask_correlations(kmdl, fold, save_dir):
    def correlation(masks1, masks2):
        return np.mean(np.logical_and(masks1, masks2))

    ckpt_files = [fold + _ for _ in os.listdir(fold)]
    ckpt_files.sort()
    values_all = list()

    masks_all1 = None
    for ckpt_file in ckpt_files:
        print(f'Processing {ckpt_file}...')
        kmdl.load_weights(ckpt_file)
        masks_all2 = list()

        for layer in kmdl.layers:
            if hasattr(layer, 'masks'):
                assert layer.formable
                masks = LhcProbe.calc_masks(layer.effects, layer.ci, layer.co, layer.cgi, layer.cgo)
                masks_all2.append(masks)

        if masks_all1 is None:
            masks_all1 = masks_all2
        else:
            values = [correlation(m1, m2) for m1, m2 in zip(masks_all1, masks_all2)]
            values_all.append(values)
            masks_all1 = masks_all2

    values_all = np.array(values_all)
    pd.DataFrame(values_all).to_csv(f'{save_dir}correlation.csv')
    pprint(values_all)

    [plt.plot(values_all[:, _], label=f'{_}') for _ in range(values_all.shape[1])]
    plt.grid()
    plt.legend()
    # plt.show()
    plt.savefig(f'{save_dir}correlation.png')
