'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            print("************world-size:", world_size)
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                           pin_memory=True)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    if mode == 'LQ': #Predictor
        from data.LQ_dataset import LQDataset as D
        dataset = D(dataset_opt)
    elif mode == 'LQGTker': #SFTMD
        from data.LQGTker_dataset import LQGTKerDataset as D
        dataset = D(dataset_opt)
    elif mode == 'LQGTker_Depth': #SFTMD
        from data.LQGTker_Depth_dataset import LQGTKerDepthDataset as D
        dataset = D(dataset_opt)
    elif mode == 'EndoScene_Depth': #SFTMD
        from data.EndoScene_depth_dataset import EndoSceneDepthDataset as D
    elif mode == 'EndoScene_Depthnew': #SFTMD
        from data.EndoScene_depthnew_dataset import EndoSceneDepthnewDataset as D
        dataset = D(dataset_opt)
    elif mode == 'SRker': #Corrector
        from data.SRker_dataset import SRkerDataset as D
        dataset = D(dataset_opt)
    # elif mode == 'LQGTseg_bg':
    #     from data.LQGT_seg_bg_dataset import LQGTSeg_BG_Dataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
