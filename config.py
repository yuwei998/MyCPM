class FLAGS(object):
    """ """
    """
    General settings
    """
    input_size = 368
    heatmap_size = 46
    cpm_stages = 3
    joint_gaussian_variance = 1.0
    center_radius = 21
    num_of_joints = 13
    color_channel = 'RGB'
    normalize_img = True
    use_gpu = True


    """
    Demo settings
    """
    # 'MULTI': show multiple stage heatmaps
    # 'SINGLE': show last stage heatmap
    # 'Joint_HM': show last stage heatmap for each joint
    # 'image or video path': show detection on single image or video
    #DEMO_TYPE = 'SINGLE'

    # model_path = 'cpm_hand'
    # cam_id = 0

    # webcam_height = 480
    # webcam_width = 640

    # use_kalman = True
    # kalman_noise = 0.03


    """
    Training settings
    """
    category='blouse'
    network_def = 'cpm_hand'
    train_img_dir = './data'
    val_img_dir = './data'
    result_dir='./result/blouse/train'
    # bg_img_dir = ''
    pretrained_model = ''
    batch_size = 16
    init_lr = 0.001
    lr_decay_rate = 0.92
    lr_decay_step = 1000
    training_iters = 30000
    #verbose_iters = 10
    validation_iters = 10
    model_save_iters = 500
    # augmentation_config = {'hue_shift_limit': (-5, 5),
                           # 'sat_shift_limit': (-10, 10),
                           # 'val_shift_limit': (-15, 15),
                           # 'translation_limit': (-0.15, 0.15),
                           # 'scale_limit': (-0.3, 0.5),
                           # 'rotate_limit': (-90, 90)}
    # hnm = True  # Make sure generate hnm files first
    # do_cropping = True

    joint_color_code = [[139, 53, 255],
                    [0, 56, 255],
                    [43, 140, 237],
                    [37, 168, 36],
                    [147, 147, 0],
                    [70, 17, 145]]









